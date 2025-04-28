# Unified train.py (Fully Refactored Version)

# === 0. Imports ===
import os
import time
import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from datasets import load_dataset
from resonantTransformer import EnhancedResonantTransformer, diversity_penalty, cosine_rampup, contrastive_loss
from torch.amp import autocast, GradScaler



# === 1. Utility Functions ===
def soft_project_onto_hypersphere(x, target_radius=1.0, tolerance=0.25, strength=0.1, eps=1e-6):
    norms = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    deviation = (norms - target_radius).abs()
    mask = (deviation > tolerance).float()
    corrected = target_radius * (x / norms)
    return x * (1 - strength * mask) + corrected * (strength * mask)

def hard_project_onto_hypersphere(x, radius=1.0, eps=1e-8):
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return x / norm * radius




# === 2. Environment & Config Setup ===
def prepare_environment():
    from config import (
        baseline, wandb_project_name, d_model, num_heads, num_layers, vocab_size, weight_decay,
        resonant_token_count, dynamic_resonant_token_count, token_learning_amplifier,
        sequence_length, max_tokens, learning_rate, batch_size, num_epochs, warmup_epochs,
        lambda_ri, lambda_rs, lambda_div, lambda_sur, lambda_attn, lambda_res,
        multihead_resonance, max_recursive_steps, recursive_convergence_tolerance,
        flux_penalty_weight, lambda_resonant_attention
    )
    if baseline:
        resonant_token_count = 0
        dynamic_resonant_token_count = 0
        multihead_resonance = False
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
             torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    token_part = "BASE" if baseline else f"{resonant_token_count}-{dynamic_resonant_token_count}"
    millions = int(max_tokens / 1_000_000)
    run_name = f"{token_part}-{d_model}-{num_heads}-{num_layers}-{sequence_length}-{millions}M"
    config_dict = {
        "baseline": baseline, "d_model": d_model, "num_heads": num_heads, "num_layers": num_layers,
        "vocab_size": vocab_size, "weight_decay": weight_decay, "resonant_token_count": resonant_token_count,
        "dynamic_resonant_token_count": dynamic_resonant_token_count, "token_learning_amplifier": token_learning_amplifier,
        "sequence_length": sequence_length, "max_tokens": max_tokens, "learning_rate": learning_rate,
        "batch_size": batch_size, "num_epochs": num_epochs, "warmup_epochs": warmup_epochs,
        "lambda_ri": lambda_ri, "lambda_rs": lambda_rs, "lambda_div": lambda_div,
        "lambda_sur": lambda_sur, "lambda_attn": lambda_attn, "lambda_res": lambda_res,
        "multihead_resonance": multihead_resonance, "max_recursive_steps": max_recursive_steps,
        "recursive_convergence_tolerance": recursive_convergence_tolerance,
        "flux_penalty_weight": flux_penalty_weight, "lambda_resonant_attention": lambda_resonant_attention
    }
    wandb.init(project=wandb_project_name, name=run_name, config=config_dict)
    return DEVICE, config_dict



# === 3. Dataset Preparation ===
def prepare_dataset(corpus_path="tinystories_cached.txt", max_stories=500_000):
    print("Prep Data")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    if not os.path.exists(corpus_path):
        with open(corpus_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                line = entry['text'].strip()
                if line:
                    f.write(line + "\n")
    return corpus_path



# === 4. Tokenizer Preparation ===
def setup_tokenizer(corpus_path, tokenizer_dir="tokenizer-tinystories", vocab_size=50257):
    print("Tokenizer Setup")
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")
    if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
        print("[Tokenizer] Training new tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[corpus_path], vocab_size=vocab_size, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_model(tokenizer_dir)
    else:
        print("[Tokenizer] Loading existing tokenizer...")
        tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
    tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.post_processor = BertProcessing(("<pad>", pad_id), ("<pad>", pad_id))
    return tokenizer




# === 5. Corpus Encoding ===
def encode_corpus(corpus_path, tokenizer, sequence_length=256, max_tokens=10_000_000):
    print("Encode Corpus")
    tokens = []
    buffer = []
    stride = sequence_length // 3
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    max_stories = 500_000
    story_count = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            story_count += 1
            if story_count >= max_stories:
                break
            line = line.strip()
            if line:
                encoded = tokenizer.encode(line)
                story_tokens = [bos_id] + encoded.ids + [eos_id]
                buffer.extend(story_tokens)
                while len(buffer) >= sequence_length:
                    chunk = buffer[:sequence_length]
                    tokens.append(chunk)
                    buffer = buffer[stride:]
                if len(tokens) * sequence_length >= max_tokens:
                    break
    if len(buffer) >= sequence_length // 2:
        pad_id = tokenizer.token_to_id("<pad>")
        padded = buffer + [pad_id] * (sequence_length - len(buffer))
        tokens.append(padded[:sequence_length])
    return tokens

# === 6. DataLoader Creation ===
def create_dataloader(all_sequences, tokenizer, batch_size=32):
    def collate_dynamic(batch):
        lengths = [len(x) for x in batch]
        max_len = max(lengths)
        pad_id = tokenizer.token_to_id("<pad>")
        padded = [x + [pad_id] * (max_len - len(x)) for x in batch]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
    print("Build Tensor From Chunks")
    loader = DataLoader(
        all_sequences, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_dynamic
    )
    return loader




# === 7. Model Building ===
def build_model(config, tokenizer, device):
    print("Build Model")
    model = EnhancedResonantTransformer(
        baseline=config["baseline"],
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        resonant_token_count=config["resonant_token_count"],
        dynamic_resonant_token_count=config["dynamic_resonant_token_count"],
        multihead=config["multihead_resonance"],
        max_recursive_steps=config["max_recursive_steps"]
    ).to(device)
    model.train()
    torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    attn_records = []

    attn_momentum = 0.99
    def attn_hook(module, inp, output):
        if isinstance(output, tuple) and output[1] is not None:
            attn_out = output[1].detach()
            if not hasattr(model, "avg_attn"):
                model.avg_attn = attn_out.mean(0)
            else:
                model.avg_attn = attn_momentum * model.avg_attn + (1 - attn_momentum) * attn_out.mean(0)
            attn_records.append(attn_out)
            if len(attn_records) > max(10, int(0.5 * config["num_layers"] * config["num_heads"])):
                del attn_records[0]

    if not config["baseline"] and config["lambda_attn"] != 0:
        for layer in model.encoder_layers:
            layer.register_forward_hook(attn_hook)

    return model, scaler, attn_records

# === 8. Optimizer and Scheduler Setup ===
def setup_optimizer_and_scheduler(model, config, dataset_size, device_type="cuda"):
    print("Setup Optimizer and Scheduler")
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    baseline = config["baseline"]

    if not baseline:
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters()
                        if all(x not in n for x in ['resonant_tokens', 'controller', 'resonator'])],
             'lr': learning_rate}
        ])
    else:
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters()
                        if 'resonant_tokens' not in n and 'controller' not in n and 'resonator' not in n],
             'lr': learning_rate, 'weight_decay': weight_decay}
        ])

    steps_per_epoch = dataset_size // batch_size
    total_training_steps = steps_per_epoch * num_epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_training_steps, eta_min=1e-5
    )

    return optimizer, scheduler




# === 9. Loss and Logging Helpers ===
def compute_losses(model, logits, tgt, inp, ctx, res, config, device, epoch, batch_idx,
                   skip_counts, attn_records, sur_baseline, res_baseline,
                   hidden_contrastive_loss, flux_penalty):
    crit = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.token_to_id("<pad>") if hasattr(model, "tokenizer") else 0)
    primary = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

    con = penalty = sur_reward = akl = res_reward = dvt = torch.tensor(0.0, device=device)

    if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None and model._res_tokens_for_ri.numel() > 0:
        dvt = diversity_penalty(model._res_tokens_for_ri)

    if not config["baseline"]:
        # === Create dummy dependency for resonant tokens ===
        if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
            primary = primary + 0.0 * model._res_tokens_for_ri.sum()

        lp = F.log_softmax(logits, dim=-1)
        tlp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        spr = -tlp
        dr = spr[:, :-1] - spr[:, 1:]
        sur = F.relu(dr).mean() / inp.size(1)

        if torch.isnan(sur) or torch.isinf(sur):
            sur = torch.tensor(0.0, device=device)
            skip_counts["surprisal_loss"] += 1

        sur = torch.clamp(sur, max=0.5)
        if sur_baseline is None:
            sur_baseline = sur.detach()
        else:
            sur_baseline = 0.99 * sur_baseline + 0.01 * sur.detach()

        sur_reward = sur - sur_baseline
        primary = primary - config["lambda_sur"] * sur_reward

        if attn_records and hasattr(model, "avg_attn"):
            ca = torch.stack(attn_records).mean(0)
            cur, avg = ca.mean(0) + 1e-8, model.avg_attn + 1e-8
            akl = F.kl_div(cur.log(), avg, reduction='batchmean')
            akl = torch.clamp(akl, max=0.1)
            primary = primary - config["lambda_attn"] * akl
            model.avg_attn = avg
            del attn_records[:]

        if logits.size(1) >= 2:
            pr = lp.exp().clamp(min=1e-5, max=1-1e-5)
            ent = -(pr * pr.log()).sum(-1)
            pen, pos = ent[:, -2], ent[:, -1]
            rr = F.relu(pen - pos).mean()
            rr = torch.clamp(rr, max=0.5)

            if res_baseline is None:
                res_baseline = rr.detach()
            else:
                res_baseline = 0.99 * res_baseline + 0.01 * rr.detach()

            res_reward = rr - res_baseline
            primary = primary - config["lambda_res"] * res_reward

        ci = inp.clone()
        ci[:, -1] = torch.randint(0, logits.size(-1), (inp.size(0),), device=device)
        with torch.no_grad():
            contrast_logits, _, _, _ = model.recursive_forward(ci, ci, tol=config["recursive_convergence_tolerance"])
            contrastive_loss_val = contrastive_loss(logits, contrast_logits)
            min_contrastive_loss = 1e-4
            if contrastive_loss_val.item() < min_contrastive_loss:
                rescue_boost = (min_contrastive_loss - contrastive_loss_val.item()) * 10.0
                contrastive_loss_val = contrastive_loss_val + rescue_boost
            con = contrastive_loss_val

        lp_ent = F.log_softmax(logits, dim=-1)
        pr_ent = lp_ent.exp().clamp(min=1e-5, max=1-1e-5)
        lp_ent = torch.log(pr_ent)
        ent_loss = -(pr_ent * lp_ent).sum(-1).mean()
        primary = primary - 1e-4 * ent_loss

        if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
            gr_tuple = torch.autograd.grad(primary, model._res_tokens_for_ri, retain_graph=True, create_graph=True, allow_unused=True)
            gr = gr_tuple[0]
            if gr is not None:
                gn = gr.norm(dim=-1).clamp(min=1e-6)
                penalty = (
                    config["lambda_ri"] * (gr * model._res_tokens_for_ri).sum(dim=-1).abs().mean() / gn.mean()
                    + config["lambda_rs"] * (1 - F.cosine_similarity(gr, model._res_tokens_for_ri, dim=-1)).mean()
                    + config["lambda_div"] * diversity_penalty(model._res_tokens_for_ri)
                )

        primary = primary - penalty
        primary = primary + 0.05 * hidden_contrastive_loss
        primary = primary + config["flux_penalty_weight"] * flux_penalty

    return primary, con, penalty, sur_reward, akl, res_reward, sur_baseline, res_baseline




# === 10. Logging ===
def log_metrics(model, primary, con, dvt, sur_reward, akl, res_reward, ctx, epoch, batch_idx, loader_len, config, skip_counts, global_repair_counter):
    import numpy as np
    now = time.time()
    if not hasattr(log_metrics, "_last_log_time"):
        log_metrics._last_log_time = now
    delta_s = now - log_metrics._last_log_time
    log_metrics._last_log_time = now

    global_step = epoch * loader_len + batch_idx
    ppl = float(np.exp(primary.item())) if primary.item() < 100 else float('inf')

    res_token_norm = 0.0
    stat_norm = 0.0
    dyn_norm = 0.0
    mh_norm = 0.0
    excess_flux_count = getattr(model, "excess_flux_count", 0)
    loss_reward_skips = skip_counts["entropy_loss"] + skip_counts["surprisal_loss"] + skip_counts["resolution_loss"]
    vector_repairs = sum(global_repair_counter.values())
    self_attn_mean = model.attention_trajectory[-1] if hasattr(model, "attention_trajectory") and model.attention_trajectory else 0.0
    recursive_steps = getattr(model, "last_recursive_steps", 0)
    last_flux_cost = getattr(model, "last_flux_cost", 0.0)
    flux_budget = getattr(model, "flux_budget", 0.0)


    if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
        res_token_norm = model._res_tokens_for_ri.norm().item()
    if hasattr(model, 'resonant_tokens') and model.resonant_tokens is not None:
        stat_norm = model.resonant_tokens.norm().item()

    ctx_vec = None
    if ctx is not None:
        if ctx.dtype in (torch.int32, torch.int64):
            if hasattr(model, 'embedding'):
                ctx_emb = model.embedding(ctx)
                ctx_vec = ctx_emb.mean(dim=1)
        else:
            ctx_vec = ctx.mean(dim=1)

    if ctx_vec is not None:
        if hasattr(model, 'controller') and model.controller is not None:
            dyn_norm = model.controller(ctx_vec).norm().item()
        if hasattr(model, 'resonator') and model.resonator is not None:
            mh_norm = model.resonator(ctx_vec).norm().item()


    wandb.log({
        'loss': primary.item(),
        'perplexity': ppl,
        'surprisal_reward': sur_reward.item() if sur_reward is not None else 0.0,
        'attention_kl': akl.item() if akl is not None else 0.0,
        'resolution_score': res_reward.item() if res_reward is not None else 0.0,
        'res_token_norm': res_token_norm,
        'static_res_norm': stat_norm,
        'dynamic_res_norm': dyn_norm,
        'multihead_res_norm': mh_norm,
        'contrastive_loss': con.item(),
        'diversity_penalty': dvt.item(),
        'excess_flux_count': excess_flux_count,
        'entropy_loss_skips': skip_counts["entropy_loss"],
        'surprisal_loss_skips': skip_counts["surprisal_loss"],
        'resolution_loss_skips': skip_counts["resolution_loss"],
        'repairs_logits': global_repair_counter["logits"],
        'repairs_resonant_output': global_repair_counter["resonant output"],
        'repairs_hidden_state': global_repair_counter["hidden state"],
        'repairs_context_vector': global_repair_counter["context vector"],

        'self_attn_mean': self_attn_mean,
        'recursive_steps': recursive_steps,
        'last_flux_cost': last_flux_cost,
        'flux_budget': flux_budget,
    }, step=global_step)

    print(f"{delta_s:.1f}: {global_step} | {epoch+1} | {batch_idx} - PPL: {ppl:.2f} | NORM: {res_token_norm:.2f} | "
          f"SUR: {sur_reward.item() if sur_reward is not None else 0.0:.4f} | AKL: {akl.item() if akl is not None else 0.0:.4f} | "
          f"RES: {res_reward.item() if res_reward is not None else 0.0:.4f} | "
          f"RI: 0.0 | RSC: 0.0 | SKIPS: {loss_reward_skips} | EFC: {excess_flux_count} | REPAIRS: {vector_repairs}")




# === 11. Training Loop ===
def train(model, loader, opt, scheduler, config, device, tokenizer):
    print("Start Training")
    last_res = None
    sur_baseline = None
    res_baseline = None
    global_repair_counter = {"logits": 0, "resonant output": 0, "hidden state": 0, "context vector": 0}
    skip_counts = {"entropy_loss": 0, "surprisal_loss": 0, "resolution_loss": 0}
    attn_records = []
    for epoch in range(config["num_epochs"]):
        last_res, sur_baseline, res_baseline = train_epoch(
            model, loader, opt, scheduler, config, device, epoch,
            last_res, sur_baseline, res_baseline,
            global_repair_counter, skip_counts, attn_records,
            tokenizer
        )
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, opt, config, epoch)
    save_final_model(model, config, last_res)


def train_epoch(model, loader, opt, scheduler, config, device, 
                epoch, last_res, sur_baseline, res_baseline, 
                global_repair_counter, skip_counts, attn_records, 
                tokenizer):
    
    print(f"Start Epoch {epoch+1}/{config['num_epochs']}")
    model.train()
    model.alpha = cosine_rampup(epoch, config["warmup_epochs"])
    
    if epoch >= config["warmup_epochs"] and hasattr(model, "max_recursive_steps") and model.max_recursive_steps < config["max_recursive_steps"]:
        model.max_recursive_steps += 1

    if epoch == config["warmup_epochs"] and hasattr(model, 'resonant_attention_scale'):
        model.resonant_attention_scale *= 2
    
    if epoch == config["warmup_epochs"]:
        print(f"=== Warmup complete at epoch {epoch}: enabling resonant parameters and rebuilding optimizer ===")
        
        # Enable grads for resonant-related parameters
        for name, param in model.named_parameters():
            param.requires_grad_(True)

        # Rebuild optimizer with resonant token params at boosted LR
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters()
                           if all(x not in n for x in ['resonant_tokens', 'controller', 'resonator'])],
                'lr': config["learning_rate"]
            }
        ]
        if config["resonant_token_count"] > 0:
            param_groups.append({
                'params': [model.resonant_tokens],
                'lr': config["learning_rate"] * config["token_learning_amplifier"]
            })
        if config["dynamic_resonant_token_count"] > 0:
            param_groups.append({
                'params': model.controller.parameters(),
                'lr': config["learning_rate"] * config["token_learning_amplifier"]
            })
        if config["multihead_resonance"]:
            param_groups.append({
                'params': model.resonator.parameters(),
                'lr': config["learning_rate"] * config["token_learning_amplifier"]
            })

        opt.param_groups.clear()
        opt.add_param_group(param_groups[0])
        if len(param_groups) > 1:
            for group in param_groups[1:]:
                opt.add_param_group(group)

        if hasattr(model, 'resonant_attention_scale'):
            model.resonant_attention_scale *= 2

        # Clamp resonant tokens cleanly after enabling
        if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
            with torch.no_grad():
                model._res_tokens_for_ri.clamp_(-1.0, 1.0)
                model._res_tokens_for_ri.requires_grad_(True)

    for batch_idx, (batch, lengths) in enumerate(loader):
        last_res, sur_baseline, res_baseline = train_batch(
            model, batch, lengths, opt, scheduler, config, device, epoch, batch_idx,
            last_res, sur_baseline, res_baseline,
            global_repair_counter, skip_counts, attn_records,
            loader_len=len(loader),
            tokenizer=tokenizer
        )
    return last_res, sur_baseline, res_baseline



def train_batch(model, batch, lengths, opt, scheduler, config, device, epoch, batch_idx,
                last_res, sur_baseline, res_baseline, global_repair_counter, skip_counts,
                attn_records, loader_len, tokenizer):

    model.excess_flux_count = 0

    inp = batch[:, :-1].to(device)
    tgt = batch[:, 1:].to(device)

    ctx = last_res if last_res is not None else inp
    padding_mask = (inp == tokenizer.token_to_id("<pad>")).to(device)

    # === Calculate smooth fade_in_strength for recursive steps ===
    if epoch < config["warmup_epochs"]:
        # During warmup: use global cosine rampup
        fade_in_strength = cosine_rampup(epoch + batch_idx / len(loader), config["warmup_epochs"])
    else:
        # After warmup: each extra step gets smooth fade
        base_steps = 2  # how many steps were present at warmup
        added_steps = model.max_recursive_steps - base_steps
        if added_steps <= 0:
            fade_in_strength = 1.0
        else:
            # How much time has passed since warmup (in epochs)
            epochs_since_warmup = (epoch - config["warmup_epochs"]) + (batch_idx / len(loader))
            total_fade_time = added_steps * 5.0  # each step gets 5 epochs to fade
            fade_in_strength = min(1.0, max(0.0, epochs_since_warmup / total_fade_time))

    logits, res, flux_penalty, hidden_contrastive_loss = model.recursive_forward(
        inp, ctx,
        tol=config["recursive_convergence_tolerance"],
        padding_mask=padding_mask,
        fade_in_strength=fade_in_strength
    )

    if batch_idx % 10 == 0:
        global_step = epoch * len(loader) + batch_idx
        wandb.log({
            'fade_in_strength': fade_in_strength,
        }, step=global_step)

    # Immediately after logits are produced: clamp large values
    if logits.abs().max() > 10.0:
        print("[logits_stabilize] Warning: clamping logits to [-10, 10].")
        logits = torch.clamp(logits, min=-10.0, max=10.0)

    # === Immediately after forward pass, repair resonant tokens ===
    if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
        with torch.no_grad():
            # Repair any NaNs
            mask = torch.isnan(model._res_tokens_for_ri)
            if mask.any():
                print("[repair] NaNs detected in resonant tokens after forward pass. Replacing...")
                model._res_tokens_for_ri.data[mask] = 0.0

            # Soft project onto hypersphere
            model._res_tokens_for_ri.data = hard_project_onto_hypersphere(
                model._res_tokens_for_ri.data,
                radius=1.0
            )

    if res is not None:
        with torch.no_grad():
            # Repair NaNs in recursive output
            mask = torch.isnan(res)
            if mask.any():
                print("[repair] NaNs detected in recursive output. Clamping...")
                res.data[mask] = 0.0

            # Clamp large values
            if res.abs().max() > 10.0:
                print("[stabilize] Large values detected in recursive output. Clamping to [-10,10].")
                res.clamp_(min=-10.0, max=10.0)

    # strip off resonant tokens
    logits = logits[:, :inp.size(1)]

    primary, con, dvt, sur_reward, akl, res_reward, sur_baseline, res_baseline = compute_losses(
        model, logits, tgt, inp, ctx, res, config, device, epoch, batch_idx,
        skip_counts, attn_records, sur_baseline, res_baseline,
        hidden_contrastive_loss, flux_penalty
    )

    opt.zero_grad()
    primary.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    if batch_idx % 10 == 0:
        log_metrics(model, primary, con, dvt, sur_reward, akl, res_reward, 
                    ctx, epoch, batch_idx, loader_len, config, skip_counts, 
                    global_repair_counter)

    if scheduler is not None:
        scheduler.step()

    last_res = res

    # === Every 100 batches: extra inner loop for resonant tokens ===
    if epoch >= config["warmup_epochs"] and batch_idx % 100 == 0 and (config["resonant_token_count"] + config["dynamic_resonant_token_count"]) > 0:
        print("[inner-loop] Resonant token refinement...")

        # Freeze everything
        for p in model.parameters():
            p.requires_grad_(False)

        # Unfreeze resonant token parts
        inner_res_params = []
        if hasattr(model, 'resonant_tokens'):
            inner_res_params.append(model.resonant_tokens)
        if hasattr(model, 'controller'):
            inner_res_params.extend(model.controller.parameters())
        if hasattr(model, 'resonator'):
            inner_res_params.extend(model.resonator.parameters())
        for p in inner_res_params:
            p.requires_grad_(True)

        for _ in range(5):
            logits_inner, res_inner, _, _ = model.recursive_forward(
                inp, ctx, tol=config["recursive_convergence_tolerance"], padding_mask=padding_mask
            )

            if logits_inner is not None:
                logits_inner = logits_inner[:, :inp.size(1)]

                crit = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.token_to_id("<pad>") if hasattr(model, "tokenizer") else 0)
                primary_inner = crit(logits_inner.reshape(-1, logits_inner.size(-1)), tgt.reshape(-1))

                # Compute RI/RS/diversity on resonant tokens
                token_loss = torch.tensor(0.0, device=device)
                if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
                    grad_tuple = torch.autograd.grad(primary_inner, model._res_tokens_for_ri, retain_graph=False, create_graph=False, allow_unused=True)
                    gr_inner = grad_tuple[0]
                    if gr_inner is None:
                        gr_inner = torch.zeros_like(model._res_tokens_for_ri)
                    ri_v_inner = (gr_inner * model._res_tokens_for_ri).sum(dim=-1).abs()
                    rs_v_inner = 1 - F.cosine_similarity(gr_inner, model._res_tokens_for_ri, dim=-1)
                    dvt_inner = diversity_penalty(model._res_tokens_for_ri)
                    token_loss = (
                        config["lambda_ri"] * ri_v_inner.mean() +
                        config["lambda_rs"] * rs_v_inner.mean() +
                        config["lambda_div"] * dvt_inner
                    )

                opt.zero_grad()
                token_loss.backward()
                opt.step()

                # Clamp resonant tokens again after update
                if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
                    with torch.no_grad():
                        model._res_tokens_for_ri.clamp_(-5.0, 5.0)
                        model._res_tokens_for_ri.requires_grad_(True)

        # Unfreeze all parameters again
        for p in model.parameters():
            p.requires_grad_(True)

    return last_res, sur_baseline, res_baseline





# === 12. Checkpoint Saving ===
def save_checkpoint(model, opt, config, epoch, filename_prefix="training-checkpoint-epoch"):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'epoch': epoch + 1,
        'config': config,
        'final_resonant_state': getattr(model, '_res_tokens_for_ri', None)
    }
    filename = f"{filename_prefix}{epoch+1}.pt"
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename} at end of epoch {epoch+1}")

def save_final_model(model, config, last_res, filename_prefix="final-model"):
    millions = int(config["max_tokens"] / 1_000_000)
    token_part = "BASE" if config["baseline"] else f"{config['resonant_token_count']}-{config['dynamic_resonant_token_count']}"
    model_filename = f"{token_part}-{config['d_model']}-{config['num_heads']}-{config['num_layers']}-{config['sequence_length']}-{millions}M.pt"
    state = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_resonant_state': last_res
    }
    torch.save(state, model_filename)
    print(f"Model saved to {model_filename} with config and final dynamic resonant state embedded.")




# === 13. Main Driver ===
if __name__ == "__main__":
    DEVICE, config = prepare_environment()
    corpus_path    = prepare_dataset()
    tokenizer      = setup_tokenizer(corpus_path, vocab_size=config["vocab_size"])
    all_sequences  = encode_corpus(corpus_path, tokenizer, sequence_length=config["sequence_length"], max_tokens=config["max_tokens"])
    loader         = create_dataloader(all_sequences, tokenizer, batch_size=config["batch_size"])
    
    model, scaler, attn_records = build_model(config, tokenizer, DEVICE)
    opt, scheduler              = setup_optimizer_and_scheduler(model, config, dataset_size=len(all_sequences), device_type=DEVICE.type)
    
    train(model, loader, opt, scheduler, config, DEVICE, tokenizer)
