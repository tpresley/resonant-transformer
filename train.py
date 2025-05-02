# Unified train.py (Fully Refactored Version)

# === 0. Imports ===
import os
import time
import random
import math
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
        sequence_length, max_tokens, learning_rate, batch_size, num_epochs, 
        lr_warmup_epochs, embedding_dropout, label_smoothing, validation_split, 
        early_stopping_patience, warmup_epochs,
        lambda_ri, lambda_rs, lambda_div, lambda_sur, lambda_attn, lambda_res,
        multihead_resonance, max_recursive_steps, recursive_convergence_tolerance,
        flux_penalty_weight, contrastive_margin, lambda_contrastive,
        lambda_dyn_var, lambda_head_entropy, lambda_inner_align
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
        "embedding_dropout": embedding_dropout, "validation_split": validation_split, "early_stopping_patience": early_stopping_patience,
        "batch_size": batch_size, "num_epochs": num_epochs, "lr_warmup_epochs": lr_warmup_epochs, 
        "label_smoothing": label_smoothing, "warmup_epochs": warmup_epochs,
        "lambda_ri": lambda_ri, "lambda_rs": lambda_rs, "lambda_div": lambda_div,
        "lambda_sur": lambda_sur, "lambda_attn": lambda_attn, "lambda_res": lambda_res,
        "multihead_resonance": multihead_resonance, "max_recursive_steps": max_recursive_steps,
        "recursive_convergence_tolerance": recursive_convergence_tolerance,
        "flux_penalty_weight": flux_penalty_weight, "contrastive_margin": contrastive_margin,
        "lambda_contrastive": lambda_contrastive, "lambda_dyn_var": lambda_dyn_var, 
        "lambda_head_entropy": lambda_head_entropy, "lambda_inner_align": lambda_inner_align
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
        max_recursive_steps=config["max_recursive_steps"],
        embedding_dropout=config["embedding_dropout"]
    ).to(device)
    model.train()
    torch.autograd.set_detect_anomaly(True)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    # attach AMP scaler to the model so train() can use it
    model.scaler = scaler
    attn_records = []

    attn_momentum = 0.95
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

    model.attn_records = attn_records

    return model, scaler, attn_records

# === 8. Optimizer and Scheduler Setup ===
def setup_optimizer_and_scheduler(model, config, dataset_size, device_type="cuda"):
    print("Setup Optimizer and Scheduler")
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    baseline = config["baseline"]

    # decide which parameter-names to exclude in baseline mode
    exclude_keys = [] if baseline else ['resonant_tokens', 'controller', 'resonator']

    # split params into decay / no_decay
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in exclude_keys):
            continue
        # no weight decay on biases or normalization layers
        if name.endswith('.bias') or 'norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped = [
        { 'params': decay_params,    'lr': learning_rate, 'weight_decay': config['weight_decay'] },
        { 'params': no_decay_params, 'lr': learning_rate, 'weight_decay': 0.0 }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped)

    # total and warmup steps
    steps_per_epoch      = dataset_size // batch_size
    total_training_steps = steps_per_epoch * num_epochs
    warmup_steps         = int(config["lr_warmup_epochs"] * steps_per_epoch)

    # build a single LR‐lambda: linear ramp for warmup, then cosine decay
    import math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linear ramp: 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        # cosine anneal from 1 → 0 over the remaining steps
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler




# === 9. Loss and Logging Helpers ===
def compute_losses(model, logits, tgt, inp, ctx, res, config, device, epoch, batch_idx,
                   skip_counts, attn_records, sur_baseline, res_baseline,
                   hidden_contrastive_loss, flux_penalty):
    
    def ramp_weight(epoch, max_weight, warmup_epochs):
        # Cosine ramp-up: 0 → max_weight over warmup_epochs
        if epoch >= warmup_epochs:
            return max_weight
        return float(max_weight) * 0.5 * (1 - math.cos(math.pi * epoch / warmup_epochs))
    
    # --- Manual masked CE: ignore tgt pads but don't let the model exploit that ---
    pad_id = model.tokenizer.token_to_id("<pad>") if hasattr(model, "tokenizer") else -100
    logits_flat = logits.reshape(-1, logits.size(-1))
    tgt_flat    = tgt.reshape(-1)
    # --- Apply label smoothing to discourage over-confident spikes ---
    ce_per_tok  = F.cross_entropy(
        logits_flat,
        tgt_flat,
        reduction='none',
        label_smoothing=config.get("label_smoothing", 0.0)
    )
    nonpad_mask = (tgt_flat != pad_id).float()
    primary     = (ce_per_tok * nonpad_mask).sum() / (nonpad_mask.sum() + 1e-12)

    # --- Explicit anti-pad penalty (tune lambda_pad in config.py) —
    pad_probs = torch.softmax(logits, dim=-1)[..., pad_id]    # [B, L]
    pad_rate  = pad_probs.mean()                              # scalar in [0,1]
    primary  += config.get('lambda_pad', 1.0) * pad_rate

    con = penalty = sur_reward = akl = res_reward = dvt = torch.tensor(0.0, device=device)
    # Ramp diversity weight from 0 → λ_div over the full training run
    diversity_weight = min(config["lambda_div"] * (epoch / config["num_epochs"]), config["lambda_div"])

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
            sur_baseline = 0.80 * sur_baseline + 0.20 * sur.detach()

        sur_reward = sur - sur_baseline
        lambda_sur = ramp_weight(epoch, config["lambda_sur"], config["warmup_epochs"])
        primary = primary - lambda_sur * sur_reward

        if attn_records and hasattr(model, "avg_attn"):
            eps = 1e-5
            ca = torch.stack(attn_records).mean(0)  # mean across recursion steps
            cur = ca.mean(0)  # mean across batch

            # Softmax normalize to make proper distributions
            cur = F.softmax(cur, dim=-1)
            avg = F.softmax(model.avg_attn, dim=-1)

            # Add epsilon for numerical stability
            cur = cur + eps
            avg = avg + eps

            log_cur = cur.log()
            akl = F.kl_div(log_cur, avg, reduction='batchmean', log_target=False)

            akl = torch.clamp(akl, max=0.1)
            lambda_attn = ramp_weight(epoch, config["lambda_attn"], config["warmup_epochs"])
            primary = primary - lambda_attn * akl

            # Correct moving average update
            momentum = 0.90  # slightly faster
            model.avg_attn = momentum * model.avg_attn + (1.0 - momentum) * ca.detach()

            del attn_records[:]

        if logits.size(1) >= 2:
            # — Stable entropy for resolution penalty (no clamp) —
            # re-use lp = F.log_softmax(logits, dim=-1) from above
            probs = lp.exp()             # [B, L, V] probabilities
            eps   = 1e-5
            mask  = probs > eps          # avoid zero→-inf
            # only accumulate p*log(p) where p > eps
            ent_terms = torch.where(
                mask, 
                probs * lp, 
                torch.zeros_like(probs)
            )                            # [B, L, V]
            ent = -ent_terms.sum(-1)     # [B, L] (stable, no nan)
            pen, pos = ent[:, -2], ent[:, -1]
            rr = F.relu(pen - pos).mean()
            rr = torch.clamp(rr, max=0.5)

            if res_baseline is None:
                res_baseline = rr.detach()
            else:
                res_baseline = 0.80 * res_baseline + 0.20 * rr.detach()

            res_reward = rr - res_baseline
            lambda_res = ramp_weight(epoch, config["lambda_res"], config["warmup_epochs"])
            primary = primary - lambda_res * res_reward

        ci = inp.clone()
        ci[:, -1] = torch.randint(0, logits.size(-1), (inp.size(0),), device=device)
        # ── compute contrastive loss with blown-up margin & heavy weight ──
        with torch.no_grad():
            contrast_logits, _, _, _ = model.recursive_forward(
                ci, ci,
                tol=config["recursive_convergence_tolerance"]
            )
        raw_con = contrastive_loss(
            logits,
            contrast_logits,
            margin=config["contrastive_margin"]
        )
        lambda_contrastive = ramp_weight(epoch, config["lambda_contrastive"], config["warmup_epochs"])
        con = lambda_contrastive * raw_con

        # — Stable entropy penalty via masked log-softmax (avoiding NaNs) —
        # 1) compute log-probs and probs
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = torch.exp(log_probs)
        # 2) build mask for p > ε to skip p=0 entries (which have log p = -inf)
        eps  = 1e-5
        mask = probs > eps
        # 3) per-element entropy contributions only where mask is true
        ent_terms = torch.where(mask,
                                probs * log_probs,      # p * log p
                                torch.zeros_like(probs))
        # 4) sum over vocab and average
        ent_loss = -ent_terms.sum(-1).mean()
        primary = primary - 1e-4 * ent_loss.detach()

        # Manual autograd.grad hack removed – resonant-token penalties
        # will now flow via the inner-loop/backward pass.
        primary = primary - penalty

        primary = primary + 0.05 * hidden_contrastive_loss
        lambda_flux = ramp_weight(epoch, config["flux_penalty_weight"], config["warmup_epochs"])
        primary = primary + lambda_flux * flux_penalty

        # Optional direct regularizers for controller & resonator
        dyn_var_penalty = torch.tensor(0.0, device=device)
        head_entropy_penalty = torch.tensor(0.0, device=device)

        if hasattr(model, 'dynamic_tokens_latest') and model.dynamic_tokens_latest is not None:
            dt = model.dynamic_tokens_latest  # shape: [B, T, D]
            if dt.numel() > 0:
                mean = dt.mean(dim=0, keepdim=True)
                var = ((dt - mean) ** 2).mean()
                dyn_var_penalty = -var  # penalize low variance (maximize var)
        if hasattr(model, 'controller') and hasattr(model.controller, 'linear'):
            # Entropy of selector distribution from MultiHeadResonance
            if hasattr(model, 'resonator') and hasattr(model.resonator, 'selector'):
                if ctx.dtype in (torch.int64, torch.int32):
                    ctx_emb = model.embedding(ctx)
                    context_vec = ctx_emb.mean(dim=1)  # [B, D]
                elif ctx.dim() == 3:
                    context_vec = ctx.mean(dim=1)
                elif ctx.dim() == 2:
                    context_vec = ctx
                else:
                    raise ValueError(f"[head_entropy_penalty] Unexpected ctx shape: {ctx.shape}")

                if context_vec.shape[-1] != model.d_model:
                    raise ValueError(f"[head_entropy_penalty] context_vec shape mismatch: got {context_vec.shape}, expected [B, {model.d_model}]")
                logits = model.resonator.selector(context_vec)  # [B, H]
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()
                head_entropy_penalty = -entropy  # maximize entropy => penalize low entropy
        lambda_dyn_var = config.get("lambda_dyn_var", 0.05)
        lambda_head_entropy = config.get("lambda_head_entropy", 0.01)
        primary = primary + lambda_dyn_var * dyn_var_penalty
        primary = primary + lambda_head_entropy * head_entropy_penalty

    return primary, con, dvt, sur_reward, akl, res_reward, sur_baseline, res_baseline




# === 10. Logging ===
def log_metrics(model, ppl, primary, con, dvt, sur_reward, akl, res_reward, ctx, epoch, batch_idx, loader_len, config, skip_counts, global_repair_counter):
    import numpy as np
    now = time.time()
    if not hasattr(log_metrics, "_last_log_time"):
        log_metrics._last_log_time = now
    delta_s = now - log_metrics._last_log_time
    log_metrics._last_log_time = now

    global_step = epoch * loader_len + batch_idx
    # ppl = float(np.exp(primary.item())) if primary.item() < 100 else float('inf')
    pplc = float(np.exp(primary.item())) if primary.item() < 100 else float('inf')

    res_token_norm = 0.0
    stat_norm = 0.0
    excess_flux_count = getattr(model, "excess_flux_count", 0)
    loss_reward_skips = skip_counts["entropy_loss"] + skip_counts["surprisal_loss"] + skip_counts["resolution_loss"]
    vector_repairs = sum(global_repair_counter.values())

    # === Compute RI / RS / RS_COS from the static bank’s parameter grads ===
    ri_val, rs_val, rs_cos_val = 0.0, 0.0, 0.0
    if hasattr(model, 'resonant_tokens') and model.resonant_tokens.grad is not None:
        # take the parameter tensor and its gradient
        bank = model.resonant_tokens.detach()          # [1, R, D]
        g = model.resonant_tokens.grad                 # same shape
        # flatten token-dimension for dot products
        ri_val = (g * bank).sum(dim=-1).abs().mean().item()
        cos_sim = F.cosine_similarity(g, bank, dim=-1)
        rs_cos_val = cos_sim.mean().item()
        rs_val = (1.0 - cos_sim).mean().item()

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
        # NEW: track the *learnable* parts on controller & multi-head
        dyn_weight_norm = dyn_grad_norm = 0.0
        if hasattr(model, 'controller') and model.controller is not None:
            w = model.controller.linear.delta_weight
            dyn_weight_norm = w.norm().item()
            if w.grad is not None:
                dyn_grad_norm = w.grad.norm().item()

        mh_weight_norm = mh_grad_norm = 0.0
        if hasattr(model, 'resonator') and model.resonator is not None:
            bank = model.resonator.resonant_bank
            mh_weight_norm = bank.norm().item()
            if bank.grad is not None:
                mh_grad_norm = bank.grad.norm().item()


    wandb.log({
        'loss': primary.item(),
        'perplexity': ppl,
        'perplexity_composite': pplc,
        'surprisal_reward': sur_reward.item() if sur_reward is not None else 0.0,
        'attention_kl': akl.item() if akl is not None else 0.0,
        'resolution_score': res_reward.item() if res_reward is not None else 0.0,
        'res_token_norm': res_token_norm,
        'static_res_norm': stat_norm,

        # controller (dynamic) parameter norms & grads
        'dyn_controller_weight_norm': dyn_weight_norm,
        'dyn_controller_grad_norm':   dyn_grad_norm,

        # multihead parameter norms & grads
        'mh_resonator_weight_norm':   mh_weight_norm,
        'mh_resonator_grad_norm':     mh_grad_norm,

        'contrastive_loss': con.item(),
        'diversity_penalty': dvt.item(),
        'excess_flux_count': excess_flux_count,
        'entropy_loss_skips': skip_counts["entropy_loss"],
        'surprisal_loss_skips': skip_counts["surprisal_loss"],
        'resolution_loss_skips': skip_counts["resolution_loss"],
        'ri': ri_val,
        'rs': rs_val,
        'rs_cos': rs_cos_val,
        'repairs_logits': global_repair_counter["logits"],
        'repairs_resonant_output': global_repair_counter["resonant output"],
        'repairs_hidden_state': global_repair_counter["hidden state"],
        'repairs_context_vector': global_repair_counter["context vector"],

        'self_attn_mean': self_attn_mean,
        'recursive_steps': 0 if config["baseline"] else getattr(model, "last_recursive_steps", 1),
        'last_flux_cost': last_flux_cost,
        'flux_budget': flux_budget,
    }, step=global_step)

    print(f"{delta_s:.1f}: {global_step} | {epoch+1} | {batch_idx} - "
          f"PPL: {ppl:.2f} | PPLC: {pplc:.2f} | NORM: {res_token_norm:.2f} | "
          f"SUR: {sur_reward.item():.4f} | AKL: {akl.item():.4f} | "
          f"RES: {res_reward.item():.4f} | "
          f"RI: {ri_val:.4e} | RS: {rs_val:.4e} | RS_COS: {rs_cos_val:.4e} | ")




# === 11. Training Loop ===
def train(model, train_loader, val_loader, opt, scheduler, config, device, tokenizer):
    print("Start Training")
    model.tokenizer = tokenizer
    last_res = None
    sur_baseline = None
    res_baseline = None
    global_repair_counter = {"logits": 0, "resonant output": 0, "hidden state": 0, "context vector": 0}
    skip_counts = {"entropy_loss": 0, "surprisal_loss": 0, "resolution_loss": 0}
    attn_records = getattr(model, "attn_records", [])
    current_recursive_target_steps = 2 if not config["baseline"] else 1
    fade_in_progress = 0.0

    best_val_ppl = float("inf")
    no_improve_epochs = 0
    for epoch in range(config["num_epochs"]):
        last_res, sur_baseline, res_baseline, current_recursive_target_steps, fade_in_progress = train_epoch(
            model, train_loader, opt, scheduler, config, device, epoch,
            last_res, sur_baseline, res_baseline,
            current_recursive_target_steps=current_recursive_target_steps,
            fade_in_progress=fade_in_progress,
            global_repair_counter=global_repair_counter,
            skip_counts=skip_counts,
            attn_records=attn_records,
            tokenizer=tokenizer
        )
        # — every epoch, evaluate on val set —
        model.eval()
        total_ce = 0.0
        total_tokens = 0
        with torch.no_grad(), autocast(device_type=device.type, enabled=(device.type=="cuda")):
            val_last_res = None
            for batch, lengths in val_loader:
                inp = batch[:, :-1].to(device)
                tgt = batch[:, 1:].to(device)
                ctx = val_last_res if val_last_res is not None else inp
                mask = (inp == tokenizer.token_to_id("<pad>")).to(device)
                # — Compute logits exactly as in train_batch —
                if config["baseline"]:
                    # baseline forward returns (logits, …)
                    logits = model.forward(inp, context=ctx, padding_mask=mask)[0]
                else:
                    logits, _, _, _ = model.recursive_forward(
                        inp, ctx,
                        tol=config["recursive_convergence_tolerance"],
                        padding_mask=mask,
                        fade_in_strength=1.0
                    )
                    val_last_res = model._res_tokens_for_ri

                # Strip off the initial self-token
                logits = logits[:, 1 : 1 + inp.size(1), :]
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_tgt    = tgt.reshape(-1)
                nonpad      = (flat_tgt != tokenizer.token_to_id("<pad>"))
                ce_loss     = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
                total_ce   += (ce_loss * nonpad).sum().item()
                total_tokens += nonpad.sum().item()
        val_ppl = math.exp(total_ce / total_tokens)
        print(f"Epoch {epoch+1}: validation perplexity = {val_ppl:.2f}")
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            no_improve_epochs = 0
            save_checkpoint(model, opt, config, epoch,
                            filename_prefix="best-val-checkpoint-epoch")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= config["early_stopping_patience"]:
                print(f"No improvement for {config['early_stopping_patience']} epochs; stopping early.")
                break
        model.train()
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, opt, config, epoch)
    save_final_model(model, config, last_res)


def train_epoch(model, loader, opt, scheduler, config, device, epoch, last_res, sur_baseline, res_baseline, global_repair_counter, skip_counts, attn_records,
                current_recursive_target_steps, fade_in_progress, tokenizer):
    
    print(f"Start Epoch {epoch+1}/{config['num_epochs']}")

    if epoch == 0 and not config["baseline"]:
        model.max_recursive_steps = 2

    model.train()
    model.alpha = cosine_rampup(epoch, config["warmup_epochs"])
    
    if not config["baseline"]:
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
                model._res_tokens_for_ri = model._res_tokens_for_ri.clamp(-1.0, 1.0)

    for batch_idx, (batch, lengths) in enumerate(loader):
        last_res, sur_baseline, res_baseline, fade_in_progress, current_recursive_target_steps = train_batch(
            model, batch, lengths, opt, scheduler, config, device, epoch, batch_idx,
            last_res, sur_baseline, res_baseline,
            global_repair_counter, skip_counts, attn_records,
            fade_in_progress=fade_in_progress,
            current_recursive_target_steps=current_recursive_target_steps,
            loader_len=len(loader),
            tokenizer=tokenizer
        )
    return last_res, sur_baseline, res_baseline, current_recursive_target_steps, fade_in_progress



def train_batch(model, batch, lengths, opt, scheduler, config, device, epoch, batch_idx,
                last_res, sur_baseline, res_baseline, global_repair_counter, skip_counts,
                attn_records, loader_len, tokenizer, fade_in_progress, current_recursive_target_steps):

    for name, p in model.named_parameters():
        if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
            print(f"[🚨 NaN/∞ DETECTED IN PARAM] {name}:",
                  "NaNs:", torch.isnan(p).sum().item(),
                  "Infs:", torch.isinf(p).sum().item())
            raise RuntimeError(f"Stopping early: {name} is invalid")

    model.excess_flux_count = 0

    inp = batch[:, :-1].to(device)
    tgt = batch[:, 1:].to(device)

    if last_res is None:
        ctx = inp
    else:
        # compute fresh‐input vs. memory context vectors
        inp_emb = model.embedding(inp)                     # [B, L, D]
        inp_emb = model.embedding_dropout(inp_emb)
        real_ctx_vec = inp_emb.mean(dim=1)                  # [B, D]

        # apply stochastic dropout to resonant tokens before pooling
        noisy_res  = F.dropout(last_res.detach(), p=0.1, training=model.training)
        mem_ctx_vec = noisy_res.mean(dim=1)                 # [B, D]

        # blend via the model’s learnable gate
        gate = model.memory_gate_layer(real_ctx_vec)  # [B, D]
        ctx = gate * mem_ctx_vec + (1 - gate) * real_ctx_vec  # [B, D]

    padding_mask = (inp == tokenizer.token_to_id("<pad>")).to(device)

    if config["baseline"]:
        fade_in_strength = 1.0
    else:
        W = config["warmup_epochs"]
        final_steps = config["max_recursive_steps"]
        init_steps = 2
        P = epoch + batch_idx / loader_len

        if P < W:
            # still in initial warmup: no recursive influence
            model.max_recursive_steps = init_steps
            fade_in_strength = 0.0
        else:
            # after warmup, stage‐1 begins the first ramp for the 3rd step
            Pp = P - W
            # which extra-step block we’re in
            max_stage = max(0, final_steps - init_steps)
            stage = min(int(Pp // W) + 1, max_stage)
            # update total rec steps
            model.max_recursive_steps = init_steps + stage
            # ramp this stage from 0→1 over one warmup block
            frac = (Pp - (stage - 1) * W) / W
            fade_in_strength = float(min(max(frac, 0.0), 1.0))


    if config["baseline"]:
        baseline_outputs = model.forward(inp, context=ctx, padding_mask=padding_mask)
        logits = baseline_outputs[0]  # <-- Fix: only take logits
        res = None
        flux_penalty = torch.tensor(0.0, device=device)
        hidden_contrastive_loss = torch.tensor(0.0, device=device)
    else:
        # recursive_forward now returns (logits, hidden, flux, contrastive_loss)
        # mixed‐precision forward
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits, _hidden, flux_penalty, hidden_contrastive_loss = model.recursive_forward(
                inp, ctx,
                tol=config["recursive_convergence_tolerance"],
                padding_mask=padding_mask,
                fade_in_strength=fade_in_strength
            )
        # grab the actual resonant-token output (was formerly the 2nd return)
        res = model._res_tokens_for_ri
        # ensure we’ll see its gradient after backward
        res.retain_grad()

    if batch_idx % 10 == 0:
        global_step = epoch * len(train_loader) + batch_idx
        wandb.log({
            'fade_in_strength': fade_in_strength,
        }, step=global_step)

    # Immediately after logits are produced: clamp large values
    if logits.abs().max() > 10.0:
        print("[logits_stabilize] Warning: clamping logits to [-10, 10].")
        logits = torch.clamp(logits, min=-10.0, max=10.0)

    # === Immediately after forward pass, repair resonant tokens ===
    if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
    #     with torch.no_grad():
    #         # Repair any NaNs
    #         mask = torch.isnan(model._res_tokens_for_ri)
    #         if mask.any():
    #             print("[repair] NaNs detected in resonant tokens after forward pass. Replacing...")
    #             model._res_tokens_for_ri.data[mask] = 0.0

    #         # Soft project onto hypersphere (in-place so we don’t break the grad graph)
    #         new_proj = hard_project_onto_hypersphere(
    #             model._res_tokens_for_ri.data,
    #             radius=1.0
    #         )
    #         model._res_tokens_for_ri.data.copy_(new_proj)
        with torch.no_grad():
            mask = torch.isnan(model._res_tokens_for_ri)
            if mask.any():
                print("[repair] NaNs detected in resonant tokens after forward pass. Replacing...")
                model._res_tokens_for_ri = torch.nan_to_num(
                    model._res_tokens_for_ri, nan=0.0, posinf=1.0, neginf=-1.0
                )

        # Apply soft projection only if norm deviates significantly
        norms = model._res_tokens_for_ri.norm(dim=-1)
        if (norms > 1.5).any() or (norms < 0.5).any():
            model._res_tokens_for_ri = soft_project_onto_hypersphere(
                model._res_tokens_for_ri, target_radius=1.0, tolerance=0.25, strength=0.2
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
                res = res.clamp(min=-10.0, max=10.0)

    # strip off resonant tokens
    logits = logits[:, 1:1 + inp.size(1)]

    # compute raw CE for correct PPL logging before any augmentations
    # For logging true PPL use the same masked CE (no ignore_index cheat)
    pad_id   = tokenizer.token_to_id("<pad>")
    flat_l   = logits.reshape(-1, logits.size(-1))
    flat_t   = tgt.reshape(-1)
    ce_per_t = F.cross_entropy(flat_l, flat_t, reduction='none')
    mask     = (flat_t != pad_id).float()
    ce_loss  = (ce_per_t * mask).sum() / (mask.sum() + 1e-12)
    ppl = float(torch.exp(ce_loss))

    # switch back to FP32 and compute losses safely
    logits = logits.float()
    with autocast(device_type=device.type, enabled=False):
        primary, con, dvt, sur_reward, akl, res_reward, sur_baseline, res_baseline = \
            compute_losses(model, logits, tgt, inp, ctx, res,
                           config, device, epoch, batch_idx,
                           skip_counts, attn_records,
                           sur_baseline, res_baseline,
                           hidden_contrastive_loss, flux_penalty)

    # --- AMP step for the global optimizer ---
    opt.zero_grad()
    total_loss = primary + con
    # scale & backward
    model.scaler.scale(total_loss).backward()
    # unscale to apply gradient clipping
    model.scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # step & update scaler
    model.scaler.step(opt)
    model.scaler.update()

    # — sanitize Adam’s momentum buffers to drop any Inf/NaN —
    for state in opt.state.values():
        for buf in ("exp_avg", "exp_avg_sq"):
            if buf in state:
                state[buf].data = torch.nan_to_num(
                    state[buf].data,
                    nan=0.0, posinf=1e4, neginf=-1e4
                )

    # ——— sanitize every parameter so no NaNs/Infs can persist ———
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                # replace NaN→0, +Inf→+1e4, –Inf→–1e4
                clean = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4)
                # if anything changed, log it once
                if torch.any(clean != p):
                    print(f"[PARAM CLEANUP] {name} had invalids, repaired")
                p.copy_(clean)


    if batch_idx % 10 == 0:
        log_metrics(model, ppl, primary, con, dvt, sur_reward, akl, res_reward, 
                    ctx, epoch, batch_idx, loader_len, config, skip_counts, 
                    global_repair_counter)

    if scheduler is not None and epoch >= config["warmup_epochs"]:
        scheduler.step()

    last_res = res

    global_step = epoch * loader_len + batch_idx
    if global_step % 500 == 0:
        sample = tokenizer.decode(logits.argmax(-1)[0].tolist())
        print(f"[SAMPLE @ {global_step}]:", sample[:200])

    # === Every 100 batches: extra inner loop for resonant tokens ===
    if epoch >= config.get("warmup_epochs", 5) and batch_idx % 100 == 0 and (config["resonant_token_count"] + config["dynamic_resonant_token_count"]) > 0:
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

        # Optionally: build a temporary optimizer with higher LR (e.g., 2x)
        temp_opt = torch.optim.Adam(inner_res_params,
                         lr=2.0 * scheduler.optimizer.param_groups[0]['lr'],
                         weight_decay=0.0)

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

                    # Boost lambda_div slightly
                    token_loss = (
                        config["lambda_ri"] * ri_v_inner.mean() +
                        config["lambda_rs"] * rs_v_inner.mean() +
                        (1.5 * config["lambda_div"]) * dvt_inner
                    )

                    # === NEW: Similarity penalty between outer and inner resonant states ===
                    if res is not None and res_inner is not None and res.shape == res_inner.shape:
                        alignment_loss = F.mse_loss(res_inner, res.detach())
                        sim_weight = config.get("lambda_inner_align", 1.0)
                        token_loss += sim_weight * alignment_loss

                # AMP step for the inner‐loop optimizer
                temp_opt.zero_grad()
                model.scaler.scale(token_loss).backward()
                model.scaler.unscale_(temp_opt)
                temp_opt.step()
                model.scaler.update()

                # Clamp resonant tokens
                if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None:
                    model._res_tokens_for_ri = model._res_tokens_for_ri.clamp(-5.0, 5.0)

        # Restore parameter requires_grad
        for p in model.parameters():
            p.requires_grad_(True)

    return last_res, sur_baseline, res_baseline, fade_in_progress, current_recursive_target_steps





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
    all_sequences = encode_corpus(corpus_path, tokenizer,
                                  sequence_length=config["sequence_length"],
                                  max_tokens=config["max_tokens"])
    # — split out a validation set —
    random.shuffle(all_sequences)
    val_size = int(len(all_sequences) * config["validation_split"])
    val_seqs = all_sequences[:val_size]
    train_seqs = all_sequences[val_size:]

    train_loader = create_dataloader(train_seqs, tokenizer,
                                     batch_size=config["batch_size"])
    val_loader   = create_dataloader(val_seqs, tokenizer,
                                     batch_size=config["batch_size"])

    model, scaler, attn_records = build_model(config, tokenizer, DEVICE)
    opt, scheduler = setup_optimizer_and_scheduler(
        model, config, dataset_size=len(train_seqs), device_type=DEVICE.type
    )

    train(model, train_loader, val_loader, opt, scheduler,
          config, DEVICE, tokenizer)
