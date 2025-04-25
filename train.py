# train.py (fully updated with proper RI/RS integration)
import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from datasets import load_dataset
from resonantTransformer import EnhancedResonantTransformer, diversity_penalty, cosine_rampup, contrastive_loss
from sklearn.decomposition import PCA
from collections import deque
import pandas as pd
import plotly.express as px
import time

# === Hyperparameters & Config ===
from config import (
    baseline,
    d_model, num_heads, num_layers,
    resonant_token_count, dynamic_resonant_token_count,
    token_learning_amplifier,
    sequence_length, max_tokens, learning_rate,
    batch_size, num_epochs, warmup_epochs,
    lambda_ri, lambda_rs, lambda_div,
    lambda_sur, lambda_attn, lambda_res,
    multihead_resonance,
    max_recursive_steps,
    recursive_convergence_tolerance
)

if baseline:
    resonant_token_count = 0
    dynamic_resonant_token_count = 0
    multihead_resonance = False

# Device setup
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

token_part = "BASE" if baseline else f"{resonant_token_count}-{dynamic_resonant_token_count}"
millions = int(max_tokens / 1_000_000)
run_name = f"{token_part}-{d_model}-{num_heads}-{num_layers}-{sequence_length}-{millions}M"

# Initialize wandb
wandb.init(project="resonant-transformer-RoPE2", name=run_name, config={
    **{k: v for k, v in locals().items() if k.startswith('lambda_') or k in [
        'd_model','num_heads','num_layers','resonant_token_count',
        'dynamic_resonant_token_count','learning_rate','batch_size',
        'num_epochs','sequence_length','max_tokens','multihead_resonance',
        'recursive_convergence_tolerance'
    ]}
})

# Data preparation (unchanged)
dataset = load_dataset("roneneldan/TinyStories", split="train")
corpus_path = "tinystories_cached.txt"
if not os.path.exists(corpus_path):
    with open(corpus_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            line = entry['text'].strip()
            if line:
                f.write(line + "\n")

# Tokenizer setup
tokenizer_dir = "tokenizer-tinystories"
if not os.path.exists(tokenizer_dir):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[corpus_path], vocab_size=16000, min_frequency=2, special_tokens=["<pad>", "<unk>"])
    tokenizer.save_model(tokenizer_dir)
else:
    tokenizer = ByteLevelBPETokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")

tokenizer.add_special_tokens(["<pad>", "<unk>"])
pad_id = tokenizer.token_to_id("<pad>")
tokenizer.post_processor = BertProcessing(("<pad>", pad_id), ("<pad>", pad_id))

# Encode corpus with stride-based chunking for more coverage
tokens = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            tokens.extend(tokenizer.encode(line).ids)
        if len(tokens) >= max_tokens:
            break

# Allow overlapping chunks to multiply training data diversity
stride = sequence_length // 2  # 50% overlap
seqs = []
for i in range(0, len(tokens) - sequence_length, stride):
    chunk = tokens[i:i+sequence_length]
    if len(chunk) == sequence_length:
        seqs.append(chunk)

sequences = torch.tensor(seqs, dtype=torch.long).to(DEVICE)

loader = DataLoader(sequences, batch_size=batch_size, shuffle=True, drop_last=True)

# Model & Hooks
model = EnhancedResonantTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    resonant_token_count = resonant_token_count,
    dynamic_resonant_token_count = dynamic_resonant_token_count,
    multihead = multihead_resonance,
    max_recursive_steps = max_recursive_steps
).to(DEVICE)
model.train()

torch.autograd.set_detect_anomaly(True)

# — Prepare a precise set of only the resonant‑token params for inner updates —
inner_res_params = set()
if resonant_token_count > 0:
    inner_res_params.add(model.resonant_tokens)
if dynamic_resonant_token_count > 0:
    inner_res_params.update(model.controller.parameters())
if multihead_resonance:
    inner_res_params.update(model.resonator.parameters())

# Attention hook setup: register on custom encoder_layers
attn_records = []
def attn_hook(module, inp, output):
    # output is (attn_output, attn_weights)
    if isinstance(output, tuple) and output[1] is not None:
        attn_records.append(output[1].detach())
if not baseline and lambda_attn != 0:
    for layer in model.encoder_layers:
        layer.register_forward_hook(attn_hook)
avg_attn = None
attn_momentum = 0.99

# Optimizer & Criterion
# opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
# in train.py, when you build the optimizer:
if not baseline:
    # During warmup, exclude resonant token params
    opt = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters()
                    if all(x not in n for x in ['resonant_tokens', 'controller', 'resonator'])],
         'lr': learning_rate}
    ])
else:
    opt = torch.optim.Adam([
        # everything else
        {'params': [p for n,p in model.named_parameters() if 'resonant_tokens' not in n
                    and 'controller' not in n and 'resonator' not in n],
        'lr': learning_rate}
    ])
    

crit = nn.CrossEntropyLoss(ignore_index=pad_id)
last_res=None

sur_baseline = None
res_baseline = None

last_run_time = time.time()


# Training
for epoch in range(num_epochs):

    # Disable updates to resonant token parameters during warmup
    if epoch < warmup_epochs:
        for p in inner_res_params:
            p.requires_grad = False
    else:
        for p in inner_res_params:
            p.requires_grad = True

    # === Rebuild optimizer at end of warmup to include resonant params ===
    if epoch == warmup_epochs:
        print(f"Rebuilding optimizer at epoch {epoch}")
        opt = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters()
                        if all(x not in n for x in ['resonant_tokens', 'controller', 'resonator'])],
             'lr': learning_rate},
            {'params': [model.resonant_tokens],
             'lr': learning_rate * token_learning_amplifier},
            {'params': model.controller.parameters(),
             'lr': learning_rate * token_learning_amplifier},
            {'params': model.resonator.parameters(),
             'lr': learning_rate * token_learning_amplifier}
        ])

    # Rebuild fresh references to resonant-token parameters
    inner_res_params = set()
    if resonant_token_count > 0:
        inner_res_params.add(model.resonant_tokens)
    if dynamic_resonant_token_count > 0:
        inner_res_params.update(model.controller.parameters())
    if multihead_resonance:
        inner_res_params.update(model.resonator.parameters())

    model.alpha = cosine_rampup(epoch, warmup_epochs)
    for bidx,batch in enumerate(loader):
        inp = batch[:,:-1]; tgt = batch[:,1:]
        ctx = last_res if last_res is not None else inp
        if not baseline:
            logits, res = model.recursive_forward(inp, ctx, tol=recursive_convergence_tolerance)
        else:
            logits, res, _, _ = model.forward(inp, ctx)
        steps = getattr(model, "last_recursive_steps", 0)
        logits = logits[:,:inp.size(1)]
        # Primary loss (CE)
        primary = crit(logits.reshape(-1,logits.size(-1)), tgt.reshape(-1))
        # — Surprisal‑drop reward (normalized, clipped, baselined) —
        if not baseline:
            lp  = F.log_softmax(logits, dim=-1)
            tlp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            spr = -tlp; dr = spr[:,:-1]-spr[:,1:]
            sur = F.relu(dr).mean() / inp.size(1)        # normalize per token
            sur = torch.clamp(sur, max=0.5)             # clip to [0, 0.5]

            # update running baseline
            if sur_baseline is None:
                sur_baseline = sur.detach()
            else:
                sur_baseline = 0.99 * sur_baseline + 0.01 * sur.detach()

            sur_reward = sur - sur_baseline             # only above‐baseline counts
            primary   = primary - lambda_sur * sur_reward
        else:
            sur_reward = torch.tensor(0.0, device=DEVICE)

        # — Attention‑KL reward (capped) —
        if not baseline and attn_records:
            ca = torch.stack(attn_records).mean(0)
            avg_attn = ca.mean(0) if avg_attn is None else (
                        attn_momentum * avg_attn + (1-attn_momentum) * ca.mean(0)
                    )
            cur, avg = ca.mean(0) + 1e-8, avg_attn + 1e-8
            akl = F.kl_div(cur.log(), avg, reduction='batchmean')
            akl = torch.clamp(akl, max=0.1)             # cap at 0.1 nats
            primary = primary - lambda_attn * akl   # modestly upweight if desired
            attn_records.clear()
        else:
            akl = torch.tensor(0.0, device=DEVICE)

        # — Resolution‑coherence reward (clipped, baselined) —
        if not baseline and logits.size(1) >= 2:
            pr  = lp.exp()
            ent = -(pr * lp).sum(-1)
            pen, pos = ent[:, -2], ent[:, -1]
            rr = F.relu(pen - pos).mean()
            rr = torch.clamp(rr, max=0.5)

            # update running baseline
            if res_baseline is None:
                res_baseline = rr.detach()
            else:
                res_baseline = 0.99 * res_baseline + 0.01 * rr.detach()

            res_reward = rr - res_baseline
            primary    = primary - lambda_res * res_reward
        else:
            res_reward = torch.tensor(0.0, device=DEVICE)

        # — Contrastive term unchanged —
        if not baseline:
            ci = inp.clone()
            ci[:, -1] = torch.randint(0, tokenizer.get_vocab_size(), (batch_size,), device=DEVICE)
            # compute contrastive logits via recursive inference
            contrast_logits, _ = model.recursive_forward(ci, ci, tol=recursive_convergence_tolerance)
            con = contrastive_loss(logits, contrast_logits)
            primary = primary + con
        else:
            con = torch.tensor(0.0, device=DEVICE)

        # — Entropy bonus (discourage low‑entropy repetition) —
        if not baseline:
            # log‑probabilities over the full vocab
            lp_ent = F.log_softmax(logits, dim=-1)      # shape [B, L, V]
            pr_ent = lp_ent.exp()                       # shape [B, L, V]
            # entropy per token = −∑ p log p; then mean over batch & sequence
            ent_loss = -(pr_ent * lp_ent).sum(-1).mean()  
            # small weight to reward higher entropy
            primary = primary - 1e-4 * ent_loss

        # first build the final loss including these terms
        term1 = term2 = term3 = 0.0
        dvt   = torch.tensor(0.0, device=DEVICE)
        if not baseline and res.numel() > 0 and hasattr(model, '_res_tokens_for_ri'):
            # compute gradient w.r.t. the resonant tokens (may be None if unused)
            grad_tuple = torch.autograd.grad(
                primary, model._res_tokens_for_ri,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            gr = grad_tuple[0]
            if gr is None:
                # no gradient flowed—use zeros
                gr = torch.zeros_like(model._res_tokens_for_ri)
            gn    = gr.norm(dim=-1).clamp(min=1e-6)
            ri_v  = (gr * model._res_tokens_for_ri).sum(dim=-1).abs() / gn
            term1 = lambda_ri * ri_v.mean()
            rs_v  = 1 - F.cosine_similarity(gr, model._res_tokens_for_ri, dim=-1)
            term2 = lambda_rs * rs_v.mean()
            dvt   = lambda_div * diversity_penalty(model._res_tokens_for_ri)
        
        self_model_loss_weight = 0.1
        if hasattr(model, 'last_self_model_loss'):
            term3 = model.last_self_model_loss * self_model_loss_weight
        
        
        final = primary - term1 - term2 + term3 + dvt

        # Backprop the full loss
        opt.zero_grad()
        final.backward()

        if not baseline:
            # NOW extract the _actual_ gradients on the resonant tokens
            # — raw‐dot RI & standard RS (no grad‑norm division) —
            if hasattr(model, '_res_tokens_for_ri') and model._res_tokens_for_ri is not None and model._res_tokens_for_ri.grad is not None:
                gr = model._res_tokens_for_ri.grad  # (B, T, D)
                # raw absolute dot ⇝ “influence” magnitude
                ri_v = (gr * model._res_tokens_for_ri).sum(dim=-1).abs()
                rs_v = 1 - F.cosine_similarity(gr, model._res_tokens_for_ri, dim=-1)
                val_ri = ri_v.mean().item()
                val_rs = rs_v.mean().item()
                cos_sim_v = F.cosine_similarity(gr, model._res_tokens_for_ri, dim=-1)
                val_cos_sim = cos_sim_v.mean().item()
            else:
                val_ri = 0.0
                val_rs = 0.0
                val_cos_sim = 0.0

            
            cos_sim_v = F.cosine_similarity(gr, model._res_tokens_for_ri, dim=-1)
            val_cos_sim = cos_sim_v.mean().item()

        
        other_params = [p for n,p in model.named_parameters()
                        if 'resonant_tokens' not in n
                        and 'controller' not in n
                        and 'resonator' not in n]
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)

        opt.step()

        # Clamp resonant token norms post-update to prevent explosion
        with torch.no_grad():
            if hasattr(model, 'resonant_tokens'):
                norm = model.resonant_tokens.norm(dim=-1, keepdim=True).clamp(min=1.0, max=10.0)
                model.resonant_tokens.copy_(model.resonant_tokens / norm)

        # every 100 batches, give tokens 5 exclusive mini‑steps:
        if epoch >= warmup_epochs and bidx % 100 == 0 and (resonant_token_count + dynamic_resonant_token_count) > 0:
            # 1) freeze all except the exact inner_res_params
            for p in model.parameters():
                p.requires_grad = False
            for p in inner_res_params:
                p.requires_grad = True

            # 2) inner token‑only loop
            for _ in range(5):

                # recompute primary to get a fresh autograd graph
                logits, res = model.recursive_forward(inp, tol=recursive_convergence_tolerance)
                logits = logits[:, :inp.size(1), :]
                primary_inner = crit(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1)
                )

                # compute the gradient of that primary w.r.t. the tokens
                grad_tuple = torch.autograd.grad(
                    primary_inner,
                    model._res_tokens_for_ri,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )
                gr_inner = grad_tuple[0]
                if gr_inner is None:
                    gr_inner = torch.zeros_like(model._res_tokens_for_ri)

                # now build the same RI/RS/div terms
                ri_v_inner = (gr_inner * model._res_tokens_for_ri).sum(dim=-1).abs()
                term1      = lambda_ri * ri_v_inner.mean()
                rs_v_inner = 1 - F.cosine_similarity(gr_inner, model._res_tokens_for_ri, dim=-1)
                term2      = lambda_rs * rs_v_inner.mean()
                dvt        = lambda_div * diversity_penalty(model._res_tokens_for_ri)

                token_loss = term1 + term2 + dvt
                opt.zero_grad()
                token_loss.backward()
                opt.step()

            # 3) unfreeze everything
            for p in model.parameters():
                p.requires_grad_(True)

        # update
        last_res = getattr(model,'_res_tokens_for_ri',None)
        if last_res is not None: last_res=last_res.detach()
        # log
        if bidx%10==0:
            global_step = epoch * len(loader) + bidx

            resolution_score = 0.0
            self_attn_mean = 0.0

            # --- Diagnostic logging ---
            if hasattr(model, 'resolution_score') and model.resolution_score is not None:
                resolution_score = model.resolution_score

            if hasattr(model, 'attention_trajectory') and model.attention_trajectory:
                # attention_trajectory now holds scalar mean-attention values
                self_attn_mean = model.attention_trajectory[-1]


            if not baseline:
                if last_res is not None:
                    ctx_emb = model.embedding(last_res) if last_res.dtype in (torch.int64, torch.int32) else last_res
                    context_vec = ctx_emb.mean(dim=1)
                elif model.global_res is not None:
                    context_vec = model.global_res.mean(dim=1).expand(batch_size, -1).contiguous()
                else:
                    context_vec = model.self_token.expand(batch_size, -1, -1).mean(dim=1)


            if (resonant_token_count + dynamic_resonant_token_count) == 0:
                val_ri = 0.0
                val_rs = 0.0
                res_token_norm = 0.0
                stat_norm = 0.0
                dyn_norm = 0.0
                mh_norm = 0.0
            else:
                val_ri = ri_v.mean().item()
                val_rs = rs_v.mean().item()
                res_token_norm = model._res_tokens_for_ri.norm().item()
                stat_norm = model.resonant_tokens.norm().item()
                dyn_norm = model.controller(context_vec).norm().item() if hasattr(model, 'controller') else 0
                mh_norm = model.resonator(context_vec).norm().item() if hasattr(model, 'resonator') else 0

            if baseline:
                val_sr = 0.0
                val_akl = 0.0
                val_rr = 0.0
                val_cos_sim = 0
            else:
                val_sr = sur_reward.item()
                val_akl = akl.item()
                val_rr = res_reward.item()
            
            now = time.time()
            delta_s = now - last_run_time
            last_run_time = now

            wandb.log({
                'loss':final.item(),
                'perplexity':float(np.exp(final.item())),
                'ri':val_ri,
                'rs':val_rs,
                'rs_cos':val_cos_sim,
                'surprisal_reward':val_sr,
                'attention_kl':val_akl,
                'resolution_score':resolution_score,
                'res_token_norm': res_token_norm,
                'static_res_norm': stat_norm,
                'dynamic_res_norm': dyn_norm,
                'multihead_res_norm': mh_norm,
                'self_attn_mean':self_attn_mean,
                'recursive_steps':steps,
                'last_flux_cost': model.last_flux_cost if hasattr(model, "last_flux_cost") else 0.0,
                'flux_budget': model.flux_budget if hasattr(model, "flux_budget") else 0.0
            }, step=global_step)
            print(f"{delta_s:.1f}: {global_step} | {epoch+1} | {bidx} - PPL: {float(np.exp(final.item())):.2f} | NORM: {res_token_norm:.2f} | SUR: {val_sr:.4f} | AKL: {val_akl:.4f} | RES: {resolution_score:.4f} | SELF: {self_attn_mean:.4f} | RI: {val_ri:.4e} | RSC: {val_cos_sim:.4e} | STEPS: {steps}")

    # === Save checkpoint at end of this epoch ===
    if (epoch + 1) % 10 == 0:
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch + 1,
            'config': {
                'vocab_size': tokenizer.get_vocab_size(),
                'sequence_length': sequence_length,
                'd_model': d_model,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'max_tokens': max_tokens,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'warmup_epochs': warmup_epochs,
                'resonant_token_count': resonant_token_count,
                'dynamic_resonant_token_count': dynamic_resonant_token_count,
                'multihead': multihead_resonance,
                'recursive_convergence_tolerance': recursive_convergence_tolerance,
                'max_recursive_steps': max_recursive_steps
            },
            'final_resonant_state': getattr(model, '_res_tokens_for_ri', None)
        }
        epoch_filename = f"training-checkpoint-epoch{epoch+1}.pt"
        torch.save(state, epoch_filename)
        print(f"Checkpoint saved to {epoch_filename} at end of epoch {epoch+1}")

# Save final state
state = {
    'model_state_dict': model.state_dict(),
    'config': {
        'baseline': baseline,
        'vocab_size': tokenizer.get_vocab_size(),
        'sequence_length': sequence_length,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'max_tokens': max_tokens,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'warmup_epochs': warmup_epochs,
        'resonant_token_count': resonant_token_count,
        'dynamic_resonant_token_count': dynamic_resonant_token_count,
        'multihead': multihead_resonance,
        'recursive_convergence_tolerance': recursive_convergence_tolerance,
        'max_recursive_steps': max_recursive_steps
    },
    'final_resonant_state': last_res
}
millions = int(max_tokens / 1_000_000)
token_part = "BASE" if baseline else f"{resonant_token_count}-{dynamic_resonant_token_count}"
model_filename = f"{token_part}-{d_model}-{num_heads}-{num_layers}-{sequence_length}-{millions}M.pt"
torch.save(state, model_filename)
print(f"Model saved to {model_filename} with config and final dynamic resonant state embedded.")
