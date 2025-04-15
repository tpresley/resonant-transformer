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

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

from config import d_model, num_heads, num_layers, resonant_token_count, sequence_length, max_tokens, learning_rate, batch_size, num_epochs, warmup_epochs, lambda_ri, lambda_rs, lambda_div

# Initialize wandb
wandb.init(project="resonant-transformer", config={
    "d_model": d_model,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "resonant_token_count": resonant_token_count,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "sequence_length": sequence_length,
    "max_tokens": max_tokens,
    "lambda_ri": lambda_ri,
    "lambda_rs": lambda_rs,
    "lambda_div": lambda_div
})

# Load TinyStories
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

# Encode corpus
tokens = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        encoded = tokenizer.encode(line.strip())
        tokens.extend(encoded.ids)
        if len(tokens) >= max_tokens:
            break
tokens = tokens[:max_tokens]

def make_sequences(tokens, seq_len):
    num_full = (len(tokens) - seq_len) // seq_len
    return torch.tensor([tokens[i*seq_len:(i+1)*seq_len] for i in range(num_full)], dtype=torch.long)

data = make_sequences(tokens, sequence_length)
vocab_size = tokenizer.get_vocab_size()
data = data.to(DEVICE)
loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

USE_CONTRASTIVE_LOSS = True
DYNAMIC_RESONANCE = True
MULTIHEAD_RESONANCE = False
if DYNAMIC_RESONANCE and MULTIHEAD_RESONANCE:
    print("[Warning] Both dynamic and multihead resonance are enabled. Defaulting to dynamic resonance.")
    MULTIHEAD_RESONANCE = False

model = EnhancedResonantTransformer(
    vocab_size, d_model, num_heads, num_layers, resonant_token_count,
    dynamic=DYNAMIC_RESONANCE, multihead=MULTIHEAD_RESONANCE
)
model.to(DEVICE)
model.alpha = 0.0  # Ensure full gradient flow through resonant tokens at start
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

print("Starting training...")
last_resonant_state = None
for epoch in range(num_epochs):
    model.alpha = cosine_rampup(epoch, warmup_epochs)  # Ramp from 0.0 to 1.0
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        context_input = last_resonant_state if last_resonant_state is not None else input_seq

        ri_per_token = torch.zeros((batch_size, resonant_token_count), device=DEVICE)
        rs_per_token = torch.ones((batch_size, resonant_token_count), device=DEVICE)
        ri = torch.tensor(0.0, device=DEVICE)
        rs = torch.tensor(1.0, device=DEVICE)

        output, res_tokens = model(input_seq, context=context_input)
        res_tokens_for_grad = getattr(model, "_res_tokens_for_ri", None)

        output = output[:, :input_seq.shape[1]]
        output = torch.clamp(output, min=-10.0, max=10.0)
        output_flat = output.reshape(-1, vocab_size)
        target_flat = target_seq.reshape(-1)
        loss = criterion(output_flat, target_flat)
        if not torch.isfinite(loss):
            raise ValueError("NaN or Inf detected in main loss")

        if res_tokens is not None and res_tokens.numel() > 0 and res_tokens_for_grad is not None:
            last_resonant_state = res_tokens_for_grad.detach()
            if model.training:
                dropout_mask = torch.rand(res_tokens.size(1), device=res_tokens.device) > 0.1
                res_tokens = res_tokens[:, dropout_mask, :]
                # Do NOT mask res_tokens_for_grad to preserve gradient flow

            loss += lambda_div * diversity_penalty(res_tokens)

            grads_res = torch.autograd.grad(loss, res_tokens_for_grad, retain_graph=True, create_graph=True, allow_unused=True)[0]
            if grads_res is None:
                print("[Warning] res_tokens_for_grad was not used in the loss computation. Check your model's forward pass.")
                grads_res = torch.zeros_like(res_tokens_for_grad)
            if grads_res is not None:
                grad_norm = torch.clamp(grads_res.norm(dim=-1), min=1e-3)
                dot = (grads_res * res_tokens_for_grad).sum(dim=-1)
                ri_per_token = torch.abs(dot) / grad_norm
                rs_per_token = 1.0 - F.cosine_similarity(grads_res, res_tokens_for_grad, dim=-1)
                rs_per_token = torch.clamp(rs_per_token, 0.0, 2.0)
                ri = ri_per_token.mean()
                rs = rs_per_token.mean()

            loss = loss - lambda_ri * ri - lambda_rs * rs

            # === Participation Reward: Compare model with vs. without resonant tokens ===
            target_seq = batch[:, 1:]
            target_flat = target_seq.reshape(-1)
            with torch.no_grad():
                empty_context = torch.empty(input_seq.size(0), 0, model.d_model, device=input_seq.device)
                output_nores, _ = model(input_seq, context=empty_context)
                output_nores = output_nores[:, :input_seq.shape[1]]
                output_nores = torch.clamp(output_nores, min=-10.0, max=10.0)
                output_nores_flat = output_nores.reshape(-1, vocab_size)
                target_flat = target_seq.reshape(-1)
                loss_nores = criterion(output_nores_flat, target_flat)
                if not torch.isfinite(loss_nores):
                    raise ValueError("NaN or Inf detected in loss_nores")

                output_with_res, _ = model(input_seq, context=context_input)
                output_with_res = output_with_res[:, :input_seq.shape[1]]
                output_with_res = torch.clamp(output_with_res, min=-10.0, max=10.0)
                output_with_res_flat = output_with_res.reshape(-1, vocab_size)
                loss_with_res = criterion(output_with_res_flat, target_flat)
                if not torch.isfinite(loss_with_res):
                    raise ValueError("NaN or Inf detected in loss_with_res")

            resonant_usage_reward = loss_nores - loss_with_res
            resonant_usage_reward = torch.clamp(resonant_usage_reward, -10.0, 10.0)
            lambda_participation = 0.1  # Tune this weight
            loss -= lambda_participation * resonant_usage_reward
            # NOTE: This doubles forward-pass cost. In future, consider gating this every N batches.

        if USE_CONTRASTIVE_LOSS:
            contrast_input = input_seq.clone()
            contrast_input[:, -1] = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
            contrast_output, _ = model(contrast_input, context=contrast_input)
            contrastive = contrastive_loss(output, contrast_output)
            loss += contrastive

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        last_resonant_state = res_tokens_for_grad.detach() if res_tokens_for_grad is not None else None

        token_metrics = {}
        if res_tokens is not None and res_tokens.numel() > 0:
            ri_token_means = ri_per_token.mean(dim=0)
            rs_token_means = rs_per_token.mean(dim=0)
            ri_token_stds = ri_per_token.std(dim=0)
            rs_token_stds = rs_per_token.std(dim=0)

            for i in range(ri_token_means.size(0)):
                token_metrics[f"ri_token_{i}_mean"] = ri_token_means[i].item()
                token_metrics[f"ri_token_{i}_std"] = ri_token_stds[i].item()
                token_metrics[f"rs_token_{i}_mean"] = rs_token_means[i].item()
                token_metrics[f"rs_token_{i}_std"] = rs_token_stds[i].item()

        influence_score = 100.0 * torch.sigmoid(2.0 * resonant_usage_reward.detach()).item() if res_tokens is not None and res_tokens.numel() > 0 else 0.0

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | PPL: {np.exp(loss.item()):.2f} | RI: {ri.item():.4f} | RS: {rs.item():.4f} | Influence: {influence_score:.2f}")
            global_step = epoch * len(loader) + batch_idx
            wandb.log({
                "loss": loss.item(),
                "perplexity": np.exp(loss.item()),
                "ri": ri.item(),
                "rs": rs.item(),
                "resonant_tokens_active": dropout_mask.sum().item() if res_tokens is not None and 'dropout_mask' in locals() else 0,
                "resonant_influence_score": influence_score,
                **token_metrics
            }, step=global_step)

torch.save(model.state_dict(), "enhanced_resonant_model.pt")
print("Model saved.")
