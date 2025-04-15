# --- Enhanced Resonant Transformer Full Program (with Humor Metaphor Extensions) ---

# ========== Imports and Hyperparameters ==========
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import re
import os
import time
import argparse
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import requests
from datasets import load_dataset

# ========== Command Line Args ==========
parser = argparse.ArgumentParser()
parser.add_argument('--log-batches', action='store_true', help='Print batch-level console logs during training')
parser.add_argument('--inference', action='store_true', help='Run inference on user input after training')
args = parser.parse_args()

# ========== Device Setup ==========
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

max_tokens = 1000000
sequence_length = 384

if not args.inference:
    # ========== Download and Prepare Corpus ==========
    print("Preparing roneneldan/TinyStories using HuggingFace datasets...")
    corpus_path = "tinystories_cached.txt"
    if not os.path.exists(corpus_path):
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        print("Saving to local cache...")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                line = entry['text'].strip()
                if line:
                    f.write(line + "\n")
        print(f"Saved roneneldan/TinyStories to {corpus_path}")
    else:
        print("Loading roneneldan/TinyStories from cache...")

    # ========== Train or Load Tokenizer ==========
    tokenizer_dir = "tokenizer-tinystories"
    if not os.path.exists(tokenizer_dir) or not os.path.exists(f"{tokenizer_dir}/vocab.json") or not os.path.exists(f"{tokenizer_dir}/merges.txt"):
        print("Training BPE tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[corpus_path],
                        vocab_size=16000,
                        min_frequency=2,
                        special_tokens=["<pad>", "<unk>"])
        tokenizer.save_model(tokenizer_dir)
    else:
        print("Loading existing tokenizer...")
        tokenizer = ByteLevelBPETokenizer(
            f"{tokenizer_dir}/vocab.json",
            f"{tokenizer_dir}/merges.txt"
        )

    tokenizer.add_special_tokens(["<pad>", "<unk>"])
    pad_id = tokenizer.token_to_id("<pad>")
    unk_id = tokenizer.token_to_id("<unk>")
    tokenizer.post_processor = BertProcessing(("<pad>", pad_id), ("<pad>", pad_id))

    # ========== Encode Corpus ==========
    tokens = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            encoded_line = tokenizer.encode(line)
            tokens.extend(encoded_line.ids)
            if len(tokens) >= max_tokens:
                tokens = tokens[:max_tokens]
                break
    print(f"Encoded {len(tokens)} tokens using BPE tokenizer.")

    # ========== Create Dataset ==========
    def make_sequences(tokens, seq_len):
        num_full_sequences = (len(tokens) - seq_len) // seq_len
        sequences = [
            tokens[i*seq_len : (i+1)*seq_len]
            for i in range(num_full_sequences)
        ]
        return torch.tensor(sequences, dtype=torch.long)

    if len(tokens) < sequence_length:
        raise ValueError(f"Corpus too small: {len(tokens)} tokens < {sequence_length} required.")

    data = make_sequences(tokens, sequence_length)
    vocab_size = tokenizer.get_vocab_size()

# ========== Flags to enable/disable features (enabled by default) ==========
BASELINE = True
ENABLE_DYNAMIC_RESONANCE = True if BASELINE is False else False
ENABLE_MULTIHEAD_RESONANCE = False if BASELINE is False else False
ENABLE_CONTRASTIVE_LOSS = True if BASELINE is False else False
ENABLE_NOISE_INJECTION = False if BASELINE is False else False

# ========== Model Definitions ==========


class MultiHeadResonance(nn.Module):
    def __init__(self, num_heads, res_tokens, d_model):
        super().__init__()
        self.resonant_bank = nn.Parameter(torch.randn(num_heads, res_tokens, d_model))
        self.selector = nn.Linear(d_model, num_heads)
        self.num_heads = num_heads
        self.res_tokens = res_tokens
        self.d_model = d_model

    def forward(self, context_vec):
        weights = torch.softmax(self.selector(context_vec), dim=-1)
        weighted_tokens = torch.einsum('bh,hnd->bnd', weights, self.resonant_bank)
        return weighted_tokens

class ResonantController(nn.Module):
    def __init__(self, d_model, res_tokens):
        super().__init__()
        self.linear = nn.Linear(d_model, res_tokens * d_model)
        self.res_tokens = res_tokens
        self.d_model = d_model

    def forward(self, context_embedding):
        out = self.linear(context_embedding).view(-1, self.res_tokens, self.d_model)
        out = F.layer_norm(out, (self.d_model,))
        return out

class EnhancedResonantTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, res_tokens):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dynamic = ENABLE_DYNAMIC_RESONANCE
        self.multihead = ENABLE_MULTIHEAD_RESONANCE
        self.res_tokens = res_tokens
        self.d_model = d_model
        self.alpha = 0.0

        if self.dynamic:
            self.controller = ResonantController(d_model, res_tokens)
        elif self.multihead:
            self.resonator = MultiHeadResonance(num_heads=4, res_tokens=res_tokens, d_model=d_model)
        else:
            self.resonant_tokens = nn.Parameter(torch.randn(1, res_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=512, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, context=None):
        x = self.embedding(x)
        B = x.size(0)

        # Determine what to use as resonant tokens based on config and context
        if context is not None:
            if context.dtype in (torch.float32, torch.float64):
                # If context is already embedded (e.g., res_tokens from last step)
                context_vec = context.mean(dim=1)
            else:
                # If context is a token ID tensor
                context_vec = self.embedding(context.long()).mean(dim=1)

        if self.dynamic and context is not None:
            res_tokens = self.controller(context_vec)
        elif self.multihead and context is not None:
            res_tokens = self.resonator(context_vec)
        else:
            res_tokens = self.resonant_tokens.repeat(B, 1, 1)

        # Track res_tokens for RI computation
        if self.training and self.res_tokens > 0 and res_tokens.numel() > 0:
            res_tokens.retain_grad()
            self._res_tokens_for_ri = res_tokens

        # Optional: Inject noise during training
        if ENABLE_NOISE_INJECTION:
            res_tokens = inject_resonant_noise(res_tokens)

        # Concatenate resonant tokens to input sequence
        x = torch.cat([amplify_grad(res_tokens, self.alpha), x], dim=1).transpose(0, 1)

        # Run through Transformer encoder
        encoded = self.encoder(x)  # (seq_len+res, batch, dim)

        # Discard resonant tokens from output
        out = encoded[self.res_tokens:].transpose(0, 1)  # (batch, seq_len, dim)

        # Project to vocab logits
        return self.output(out), res_tokens


def contrastive_loss(original_logits, contrast_logits, margin=1.0):
    orig_repr = original_logits.mean(dim=1)
    contrast_repr = contrast_logits.mean(dim=1)
    dist = F.pairwise_distance(orig_repr, contrast_repr, p=2)
    return torch.clamp(margin - dist, min=0).mean()

def inject_resonant_noise(tokens, noise_level=0.1):
    noise = torch.randn_like(tokens)
    noise = F.normalize(noise, dim=-1)
    return tokens + noise_level * noise

def diversity_penalty(tokens):
    # tokens: (B, T, D) → resonant tokens for each batch
    # Normalize across dim=-1 to focus on direction
    normed = F.normalize(tokens, dim=-1)  # (B, T, D)

    # Cosine similarity between all pairs of tokens
    sim_matrix = torch.einsum('btd,bkd->btk', normed, normed)  # (B, T, T)

    # Remove self-similarity (diagonal = 1.0)
    eye = torch.eye(sim_matrix.size(-1), device=sim_matrix.device).unsqueeze(0)
    sim_matrix = sim_matrix * (1 - eye)

    # Mean non-diagonal similarity → lower is better
    penalty = sim_matrix.sum(dim=(1, 2)) / (tokens.size(1) * (tokens.size(1) - 1))
    return penalty.mean()

def amplify_grad(x, alpha):
    return (x * alpha).detach() + x * (1 - alpha)

def cosine_rampup(t, warmup_epochs):
    if t >= warmup_epochs:
        return 1.0
    return 0.5 * (1 - math.cos(math.pi * t / warmup_epochs))

# ========== Training and Inference ==========
d_model = 32
num_heads = 2
num_layers = 2
resonant_token_count = 8 if BASELINE is False else 0
learning_rate = 1e-4
batch_size = 64
num_epochs = 400
warmup_epochs = 10


if not args.inference:
    model = EnhancedResonantTransformer(vocab_size, d_model, num_heads, num_layers, resonant_token_count)
    wandb.init(project="resonant-transformer-extended", config={
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "resonant_token_count": resonant_token_count,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "sequence_length": sequence_length,
        "vocab_size": vocab_size,
        "max_tokens": max_tokens,
        "dynamic": ENABLE_DYNAMIC_RESONANCE,
        "multihead": ENABLE_MULTIHEAD_RESONANCE,
        "contrastive": ENABLE_CONTRASTIVE_LOSS,
        "noise_injection": ENABLE_NOISE_INJECTION
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    data = data.to(DEVICE)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    model.to(DEVICE)
    model.train()

    ri_collapse_threshold = 0.0001
    last_resonant_state = None

    for epoch in range(num_epochs):
        total_loss = 0
        lambda_ri = 0.01 if resonant_token_count > 0 else 0.00
        lambda_rs = 0.02 if resonant_token_count > 0 else 0.00
        lambda_div = 0.05 if resonant_token_count > 0 else 0.00
        model.alpha = 1.0 - cosine_rampup(epoch, warmup_epochs)
        skip_count_epoch = 0
        actual_updates = 0
        initial_weights = model.state_dict()
        for batch_idx, batch in enumerate(loader):
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            input_seq, target_seq = input_seq.to(DEVICE), target_seq.to(DEVICE)

            context_input = last_resonant_state if last_resonant_state is not None else input_seq
            output, res_tokens = model(input_seq, context=context_input)

            output = output[:, :target_seq.shape[1]]  # ensure alignment
            loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))

            # Compute RI and RS
            res_tokens = getattr(model, "_res_tokens_for_ri", None)



            if res_tokens is not None and res_tokens.numel() > 0:
                loss += lambda_div * diversity_penalty(res_tokens)
                
                grads = torch.autograd.grad(loss, res_tokens, retain_graph=True, create_graph=True)[0]
                token_vecs = res_tokens
                if grads is not None:
                    ri_per_token = torch.abs(torch.sum(grads * token_vecs, dim=-1)) / torch.clamp(torch.norm(grads, dim=-1), min=1e-3)
                    ri = ri_per_token.mean()
                    rs_per_token = 1.0 - F.cosine_similarity(grads, token_vecs, dim=-1)
                    rs_per_token = torch.clamp(rs_per_token, min=0.0, max=2.0)
                    rs = rs_per_token.mean()
                    aux_loss = -lambda_ri * ri - lambda_rs * rs
                    loss = loss + aux_loss
                else:
                    ri = torch.tensor(0.0)
                    rs = torch.tensor(0.0)
                    ri_per_token = torch.zeros(model.res_tokens)
            else:
                ri = torch.tensor(0.0)
                rs = torch.tensor(0.0)
                ri_per_token = torch.zeros(1)


            # if resonant_token_count > 0 and ri.item() < ri_collapse_threshold:
            #     skip_count_epoch += 1
            #     print(f"Skipping Batch Update due to low RI: {ri.item():.4e}")
            #     # Reinitialize weak resonant tokens
            #     if not ENABLE_DYNAMIC_RESONANCE and not ENABLE_MULTIHEAD_RESONANCE:
            #         with torch.no_grad():
            #             mean_ri_per_token = ri_per_token.mean(dim=0)
            #             low_ri_mask = mean_ri_per_token < ri_collapse_threshold
            #             num_low = low_ri_mask.sum().item()
            #             if num_low > 0:
            #                 print(f"Reinitializing {int(num_low)} low-RI resonant tokens...")
            #                 model.resonant_tokens[0][low_ri_mask] = torch.randn_like(model.resonant_tokens[0][low_ri_mask])
            #     continue

            if ENABLE_CONTRASTIVE_LOSS:
                contrast_input = input_seq.clone()
                contrast_input[:, -1] = torch.randint(0, vocab_size, (batch_size,), device=DEVICE)
                contrast_output, _ = model(contrast_input, context=contrast_input)
                contrastive = contrastive_loss(output, contrast_output)
                loss += contrastive

            if not torch.isfinite(loss):
                print(f"Non-finite loss detected at batch {batch_idx}, resetting model weights.")
                model.load_state_dict(initial_weights)
                continue

            if batch_idx % 10 == 0:
                if res_tokens is not None and res_tokens.numel() > 0:
                    res_norm = res_tokens.norm(dim=-1)
                else:
                    res_norm = torch.tensor(0.0)
                if args.log_batches:
                    print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | RI: {ri.item():.4e} | RS: {rs.item():.4e} | PPL: {np.exp(loss.item()):.2f}")
                wandb_step = epoch * len(loader) + batch_idx if 'batch_idx' in locals() else epoch
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_ri": ri.item() if resonant_token_count > 0 else 0.0,
                    "batch_rs": rs.item() if resonant_token_count > 0 else 0.0,
                    "batch_ri_tokens_mean": ri_per_token.mean().item() if resonant_token_count > 0 else 0.0,
                    "batch_ri_tokens_std": ri_per_token.std().item() if resonant_token_count > 0 else 0.0,
                    "batch_perplexity": np.exp(loss.item()),
                    "batch_resonant_norm_mean": res_norm.mean().item() if resonant_token_count > 0 else 0.0,
                    "batch_resonant_norm_std": res_norm.std().item() if resonant_token_count > 0 else 0.0,
                    **({f"batch_resonant_norm_{i}": v for i, v in enumerate(res_norm.mean(dim=0).tolist())}  if resonant_token_count > 0 else {})
                }, step=wandb_step)


            optimizer.zero_grad()
            # removed second backward call
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            actual_updates += 1
            optimizer.step()
            last_resonant_state = res_tokens.detach() if res_tokens is not None and res_tokens.numel() > 0 else None


            total_loss += loss.item()

        if actual_updates > 0:
            avg_loss = total_loss / actual_updates
        else:
            avg_loss = float('inf')
        perplexity = np.exp(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | RI: {ri.item():.4e} | RS: {rs.item():.4e} | Perplexity: {perplexity:.2f}")
        
        if res_tokens is not None and res_tokens.numel() > 0:
            res_epoch_norm = res_tokens.norm(dim=-1)
        else:
            res_epoch_norm = torch.tensor(0.0)        
        

    torch.save(model.state_dict(), "enhanced_resonant_model.pt")
    print("Model saved to enhanced_resonant_model.pt")

else:
    # === Load Tokenizer for Inference ===
    tokenizer_dir = "tokenizer-tinystories"
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_dir}/vocab.json",
        f"{tokenizer_dir}/merges.txt"
    )
    tokenizer.add_special_tokens(["<pad>", "<unk>"])
    pad_id = tokenizer.token_to_id("<pad>")
    unk_id = tokenizer.token_to_id("<unk>")
    tokenizer.post_processor = BertProcessing(("<pad>", pad_id), ("<pad>", pad_id))

    vocab_size = tokenizer.get_vocab_size()

    model = EnhancedResonantTransformer(vocab_size, d_model=32, num_heads=2, num_layers=2, res_tokens=resonant_token_count if not BASELINE else 0)
    model.load_state_dict(torch.load("enhanced_resonant_model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Inference mode. Type a sentence:")
    while True:
        user_input = input("> ")
        if not user_input.strip():
            break

        encoded = tokenizer.encode(user_input)
        tokens = encoded.ids[:sequence_length]
        if len(tokens) < sequence_length:
            tokens += [pad_id] * (sequence_length - len(tokens))

        generated = tokens[:sequence_length]
        with torch.no_grad():
            for _ in range(100):
                input_seq = torch.tensor(generated[-sequence_length:], dtype=torch.long).unsqueeze(0).to(DEVICE)
                logits, _ = model(input_seq, context=input_seq)
                probs = logits[0, -1] / 0.8
                probs = torch.softmax(probs, dim=0)
                topk_probs, topk_indices = torch.topk(probs, 40)
                next_token = topk_indices[torch.multinomial(topk_probs, 1)].item()
                generated.append(next_token)

        output_text = tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)
        output_text = output_text.replace("@ @", "").replace("@", "").replace("Ġ", " ").strip()
        print("Generated continuation:", output_text)
