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

# === Hyperparameters & Config ===
from config import (
    d_model, num_heads, num_layers,
    resonant_token_count, dynamic_resonant_token_count,
    sequence_length, max_tokens, learning_rate,
    batch_size, num_epochs, warmup_epochs,
    lambda_ri, lambda_rs, lambda_div,
    lambda_sur, lambda_attn, lambda_res,
    multihead_resonance
)

# Device setup
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else (
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Initialize wandb
wandb.init(project="resonant-transformer-punchline", config={
    **{k: v for k, v in locals().items() if k.startswith('lambda_') or k in [
        'd_model','num_heads','num_layers','resonant_token_count',
        'dynamic_resonant_token_count','learning_rate','batch_size',
        'num_epochs','sequence_length','max_tokens','multihead_resonance'
    ]}
})

# Data preparation (unchanged)
pca_data_buffer = deque(maxlen=100 * batch_size)
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
        tokens.extend(tokenizer.encode(line.strip()).ids)
        if len(tokens) >= max_tokens:
            break
sequences = torch.tensor([tokens[i:i+sequence_length] for i in range(0, len(tokens)-sequence_length, sequence_length)], dtype=torch.long).to(DEVICE)
loader = DataLoader(sequences, batch_size=batch_size, shuffle=True, drop_last=True)

# Model initialization
model = EnhancedResonantTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    resonant_token_count=resonant_token_count,
    dynamic_resonant_token_count=dynamic_resonant_token_count,
    multihead=multihead_resonance
).to(DEVICE)
model.train()

# Attention hook setup: register on custom encoder_layers
attn_records = []
def attn_hook(module, inp, output):
    # output is (attn_output, attn_weights)
    if isinstance(output, tuple) and output[1] is not None:
        attn_records.append(output[1].detach())
for layer in model.encoder_layers:
    layer.self_attn.register_forward_hook(attn_hook)
avg_attn = None
attn_momentum = 0.99

# Optimizer & Criterion
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
crit = nn.CrossEntropyLoss(ignore_index=pad_id)
last_res=None

# Training
for epoch in range(num_epochs):
    model.alpha = cosine_rampup(epoch, warmup_epochs)
    for bidx,batch in enumerate(loader):
        inp = batch[:,:-1]; tgt = batch[:,1:]
        ctx = last_res if last_res is not None else inp
        logits, res = model(inp, context=ctx)
        logits = logits[:,:inp.size(1)]
        # Primary loss (CE)
        primary = crit(logits.reshape(-1,logits.size(-1)), tgt.reshape(-1))
        # Surprisal-drop
        lp = F.log_softmax(logits,dim=-1)
        tlp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        spr = -tlp; dr = spr[:,:-1]-spr[:,1:]
        sur = F.relu(dr).mean()
        primary = primary - lambda_sur*sur
        # Attention KL
        if attn_records:
            ca = torch.stack(attn_records).mean(0)
            avg_attn = ca.mean(0) if avg_attn is None else attn_m*avg_attn+(1-attn_m)*ca.mean(0)
            cur,avg = ca.mean(0)+1e-8, avg_attn+1e-8
            akl = F.kl_div(cur.log(),avg,reduction='batchmean')
            primary = primary - lambda_attn*akl
            attn_records.clear()
        else: akl=torch.tensor(0.,device=DEVICE)
        # Resolution-coherence
        if logits.size(1)>=2:
            pr = lp.exp(); ent=-(pr*lp).sum(-1)
            pen, pos = ent[:,-2], ent[:,-1]
            rr = F.relu(pen-pos).mean()
            primary = primary - lambda_res*rr
        else: rr=torch.tensor(0.,device=DEVICE)
        # Contrastive
        ci = inp.clone(); ci[:,-1]=torch.randint(0,tokenizer.get_vocab_size(),(batch_size,),device=DEVICE)
        coh,_=model(ci,ci); con=contrastive_loss(logits,coh)
        primary = primary + con
        # RI/RS/diversity on primary only
        final = primary
        if res.numel()>0 and hasattr(model,'_res_tokens_for_ri'):
            gr = torch.autograd.grad(primary, model._res_tokens_for_ri, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if gr is not None:
                gn=gr.norm(-1).clamp(1e-6)
                ri_v=(gr*model._res_tokens_for_ri).sum(-1).abs()/gn; term1=lambda_ri*ri_v.mean()
                rs_v=1-F.cosine_similarity(gr,model._res_tokens_for_ri,dim=-1); term2=lambda_rs*rs_v.mean()
            else: term1=term2=0.
            dvt=lambda_div*diversity_penalty(model._res_tokens_for_ri)
            final = primary - term1 - term2 + dvt
        # Backprop
        opt.zero_grad(); final.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        # update
        last_res = getattr(model,'_res_tokens_for_ri',None)
        if last_res is not None: last_res=last_res.detach()
        # log
        if bidx%10==0:
            wandb.log({
                'loss':final.item(),
                'perplexity':float(np.exp(final.item())),
                'surprisal_reward':sur.item(),
                'attention_kl':akl.item(),
                'resolution_reward':rr.item()
            })
            print(f"E{epoch+1} B{bidx} P{float(np.exp(final.item()))} L{final.item():.4f} S{sur.item():.4f} A{akl.item():.4f} R{rr.item():.4f}")

# Save final state
state = {
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': tokenizer.get_vocab_size(),
        'sequence_length': sequence_length,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'resonant_token_count': resonant_token_count,
        'dynamic_resonant_token_count': dynamic_resonant_token_count,
        'multihead': multihead_resonance
    },
    'final_resonant_state': last_res_state
}
millions = int(max_tokens / 1_000_000)
model_filename = f"{resonant_token_count}-{dynamic_resonant_token_count}-{d_model}-{num_heads}-{num_layers}-{sequence_length}-{millions}M.pt"
torch.save(state, model_filename)
print(f"Model saved to {model_filename} with config and final dynamic resonant state embedded.")
