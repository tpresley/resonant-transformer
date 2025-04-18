import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if self.res_tokens == 0:
            return torch.empty(context_embedding.size(0), 0, self.d_model, device=context_embedding.device)
        if context_embedding.dim() == 1:
            context_embedding = context_embedding.unsqueeze(0)
        out = self.linear(context_embedding).view(-1, self.res_tokens, self.d_model)
        out = F.layer_norm(out, (self.d_model,))
        return out

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class EnhancedResonantTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 resonant_token_count=0, dynamic_resonant_token_count=0,
                 multihead=False,
                 max_recursive_steps: int = 3):
        super().__init__()
        self.global_res = None
        self.surprisal_trajectory = []
        self.attention_trajectory = []
        self.resolution_score = None
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.static_resonant_token_count = resonant_token_count
        self.dynamic_resonant_token_count = dynamic_resonant_token_count
        self.multihead = multihead
        self.max_recursive_steps = max_recursive_steps
        self.d_model = d_model
        self.alpha = 0.0
        self.self_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        if self.dynamic_resonant_token_count > 0:
            self.controller = ResonantController(d_model, self.dynamic_resonant_token_count)
        if self.static_resonant_token_count > 0:
            self.resonant_tokens = nn.Parameter(torch.randn(1, self.static_resonant_token_count, d_model))
        if self.multihead:
            self.resonator = MultiHeadResonance(num_heads=num_heads,
                                                res_tokens=resonant_token_count,
                                                d_model=d_model)
        layers = [CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        ) for _ in range(num_layers)]
        self.encoder_layers = nn.ModuleList(layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, context=None):
        emb = self.embedding(x)
        B = emb.size(0)
        self_tok = self.self_token.expand(B, -1, -1)
        emb = torch.cat([self_tok, emb], dim=1)

        # Compute context vector
        if context is not None:
            if context.dtype in (torch.int64, torch.int32):
                ctx_emb = self.embedding(context)
            else:
                ctx_emb = context
            context_vec = ctx_emb.mean(dim=1)
        elif self.global_res is not None:
            # global_res shape: (1, T, D), so mean -> (1, D), expand to (B, D)
            context_vec = self.global_res.mean(dim=1).expand(x.size(0), -1).contiguous()
        else:
            # [SELF] fallback — shape (1, 1, D), expand then mean -> (B, D)
            context_vec = self.self_token.expand(x.size(0), -1, -1).mean(dim=1)

        res_list = []
        if self.static_resonant_token_count > 0:
            # replicate static bank across the batch
            static = self.resonant_tokens.expand(B, -1, -1)
            res_list.append(static)
        if self.dynamic_resonant_token_count > 0:
            # controller generates dynamic tokens from context
            dyn = self.controller(context_vec)
            res_list.append(dyn)
        if self.multihead:
            # multi‑head mixture of static bank
            mh = self.resonator(context_vec)
            res_list.append(mh)

        tokens = torch.cat(res_list, dim=1)

        if self.training and tokens.numel() > 0 and tokens.requires_grad:
            tokens.retain_grad()
            self._res_tokens_for_ri = tokens

        inp = torch.cat([tokens.detach() * (1 - self.alpha) + tokens * self.alpha, emb], dim=1)
        attn_maps = []
        out = inp
        for layer in self.encoder_layers:
            out, weights = layer(out)
            attn_maps.append(weights)

        seq_out = out[:, tokens.size(1):, :]
        logits = self.output(seq_out)
        res = tokens

        # Update global_res using EMA
        momentum = 0.9
        batch_mean_res = res.mean(dim=0, keepdim=True)  # shape: (1, res_tokens, d_model)
        if self.global_res is None:
            self.global_res = batch_mean_res.detach()
        else:
            self.global_res = momentum * self.global_res + (1 - momentum) * batch_mean_res.detach()

        return logits, res, attn_maps

    def recursive_forward(self, x, max_steps=3, tol=1e-3):
        self.surprisal_trajectory.clear()
        self.attention_trajectory.clear()

        prev_res = None
        res = None
        logits = None

        prev_score = None
        epsilon = 1e-3  # Resolution convergence threshold

        for step in range(max_steps):
            if max_steps is None:
                max_steps = self.max_recursive_steps
            logits, res, attn_maps = self.forward(x)

            if logits is not None:
                self.surprisal_trajectory.append(logits.detach())
            if attn_maps:
                self.attention_trajectory.append(attn_maps[-1].detach())

            if len(self.surprisal_trajectory) > 1:
                init = self.surprisal_trajectory[0]
                final = self.surprisal_trajectory[-1]
                self.resolution_score = (init - final).norm().item()

                if prev_score is not None and abs(prev_score - self.resolution_score) < epsilon:
                    break  # stop early due to semantic convergence

                prev_score = self.resolution_score

            if prev_res is not None and res is not None:
                delta = (res - prev_res).norm()
                if delta < tol:
                    break
            prev_res = res.detach()

        self.last_recursive_steps = step + 1
        return logits, res

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
    normed = F.normalize(tokens, dim=-1)
    sim_matrix = torch.einsum('btd,bkd->btk', normed, normed)
    eye = torch.eye(sim_matrix.size(-1), device=sim_matrix.device).unsqueeze(0)
    sim_matrix = sim_matrix * (1 - eye)
    penalty = sim_matrix.sum(dim=(1, 2)) / (tokens.size(1) * (tokens.size(1) - 1))
    return penalty.mean()

def cosine_rampup(t, warmup_epochs):
    if t >= warmup_epochs:
        return 1.0
    return 0.5 * (1 - torch.cos(torch.tensor(torch.pi * t / warmup_epochs)))
