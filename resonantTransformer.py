import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(q, k, seq_dim=1):
    """
    Applies rotary positional embedding to query and key tensors.
    Assumes input shapes [B, L, H, D] or [L, B, H, D] depending on usage.
    """
    d = q.size(-1)
    half = d // 2
    freqs = torch.exp(-torch.arange(0, half, dtype=torch.float32, device=q.device) * (math.log(10000.0) / half))
    pos = torch.arange(q.size(seq_dim), device=q.device, dtype=torch.float32)
    sinusoid = torch.einsum("l,d->ld", pos, freqs)
    sin = sinusoid.sin()[None, :, None, :]
    cos = sinusoid.cos()[None, :, None, :]

    def rotate(t):
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat([t1 * cos - t2 * sin, t2 * cos + t1 * sin], dim=-1)

    return rotate(q), rotate(k)

class MultiHeadResonance(nn.Module):
    def __init__(self, num_heads, res_tokens, d_model):
        super().__init__()
        self.resonant_bank = nn.Parameter(torch.randn(num_heads, res_tokens, d_model))
        self.selector = nn.Linear(d_model, num_heads)
        self.num_heads = num_heads
        self.res_tokens = res_tokens
        self.d_model = d_model

    def forward(self, context_vec):
        weights = torch.softmax(self.selector(context_vec), dim=-1)  # shape: [B, H]
        weighted_tokens = torch.einsum('bh,hnd->bnd', weights, self.resonant_bank)  # [B, N, D]

        # Normalize each token vector (across the feature dimension)
        normed = F.normalize(weighted_tokens, dim=-1)  # ensures unit-norm per token
        # if self.training:
        #     print("Multihead output norm (per token):", normed.norm(dim=-1).mean().item())

        # Optional: Scale each token to a desired magnitude (e.g., 1.0)
        scaled = normed * 0.5

        return scaled  # shape: [B, N, D]

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
        out = F.normalize(out, dim=-1) * 0.5  # target norm = 1.0 per token
        return out


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        B, L, D = src.size()
        H = self.nhead
        Dh = self.head_dim

        q = self.q_proj(src).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        k = self.k_proj(src).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(src).view(B, L, H, Dh).transpose(1, 2)

        # === RoPE here ===
        q, k = apply_rope(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B, H, L, L]
        if src_mask is not None:
            attn_scores += src_mask.unsqueeze(1)
        if src_key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                src_key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v)  # [B, H, L, Dh]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        src2 = self.out_proj(attn_output)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)
        return src, attn_weights.mean(dim=1)  # average heads


class EnhancedResonantTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 resonant_token_count=0, dynamic_resonant_token_count=0,
                 hidden_dim=128,
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
        self.token_scale = nn.Parameter(torch.tensor(1.0))
        # Self-model for recursive state prediction
        self.self_model = SelfModel(hidden_size=self.d_model, depth=3)
        self.self_model_loss_fn = nn.MSELoss()


        if self.dynamic_resonant_token_count > 0:
            self.controller = ResonantController(d_model, self.dynamic_resonant_token_count)
        if self.static_resonant_token_count > 0:
            self.resonant_tokens = nn.Parameter(
                torch.randn(1, self.static_resonant_token_count, d_model) * 0.01
            )
        if self.multihead:
            self.resonator = MultiHeadResonance(num_heads=num_heads,
                                                res_tokens=resonant_token_count,
                                                d_model=d_model)
        layers = [CustomTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1
        ) for _ in range(num_layers)]
        self.encoder_layers = nn.ModuleList(layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, context=None, update_global=True):
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
            context_vec = self.global_res.mean(dim=1).expand(x.size(0), -1).contiguous()
        else:
            context_vec = self.self_token.expand(x.size(0), -1, -1).mean(dim=1)

        # Gather resonant token sources
        res_list = []
        if self.static_resonant_token_count > 0:
            static = self.resonant_tokens.expand(B, -1, -1)
            res_list.append(static)
        if self.dynamic_resonant_token_count > 0:
            dyn = self.controller(context_vec)
            res_list.append(dyn)
        if self.multihead:
            mh = self.resonator(context_vec)
            res_list.append(mh)

        # If no resonant tokens are configured, create an empty placeholder
        if res_list:
            # if self.training:
            #     for i, t in enumerate(res_list):
            #         print(f"Token source {i} norm: {t.norm().item():.4f}")
            #     print(f"Token scale: {self.token_scale.item():.4f}")
            safe_scale = self.token_scale.clamp(min=1.0, max=3.0)
            tokens = torch.cat(res_list, dim=1) * safe_scale

        else:
            tokens = torch.empty(B, 0, self.d_model, device=emb.device)

        if self.training and tokens.numel() > 0 and tokens.requires_grad:
            tokens.retain_grad()
            self._res_tokens_for_ri = tokens

        inp = torch.cat([tokens.detach() * (1 - self.alpha) + tokens * self.alpha, emb], dim=1)
        attn_maps = []
        out = inp
        for layer in self.encoder_layers:
            out, weights = layer(out)
            attn_maps.append(weights)

        # Extract sequence hidden states (excluding resonant tokens)
        seq_out = out[:, tokens.size(1):, :]
        hidden = seq_out
        logits = self.output(hidden)
        res = tokens

        # Update global_res using EMA only if flagged
        if update_global and res.numel() > 0:
            momentum = 0.9
            batch_mean_res = res.mean(dim=0, keepdim=True)  # (1, res_tokens, d_model)
            if self.global_res is None:
                self.global_res = batch_mean_res.detach()
            else:
                self.global_res = momentum * self.global_res + (1 - momentum) * batch_mean_res.detach()

        return logits, res, attn_maps, hidden

    def recursive_forward(self, x, context=None, max_steps=None, tol=1e-5):
        """
        Recursive inference: update context per iteration so resonant tokens can evolve.
        """
        self.surprisal_trajectory.clear()
        self.attention_trajectory.clear()

        prev_res = None
        prev_score = None
        epsilon = 1e-2
        current_context = context

        if max_steps is None:
            max_steps = self.max_recursive_steps

        current_context = context
        past_internal_states = []
        for step in range(max_steps):
            # run forward but don’t update global_res; feed in current_context
            logits, res, attn_maps, hidden = self.forward(x, current_context, update_global=False)

            # record scalar summaries instead of full tensors
            self.surprisal_trajectory.append(logits.detach().norm().item())
            if attn_maps:
                self.attention_trajectory.append(attn_maps[-1].detach().mean().item())

            # compute resolution score
            if len(self.surprisal_trajectory) > 1:
                init = self.surprisal_trajectory[0]
                final = self.surprisal_trajectory[-1]
                # simple absolute difference for scalar logs
                self.resolution_score = abs(init - final)
                if prev_score is not None and abs(prev_score - self.resolution_score) < epsilon:
                    break
                prev_score = self.resolution_score

            # check token convergence
            if prev_res is not None:
                delta = (res - prev_res).norm()
                if delta < tol:
                    break
            prev_res = res.detach()

            # update context for next iteration
            current_context = hidden.detach()

            hidden_state_t = hidden[:, 0, :]  # track first token (position 0) as representative
            past_internal_states.append(hidden_state_t.detach())
            if len(past_internal_states) > self.self_model.depth:
                past_internal_states.pop(0)


        if len(past_internal_states) == self.self_model.depth:
            past_tensor = torch.stack(past_internal_states, dim=1)  # shape: (batch, depth, hidden)
            predicted_next = self.self_model(past_tensor)
            target_next = hidden[:, 0, :].detach()  # final real internal state
            self_model_loss = self.self_model_loss_fn(predicted_next, target_next)
        else:
            self_model_loss = torch.tensor(0.0, device=hidden.device)

        self.last_self_model_loss = self_model_loss


        # Store it on self for training access
        self.last_self_model_loss = self_model_loss

        # Single EMA update after recursion
        momentum = 0.9
        batch_mean_res = res.mean(dim=0, keepdim=True)
        if self.global_res is None:
            self.global_res = batch_mean_res.detach()
        else:
            self.global_res = momentum * self.global_res + (1 - momentum) * batch_mean_res.detach()

        if res.numel() > 0:
            momentum = 0.9
            batch_mean_res = res.mean(dim=0, keepdim=True)
            if self.global_res is None:
                self.global_res = batch_mean_res.detach()
            else:
                self.global_res = momentum * self.global_res + (1 - momentum) * batch_mean_res.detach()

        return logits, res

class SelfModel(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super().__init__()
        self.depth = depth
        self.linear = nn.Sequential(
            nn.Linear(hidden_size * depth, hidden_size),  # <-- update this line
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, past_states):  # shape: (batch, depth, hidden)
        x = past_states.reshape(past_states.size(0), -1)  # flatten depth × hidden
        return self.linear(x)



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
