import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.amp import autocast

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

def repair_if_invalid(x, name="tensor", counter=None):
    if x is None:
        return x
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"[repair_if_invalid] Warning: detected NaNs/Infs in {name}! Repairing…")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = x.clamp(min=-1e4, max=1e4)
        if counter is not None and name in counter:
            counter[name] += 1
    return x

class MultiHeadResonance(nn.Module):
    def __init__(self, num_heads, res_tokens, d_model):
        super().__init__()
        self.resonant_bank = nn.Parameter(torch.randn(num_heads, res_tokens, d_model))
        self.selector = LawfulLinear(d_model, num_heads)
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
        self.linear = LawfulLinear(d_model, res_tokens * d_model)
        self.res_tokens = res_tokens
        self.d_model = d_model

    def forward(self, context_embedding):
        assert context_embedding.shape[-1] == self.d_model, (
            f"Expected input dim {self.d_model}, got {context_embedding.shape[-1]}"
        )
        if self.res_tokens == 0:
            return torch.empty(context_embedding.size(0), 0, self.d_model, device=context_embedding.device)
        if context_embedding.dim() == 1:
            context_embedding = context_embedding.unsqueeze(0)
        out = self.linear(context_embedding).view(-1, self.res_tokens, self.d_model)
        out = F.layer_norm(out, (self.d_model,))
        out = F.normalize(out, dim=-1) * 0.5  # target norm = 1.0 per token
        out = torch.clamp(out, min=-1.0, max=1.0)  # Clamp dynamic resonant tokens
        return out


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0

        self.q_proj = LawfulLinear(d_model, d_model)
        self.k_proj = LawfulLinear(d_model, d_model)
        self.v_proj = LawfulLinear(d_model, d_model)
        self.out_proj = LawfulLinear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = LawfulLinear(d_model, dim_feedforward)
        self.linear2 = LawfulLinear(dim_feedforward, d_model)
        self.activation = nn.GELU()
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
            # mask out padding positions with the dtype-specific minimum (fits float16)
            neg_val = torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores.masked_fill(
                src_key_padding_mask.unsqueeze(1).unsqueeze(2),
                neg_val
            )
            # then clamp into a finite window for stable softmax
            attn_scores = attn_scores.clamp(min=-30.0, max=30.0)

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
                 baseline=False,
                 resonant_token_count=0, dynamic_resonant_token_count=0,
                 multihead=False,
                 max_recursive_steps: int = 3,
                 embedding_dropout: float = 0.1):
        super().__init__()
        self.baseline = baseline
        # embedding dropout
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.global_res = None
        self.surprisal_trajectory = []
        self.attention_trajectory = []
        self.resolution_score = None
        self.embedding = nn.Embedding(vocab_size, d_model)
        # — reinitialize embedding to match LawfulLinear’s Kaiming‐uniform scale —
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))
        self.static_resonant_token_count = resonant_token_count
        self.dynamic_resonant_token_count = dynamic_resonant_token_count
        self.multihead = multihead
        self.max_recursive_steps = max_recursive_steps
        self.d_model = d_model
        self.alpha = 0.0
        self.self_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.token_scale = nn.Parameter(torch.tensor(1.0))
        # learnable gate for blending fresh‐input vs. resonant‐memory contexts
        self.memory_gate_layer = nn.Sequential(
            LawfulLinear(d_model, d_model),
            nn.Sigmoid()
        )

        # Optional segment embedding (2 segments: A and B)
        self.segment_embedding = nn.Embedding(2, d_model)
        # Self-model for recursive state prediction
        self.self_model = SelfModel(hidden_size=self.d_model, depth=3)
        self.self_model_loss_fn = nn.MSELoss()
        self._res_tokens_for_ri = None

        self.flux_budget = None         # No initial limit
        self.excess_flux_count = 0
        self.budget_ema = None          # Moving baseline
        self.budget_alpha = 2.5         # Multiplier over recent average cost
        self.budget_beta = 0.05         # EMA smoothing factor

        self.entropy_ema = torch.tensor(1.0)
        self.attn_kl_ema = torch.tensor(1.0)
        self.resolution_ema = torch.tensor(1.0)
        self.ema_alpha = 0.01  # You can tune this, but it's stable and general
        self.recursion_feedback_strength = 1.0
        self.token_scale = nn.Parameter(torch.full((1, 1, d_model), 0.2))  # per-dimension scaling


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
        # final token‐to‐vocab projection
        self.output = LawfulLinear(d_model, vocab_size)
        # --- Weight tying: share input & output weights ---
        # Ensure the output’s weight_base Parameter references the same data as the embedding
        # (they must have identical shape: [vocab_size, d_model])
        # self.output.weight_base = self.embedding.weight

        # remove the old base, tie in the embedding
        del self.output._parameters['weight_base']
        self.output.register_parameter('weight_base', self.embedding.weight)
        # now that weight_base is tied, re-reset base and bias to Kaiming init
        self.output.reset_parameters()

    def forward(self, x, context=None, update_global=True, padding_mask=None, segment_ids=None):
        emb = self.embedding(x)
        emb = self.embedding_dropout(emb)
        B = emb.size(0)
        self_tok = self.self_token.expand(B, -1, -1)
        emb = torch.cat([self_tok, emb], dim=1)

        if padding_mask is not None:
            B, L = padding_mask.shape
            padding_mask = torch.cat([
                torch.zeros(B, 1, dtype=padding_mask.dtype, device=padding_mask.device),
                padding_mask
            ], dim=1)

        if segment_ids is not None:
            seg_emb = self.segment_embedding(segment_ids)
            emb = emb + seg_emb

        # === Derive context vector ===
        if context is not None:
            if context.dtype in (torch.int64, torch.int32):
                ctx_emb = self.embedding(context)  # [B, L, D]
                ctx_emb = self.embedding_dropout(ctx_emb)
                context_vec = ctx_emb.mean(dim=1)  # [B, D]
            elif context.dim() == 2:
                # Already [B, D], use directly
                context_vec = context
            elif context.dim() == 3:
                context_vec = context.mean(dim=1)  # [B, D]
            else:
                raise ValueError(f"[forward] Unsupported context shape: {context.shape}")
        else:
            if self.global_res is not None:
                context_vec = self.global_res.mean(dim=1).expand(x.size(0), -1).contiguous()  # [B, D]
            else:
                context_vec = self.self_token.expand(x.size(0), -1, -1).mean(dim=1)  # [B, D]

        # Assert safety
        assert context_vec.shape[-1] == self.d_model, f"context_vec has invalid last dim: {context_vec.shape[-1]}"


        # Gather resonant token sources
        res_list = []
        if self.static_resonant_token_count > 0:
            # use an expand-view directly (no clone) so grads flow back into the param
            static = self.resonant_tokens.expand(B, -1, -1)
            res_list.append(static)

        if self.dynamic_resonant_token_count > 0:
            dyn = self.controller(context_vec)
            self.dynamic_tokens_latest = dyn.detach()
            res_list.append(dyn)
        else:
            self.dynamic_tokens_latest = None

        if self.multihead:
            mh = self.resonator(context_vec)
            res_list.append(mh)

        # If no resonant tokens are configured, create an empty placeholder
        if res_list:
            tokens = torch.cat(res_list, dim=1)
            tokens = F.normalize(tokens, dim=-1) * 0.3
            # Apply clamped, learnable per-dim scaling
            gate = self.token_scale.clamp(0.0, 1.0)  # shape: [1, 1, D]
            tokens = tokens * gate

        else:
            tokens = torch.empty(B, 0, self.d_model, device=emb.device)

        if padding_mask is not None:
            num_res_tokens = tokens.size(1)
            padding_mask = torch.cat([
                torch.zeros(B, num_res_tokens, dtype=padding_mask.dtype, device=padding_mask.device),
                padding_mask
            ], dim=1)

        if tokens.numel() > 0:
            # ensure tokens require gradients so we can retain them for RI/RS
            self._res_tokens_for_ri = tokens.requires_grad_(True)
            self._res_tokens_for_ri.retain_grad()
        else:
            self._res_tokens_for_ri = tokens  # empty tensor

        # **Use** self._res_tokens_for_ri in the forward pass so gradients land there**
        if self.training and self._res_tokens_for_ri.numel() > 0:
            # ramp influence by alpha
            blended_tokens = (self._res_tokens_for_ri.detach() * (1 - self.alpha)
                              + self._res_tokens_for_ri * self.alpha)
        else:
            blended_tokens = tokens

        inp = torch.cat([blended_tokens, emb], dim=1)
        # ensure padding_mask spans the full sequence (resonant tokens + self-token + input)
        if padding_mask is not None:
            B, old_len = padding_mask.shape
            new_len = inp.size(1)
            pad_count = new_len - old_len
            padding_mask = torch.cat([
                torch.zeros(B, pad_count, dtype=padding_mask.dtype, device=padding_mask.device),
                padding_mask
            ], dim=1)

        # Normalize after blending resonant tokens and input embedding
        inp = F.layer_norm(inp, (self.d_model,))

        attn_maps = []
        out = inp
        for layer in self.encoder_layers:
            out, weights = layer(out, src_key_padding_mask=padding_mask)
            attn_maps.append(weights)

        # Extract sequence hidden states (excluding resonant tokens)
        seq_out = out[:, tokens.size(1):, :]
        hidden = seq_out
        logits = self.output(hidden)
        # Clamp logits before using them to prevent exploding values
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        res = tokens

        # Update global_res using EMA only if flagged
        if update_global and res.numel() > 0:
            momentum = 0.9
            batch_mean_res = res.mean(dim=0, keepdim=True)  # (1, R, D)

            # Collapse detection
            if self.global_res is not None:
                gr_norm = self.global_res.norm().item()
                gr_std = self.global_res.std().item()
                if gr_norm < 1e-3 or gr_std < 1e-3:
                    print("[warn] global_res collapsed. Reinitializing from batch...")
                    self.global_res = batch_mean_res.detach()
                else:
                    self.global_res = momentum * self.global_res + (1 - momentum) * batch_mean_res.detach()
            else:
                self.global_res = batch_mean_res.detach()

        return logits, res, attn_maps, hidden, out  # <--- include full transformer output

    def recursive_forward(self, x, context=None, max_steps=None, tol=1e-5, padding_mask=None, inference_mode=False, fade_in_strength=1.0):
        """
        Full recursive inference with:
        - entropy / resolution / attention tracking
        - self-modeling loss
        - RAF modulation
        - adaptive early halting (via tolerance)
        """

        if inference_mode:
            training_was_enabled = self.training
            self.eval()
            torch.set_grad_enabled(False)

        training_was_enabled = self.training
        self.eval()

        self.surprisal_trajectory.clear()
        self.attention_trajectory.clear()

        num_steps = max_steps if max_steps is not None else self.max_recursive_steps
        ema_decay = 0.9  # controls smoothing; higher = slower update

        entropy_deltas = []
        attention_kls = []
        resolution_scores = []
        past_internal_states = []

        resolution_score = None

        previous_entropy = None
        previous_attention = None
        previous_resolution = None

        logits = None
        hidden = None
        res = None
        attn_maps = None

        self.last_recursive_steps = 1
        # self.excess_flux_count = 0

        total_flux_penalty = torch.tensor(0.0, device=x.device)  # Accumulate excess flux

        previous_hidden = None
        hidden_steps = []  # track hidden states across steps

        for step in range(num_steps):
            res_token_delta = hidden_delta = attn_delta = None
            feedback_strength = 1.0
            logits, res, attn_maps, hidden, full_out = self.forward(x, context=context, update_global=False, padding_mask=padding_mask)

            # logits = repair_if_invalid(logits, name="logits", counter=self.repair_counter if hasattr(self, 'repair_counter') else None)
            # res = repair_if_invalid(res, name="resonant output", counter=self.repair_counter if hasattr(self, 'repair_counter') else None)
            # hidden = repair_if_invalid(hidden, name="hidden state", counter=self.repair_counter if hasattr(self, 'repair_counter') else None)
            # context = repair_if_invalid(context, name="context vector", counter=self.repair_counter if hasattr(self, 'repair_counter') else None)

            # === Inject low-rank structured noise into context ===
            if self.training and context is not None:
                structured_noise = (torch.randn_like(context) * 0.05)
                structured_noise = F.normalize(structured_noise, dim=-1) * 0.05
                context = context + structured_noise

            # === Tiny noise injection to hidden states to maintain diversity ===
            if self.training:
                noise_strength = 1e-3  # adjustable: lower = safer
                hidden = hidden + noise_strength * torch.randn_like(hidden)

            # === Soft fade-in for all recursive steps after warmup ===
            if previous_hidden is not None:
                feedback_strength = self.recursion_feedback_strength * fade_in_strength
                hidden = (1.0 - feedback_strength) * previous_hidden + feedback_strength * hidden

            # Normalize hidden immediately after blending
            hidden = F.layer_norm(hidden, (hidden.size(-1),))
            hidden_steps.append(hidden.detach())  # store step hidden
            
            previous_hidden = hidden.detach()  # update stored hidden

            # Normalize hidden state to control norm drift
            hidden = F.layer_norm(hidden, (hidden.size(-1),))

            # === Entropy / surprisal tracking ===
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            self.surprisal_trajectory.append(entropy.item())

            if previous_entropy is not None:
                entropy_deltas.append((entropy - previous_entropy).abs())
            previous_entropy = entropy

            # === Attention KL divergence ===
            if previous_attention is not None and attn_maps:
                current_attn = attn_maps[-1]
                self.attention_trajectory.append(current_attn.mean().item())
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(current_attn, dim=-1),
                    torch.softmax(previous_attention, dim=-1),
                    reduction='batchmean'
                )
                attention_kls.append(kl)
            if attn_maps:
                previous_attention = attn_maps[-1]
                self.attention_trajectory.append(previous_attention.mean().item())

            # === Resolution score via recursive delta ===
            if step == 0:
                previous_res = res.detach()
                resolution_score = torch.tensor(0.0, device=res.device)
                self.resolution_score = resolution_score
            else:
                # Δresonant token change
                res_token_delta = torch.norm(res - previous_res, dim=-1).mean()
                previous_res = res.detach()

                # Track as resolution score
                resolution_score = res_token_delta
                self.resolution_score = resolution_score

                # Δhidden state change
                hidden_delta = torch.norm(hidden - previous_hidden, dim=-1).mean()

                # Δattention change
                if attn_maps and previous_attention is not None:
                    cur_attn = attn_maps[-1]
                    prev_attn = previous_attention
                    attn_delta = torch.norm(cur_attn - prev_attn, dim=-1).mean()

                # === Composite delta ===
                deltas = [res_token_delta, hidden_delta, attn_delta]
                delta_vals = [d.item() for d in deltas if d is not None]
                composite_delta = sum(delta_vals) / len(delta_vals)

                if composite_delta < tol:
                    print(f"Converged early at step {step}: Δ*={composite_delta:.2e} < tol={tol}")
                    break
            
            resolution_scores.append(resolution_score)

            eps = 1e-8  # Prevent divide-by-zero

            # --- Get raw values ---
            entropy_raw = entropy_deltas[-1] if entropy_deltas else torch.tensor(0.0, device=x.device)
            attn_raw = attention_kls[-1] if attention_kls else torch.tensor(0.0, device=x.device)
            res_raw = torch.abs(resolution_score - previous_resolution) if previous_resolution is not None else torch.tensor(0.0, device=x.device)

            # --- Update EMAs ---
            self.entropy_ema = (1 - self.ema_alpha) * self.entropy_ema + self.ema_alpha * entropy_raw.detach()
            self.attn_kl_ema = (1 - self.ema_alpha) * self.attn_kl_ema + self.ema_alpha * attn_raw.detach()
            self.resolution_ema = (1 - self.ema_alpha) * self.resolution_ema + self.ema_alpha * res_raw.detach()

            # --- Compute normalized flux terms ---
            norm_entropy = torch.abs(entropy_raw - self.entropy_ema) / (self.entropy_ema + eps)
            norm_attn = torch.abs(attn_raw - self.attn_kl_ema) / (self.attn_kl_ema + eps)
            norm_res = torch.abs(res_raw - self.resolution_ema) / (self.resolution_ema + eps)

            flux_cost = norm_entropy + norm_attn + norm_res
            flux_cost = flux_cost.mean()

            # === Self-calibrate dynamic flux budget ===
            if self.budget_ema is None:
                self.budget_ema = flux_cost.item()
                self.flux_budget = self.budget_alpha * self.budget_ema
            else:
                self.budget_ema = (1 - self.budget_beta) * self.budget_ema + self.budget_beta * flux_cost.item()
                self.flux_budget = self.budget_alpha * self.budget_ema

            self.last_flux_cost = flux_cost.item()

            # === Soft penalty instead of hard cutoff ===
            excess_flux = flux_cost - self.flux_budget
            excess_flux_penalty = torch.relu(excess_flux)
            if excess_flux_penalty.item() > 0:
                self.excess_flux_count += 1
            total_flux_penalty += excess_flux_penalty

            # === Hidden state for self-modeling ===
            past_internal_states.append(hidden[:, 0, :].detach())
            if len(past_internal_states) > self.self_model.depth:
                past_internal_states.pop(0)

            # === Update context_vec via EMA ===
            if not self.baseline and res is not None and res.numel() > 0:
                res_tokens_only = full_out[:, :res.size(1), :].detach()  # [B, R, D]
                new_context_vec = res_tokens_only.mean(dim=1)
            else:
                # Recompute context from input tokens safely
                ctx_emb = self.embedding(x)  # [B, L, D]
                new_context_vec = ctx_emb.mean(dim=1)  # [B, D]

            if context is not None:
                if context.dtype in (torch.int64, torch.int32):
                    ctx_emb = self.embedding(context)
                    old_context_vec = ctx_emb.mean(dim=1)  # [B, D]
                elif context.dim() == 2:
                    old_context_vec = context  # already [B, D]
                elif context.dim() == 3:
                    old_context_vec = context.mean(dim=1)  # [B, D]
                else:
                    raise ValueError(f"[recursive_forward] Unsupported context shape: {context.shape}")
            else:
                old_context_vec = new_context_vec

            context_vec_ema = ema_decay * old_context_vec + (1 - ema_decay) * new_context_vec
            context_vec_ema = (1.0 - feedback_strength) * old_context_vec + feedback_strength * context_vec_ema
            context_vec_ema = F.layer_norm(context_vec_ema, (context_vec_ema.size(-1),))

            # === Track context flux for bonus ===
            flux_movement = (new_context_vec - old_context_vec).pow(2).mean()
            self.latest_context_flux = flux_movement
            context = context_vec_ema.detach()  # Update for next step

            # Normalize context after EMA update
            context = F.layer_norm(context, (context.size(-1),))

            # Assert
            assert context.shape[-1] == self.d_model, f"Context shape mismatch: expected {self.d_model}, got {context.shape}"



        # === Self-model prediction and loss ===
        if len(past_internal_states) == self.self_model.depth:
            past_tensor = torch.stack(past_internal_states, dim=1)  # (batch, depth, hidden)
            predicted_next = self.self_model(past_tensor)
            target_next = hidden[:, 0, :].detach()
            self_model_loss = self.self_model_loss_fn(predicted_next, target_next)
        else:
            self_model_loss = torch.tensor(0.0, device=hidden.device)
        self.last_self_model_loss = self_model_loss

        self.last_recursive_steps = step + 1

        # === RAF modulation from recursive flux ===
        if entropy_deltas and attention_kls:
            recursive_flux = torch.stack(entropy_deltas).mean() + torch.stack(attention_kls).mean()
            modulation_signal = compute_modulation_signal(recursive_flux)
            # guard against NaNs/Infs from FP16
            modulation_signal = torch.nan_to_num(
                modulation_signal,
                nan=0.0,        # replace NaN with 0 (no modulation)
                posinf=1.0,     # if +Inf, treat as full modulation
                neginf=0.0      # if –Inf, treat as zero modulation
            ).clamp(0.0, 1.0)

            for module in self.modules():
                if isinstance(module, LawfulLinear):
                    module.raf_modulation = float(modulation_signal.item())

        # === Global resonant token update ===
        if res is not None and res.numel() > 0:
            momentum = 0.9
            batch_mean_res = res.mean(dim=0, keepdim=True)
            if self.global_res is None:
                self.global_res = batch_mean_res.detach()
            else:
                self.global_res = momentum * self.global_res + (1 - momentum) * batch_mean_res.detach()

        if training_was_enabled:
            self.train()        
        
        if inference_mode and training_was_enabled:
            self.train()
            torch.set_grad_enabled(True)

        hidden_contrastive_loss = torch.tensor(0.0, device=x.device)
        if len(hidden_steps) >= 2:
            for i in range(len(hidden_steps) - 1):
                diff = (hidden_steps[i] - hidden_steps[i + 1]).pow(2).mean()
                hidden_contrastive_loss += diff
            hidden_contrastive_loss = hidden_contrastive_loss / (len(hidden_steps) - 1)

        return logits, hidden, total_flux_penalty.detach(), hidden_contrastive_loss.detach()




class SelfModel(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super().__init__()
        self.depth = depth
        self.linear = nn.Sequential(
            LawfulLinear(hidden_size * depth, hidden_size),  # <-- update this line
            nn.ReLU(),
            LawfulLinear(hidden_size, hidden_size)
        )

    def forward(self, past_states):  # shape: (batch, depth, hidden)
        x = past_states.reshape(past_states.size(0), -1)  # flatten depth × hidden
        return self.linear(x)


class LawfulLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Frozen weights (pretrained or randomly initialized)
        self.weight_base = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias_base = nn.Parameter(torch.empty(out_features), requires_grad=False) if bias else None

        # Learnable delta (lawful recursive updates)
        self.delta_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.delta_bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.raf_modulation = 1.0  # Default modulation factor

        self.device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_base, a=5 ** 0.5)
        if self.bias_base is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_base)
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias_base, -bound, bound)

    def forward(self, x):
        # Fully run this layer in FP32: disable AMP and cast input to float32
        with autocast(device_type=self.device, enabled=False):
            weight = self.weight_base + self.delta_weight * self.raf_modulation
            if self.bias_base is not None:
                bias = self.bias_base + self.delta_bias * self.raf_modulation
            else:
                bias = None
            # cast the incoming tensor to FP32 so mat1/mat2 dtypes match
            x_fp32 = x.float()
            return F.linear(x_fp32, weight, bias)




def contrastive_loss(original_logits, contrast_logits, margin=1.0):
    """
    Soft‐hinge contrastive: F.softplus(margin – dist) keeps a smooth, nonzero gradient
    even once dist > margin.
    """
    orig_repr     = original_logits.mean(dim=1)
    contrast_repr = contrast_logits.mean(dim=1)
    dist          = F.pairwise_distance(orig_repr, contrast_repr, p=2)
    loss          = F.softplus(margin - dist)    # => always > 0, grad = sigmoid(margin - dist)
    return loss.mean()

def inject_resonant_noise(tokens, noise_level=0.1):
    noise = torch.randn_like(tokens)
    noise = F.normalize(noise, dim=-1)
    return tokens + noise_level * noise

def diversity_penalty(x):
    # sanitize in-graph
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    x = F.normalize(x, dim=-1, eps=1e-6)
    b,t,d = x.shape
    sim = torch.einsum('btd,bkd->btk', x, x)
    mask = torch.eye(t, device=x.device).bool().unsqueeze(0)
    sim = sim.masked_fill(mask, 0.0)
    return sim.pow(2).mean()

def cosine_rampup(t, warmup_epochs):
    if t >= warmup_epochs:
        return 1.0
    return 0.5 * (1 - torch.cos(torch.tensor(torch.pi * t / warmup_epochs)))

def compute_recursive_flux(entropy_deltas, attn_kls):
    # You can later weight these by importance
    return entropy_deltas.mean(dim=-1) + attn_kls.mean(dim=-1)

def compute_modulation_signal(recursive_flux, threshold=0.5):
    # Sigmoid-shaped scaling function
    return torch.tanh((recursive_flux - threshold) * 5.0).clamp(0.0, 1.0)
