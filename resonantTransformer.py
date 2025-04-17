# resonantTransformer.py (fixed to align batch_first and return attention weights)
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

class EnhancedResonantTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 resonant_token_count=0, dynamic_resonant_token_count=0,
                 multihead=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.static_resonant_token_count = resonant_token_count
        self.dynamic_resonant_token_count = dynamic_resonant_token_count
        self.multihead = multihead
        self.d_model = d_model
        self.alpha = 0.0

        if self.dynamic_resonant_token_count > 0:
            self.controller = ResonantController(d_model, self.dynamic_resonant_token_count)
        if self.static_resonant_token_count > 0:
            self.resonant_tokens = nn.Parameter(torch.randn(1, self.static_resonant_token_count, d_model))
        if self.multihead:
            self.resonator = MultiHeadResonance(num_heads=num_heads,
                                                res_tokens=resonant_token_count,
                                                d_model=d_model)
        # Use batch_first=True so inputs are (batch, seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, context=None):
        # x: (batch, seq)
        emb = self.embedding(x)  # (batch, seq, d_model)
        B = emb.size(0)

        # build resonant tokens if any
        res_tokens = []
        if context is not None:
            # context may be token ids or embeddings
            if context.dtype in (torch.int64, torch.int32):
                ctx_emb = self.embedding(context)
            else:
                ctx_emb = context
            context_vec = ctx_emb.mean(dim=1)
        if self.dynamic_resonant_token_count > 0 and context is not None:
            dyn = self.controller(context_vec)
            res_tokens.append(dyn)
        if self.static_resonant_token_count > 0:
            stat = self.resonant_tokens.expand(B, -1, -1)
            res_tokens.append(stat)
        if res_tokens:
            tokens = torch.cat(res_tokens, dim=1)
        else:
            tokens = torch.empty(B, 0, self.d_model, device=emb.device)

        # retain grad on resonant tokens
        if self.training and tokens.numel() > 0 and tokens.requires_grad:
            tokens.retain_grad()
            self._res_tokens_for_ri = tokens

        # concatenate and encode (batch_first)
        inp = torch.cat([tokens.detach() * (1-self.alpha) + tokens * self.alpha, emb], dim=1)
        # inp shape: (batch, res + seq, d_model)
        encoded = self.encoder(inp)  # returns (batch, res+seq, d_model)
        # strip off resonance prefix
        out = encoded[:, tokens.size(1):, :]
        logits = self.output(out)
        return logits, tokens

# Same utility functions as before
def amplify_grad(x, alpha):
    return x.detach() * (1 - alpha) + x * alpha


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
