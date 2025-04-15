# resonantTransformer.py
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
        out = self.linear(context_embedding).view(-1, self.res_tokens, self.d_model)
        out = F.layer_norm(out, (self.d_model,))
        return out

class EnhancedResonantTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, res_tokens, 
                 dynamic=True, multihead=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dynamic = dynamic
        self.multihead = multihead
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

        if context is not None:
            context_vec = context.mean(dim=1) if context.dtype in (torch.float32, torch.float64) \
                          else self.embedding(context.long()).mean(dim=1)

        if self.dynamic and context is not None:
            res_tokens = self.controller(context_vec)
        elif self.multihead and context is not None:
            res_tokens = self.resonator(context_vec)
        else:
            res_tokens = self.resonant_tokens.repeat(B, 1, 1) if self.res_tokens > 0 else torch.empty(B, 0, self.d_model, device=x.device)

        if self.training and self.res_tokens > 0 and res_tokens.numel() > 0 and res_tokens.requires_grad:
            res_tokens.retain_grad()
            self._res_tokens_for_ri = res_tokens

        if self.res_tokens > 0:
            x = torch.cat([amplify_grad(res_tokens, self.alpha), x], dim=1)

        x = x.transpose(0, 1)
        encoded = self.encoder(x)
        out = encoded[self.res_tokens:].transpose(0, 1)
        return self.output(out), res_tokens

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
