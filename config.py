# config.py

def get_config():
    return {k: v for k, v in globals().items() if not k.startswith("__") and not callable(v)}

#####
# Standard Hyperparameters
#####

baseline=False   # disable all non-vanilla transformer features
wandb_project_name = "resonant-transformer-refactored"

d_model = 256
num_heads = 4
num_layers = 6
vocab_size = 1024
sequence_length = 128
max_tokens = 20_000_000
learning_rate = 1e-4
weight_decay = 0.01
batch_size = 256
num_epochs = 50


# How many epochs to linearly warm up LR (e.g. 5% of total)
lr_warmup_epochs = max(1, int(0.05 * num_epochs))
label_smoothing = 0.1
embedding_dropout = 0.1
validation_split = 0.1
# Number of epochs with no val-PPL improvement before early stopping
early_stopping_patience = 3


#####
# Resonant Token Parameters
#####

multihead_resonance = True  # boolean flag for multihead resonance
warmup_epochs = 3  # controls ramp of influence of all resonant features
max_recursive_steps = 5

resonant_token_count = 6   # Number of Static Resonant Tokens (zero to disable)
dynamic_resonant_token_count = 4   # Number of Dynamic Resonant Tokens to use (zero to disable)
token_learning_amplifier = 1.0   # Multiple over the standard model learning to amplify learning in reonant tokens

lambda_ri = 0.05            # strength of resonant token alignment to gradient (RI: relevance index)
lambda_rs = 0.01            # penalty for angular misalignment of tokens and their gradients (RS: semantic drift)
lambda_div = 0.2            # penalty for low pairwise diversity among resonant tokens (reduces redundancy)

lambda_sur = 5e-5           # weight for surprisal drop reward (discourages degenerate sharp token predictions)
lambda_attn = 0.05          # weight for attention KL-divergence penalty (encourages temporal attention coherence)
lambda_res = 0.2            # weight for resolution score reward (stability of final token predictions)

lambda_dyn_var = 0.05       # penalty for low variance in dynamic resonant tokens (encourages diverse generation)
lambda_head_entropy = 0.01  # penalty for low entropy in multi-head selector (promotes distributed head usage)
lambda_inner_align = 0.5    # weight for matching outer and inner resonant token states (encourages consistency)
lambda_entropy = 1e-4       # global entropy regularization (penalizes overconfident softmax distributions)
lambda_token_sparsity = 0.01  # encourages sparse participation of tokens (prevents overly diffuse token influence)

recursive_convergence_tolerance = 1e-5     # minimum delta threshold across steps for early halting in recursive refinement
flux_penalty_weight = 0.01                 # weight for penalty when entropy, attention, or resolution deviate too much from EMA baselines

contrastive_margin   = 10.0                # margin for contrastive loss (softplus(margin - dist)); large margin encourages separation
lambda_contrastive   = 0.2                 # weight for contrastive loss between original and corrupted predictions
