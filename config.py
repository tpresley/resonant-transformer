# config.py

#####
# Standard Hyperparameters
#####

baseline=False

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
# Fraction of data to reserve for validation
validation_split = 0.1
# Number of epochs with no val-PPL improvement before early stopping
early_stopping_patience = 3

#####
# Resonant Token Parameters
#####

# Number of Static Resonant Tokens (zero to disable)
resonant_token_count = 6
# Number of Dynamic Resonant Tokens to use (zero to disable)
dynamic_resonant_token_count = 4
# How many epochs in the beginning to prevent Resonant Tokens
# from influencing model weights (helps prevent collapse from early misalignment)
warmup_epochs = 1
# Multiple over the standard model learning to amplify learning in reonant tokens
token_learning_amplifier = 5
# Scalar controlling how strongly resonant tokens are optimized to align with
# their gradients (acts like a targeted learning rate for participation)
lambda_ri = 0.2
# Weight for penalizing angular misalignment between resonant tokens and their gradients
# (discourages semantic drift)
# lower value gives tokens more "creativity", but risks model collapse
lambda_rs = 0.05
# Weight for penalizing redundancy among resonant tokens (encourages diversity in their representations)
lambda_div = 0.1

lambda_sur = 5e-5      # surprisal-drop weight
lambda_attn = 0.05     # attention-divergence weight
lambda_res = 0.5      # resolution-coherence weight

multihead_resonance = True  # boolean flag for multihead resonance
max_recursive_steps = 5
recursive_convergence_tolerance = 1e-5
flux_penalty_weight = 0.01

#####
# Contrastive Loss Parameters
#####
contrastive_margin   = 50.0    # Blow up the margin so dist rarely exceeds it
lambda_contrastive   = 5.0     # Heavily weight contrastive loss in the total

lambda_dyn_var = 0.05
lambda_head_entropy = 0.01
lambda_inner_align = 1.0