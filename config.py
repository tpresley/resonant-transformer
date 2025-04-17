# config.py

#####
# Standard Hyperparameters
#####

d_model = 256
num_heads = 2
num_layers = 4
sequence_length = 384
max_tokens = 2000000
learning_rate = 1e-4
batch_size = 64
num_epochs = 150


#####
# Resonant Token Parameters
#####

# Number of Static Resonant Tokens (zero to disable)
resonant_token_count = 16
# Number of Dynamic Resonant Tokens to use (zero to disable)
dynamic_resonant_token_count = 16
# How many epochs in the beginning to prevent Resonant Tokens
# from influencing model weights (helps prevent collapse from early misalignment)
warmup_epochs = 20
# Scalar controlling how strongly resonant tokens are optimized to align with
# their gradients (acts like a targeted learning rate for participation)
lambda_ri = 0.01
# Weight for penalizing angular misalignment between resonant tokens and their gradients
# (discourages semantic drift)
# lower value gives tokens more "creativity", but risks model collapse
lambda_rs = 0.02
# Weight for penalizing redundancy among resonant tokens (encourages diversity in their representations)
lambda_div = 0.05
