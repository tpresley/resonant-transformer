import os
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from resonantTransformer import EnhancedResonantTransformer

from config import recursive_convergence_tolerance, baseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List .pt files
pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
if not pt_files:
    raise FileNotFoundError("No .pt files found in the current directory.")

print("Select a model file:")
for i, f in enumerate(pt_files):
    print(f"{i}: {f}")

selected_idx = int(input("Enter the number of the model file to load: ").strip())
model_path = pt_files[selected_idx]

# Load state from file
state = torch.load(model_path, map_location=DEVICE)
model_config = state["config"]
resonant_context = state.get("final_dynamic_resonant_state", None)

# Load tokenizer
tokenizer_dir = "tokenizer-tinystories"
tokenizer = ByteLevelBPETokenizer(
    f"{tokenizer_dir}/vocab.json",
    f"{tokenizer_dir}/merges.txt"
)
tokenizer.add_special_tokens(["<pad>", "<unk>"])
pad_id = tokenizer.token_to_id("<pad>")
tokenizer.post_processor = BertProcessing(("<pad>", pad_id), ("<pad>", pad_id))

vocab_size = tokenizer.get_vocab_size()
sequence_length = model_config["sequence_length"]  # Can be inferred/stored if desired

# Instantiate model using saved config
model = EnhancedResonantTransformer(
    # baseline=model_config["baseline"],
    vocab_size=model_config["vocab_size"],
    d_model=model_config["d_model"],
    num_heads=model_config["num_heads"],
    num_layers=model_config["num_layers"],
    resonant_token_count=model_config["resonant_token_count"],
    dynamic_resonant_token_count=model_config["dynamic_resonant_token_count"],
    multihead=model_config.get("multihead", False)
)
# Resize saved context to match inference-time batch size
if resonant_context is not None:
    if resonant_context.size(0) != 1:
        resonant_context = resonant_context[0:1]  # Just use first row for single-sample inference

model.load_state_dict(state["model_state_dict"])
model.to(DEVICE)
# Restore resonant memory so recursive_forward can actually use it
if resonant_context is not None:
    model.global_res = resonant_context.to(DEVICE)
model.eval()

# Inference loop
print("Inference mode. Type a sentence (empty line to quit):")
while True:
    user_input = input("> ").strip()
    if not user_input:
        break

    encoded = tokenizer.encode(user_input)
    tokens = encoded.ids[:sequence_length]
    if len(tokens) < sequence_length:
        tokens += [pad_id] * (sequence_length - len(tokens))

    generated = tokens[:sequence_length]
    with torch.no_grad():
        for _ in range(100):
            input_seq = torch.tensor(generated[-sequence_length:], dtype=torch.long) \
                                .unsqueeze(0).to(DEVICE)
            baseline = True
            if not baseline:
                logits, _ = model.recursive_forward(input_seq, tol=recursive_convergence_tolerance)
            else:
                logits, _, _, _ = model.forward(input_seq)
            # 1) temperature
            temperature = 0.8
            probs = torch.softmax(logits[0, -1] / temperature, dim=0)
            # 1) sort by descending probability
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            # 2) compute cumulative sum to do top‑p
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            p = 0.9
            mask = cum_probs <= p
            # ensure at least the highest‑prob token remains
            mask[0] = True
            filtered = sorted_probs * mask
            filtered_sum = filtered.sum()
            # fallback if something went wrong (zero sum)
            if filtered_sum <= 0 or torch.isnan(filtered_sum):
                filtered = sorted_probs[:1]
                sorted_indices = sorted_indices[:1]
                filtered_sum = filtered.sum()
            # normalize to get a valid distribution
            filtered = filtered / filtered_sum
            next_token = sorted_indices[torch.multinomial(filtered, 1)].item()
            generated.append(next_token)


    output_text = tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)
    output_text = output_text.replace("Ġ", " ").replace("@@", "").replace("â", "'").strip()
    print("\nGenerated continuation:\n", output_text.strip())

    # --- Diagnostics ---
    if not baseline:
        if hasattr(model, 'resolution_score'):
            print(f"[resolution score]: {model.resolution_score:.4f}")
        if hasattr(model, 'attention_trajectory') and model.attention_trajectory:
            last_attn = model.attention_trajectory[-1]
            if isinstance(last_attn, torch.Tensor) and last_attn.dim() == 4:
                self_attn_mean = last_attn[:, :, 0, 0].mean().item()
                print(f"[self-attn mean]: {self_attn_mean:.4f}")
            else:
                # (fall‑through or alternative for non‑tensor last_attn)
                # last_attn is just a scalar mean‐attention; handle or skip accordingly
                self_attn_mean = float(last_attn)
        if hasattr(model, 'surprisal_trajectory'):
            print(f"[steps to converge]: {len(model.surprisal_trajectory)}")
