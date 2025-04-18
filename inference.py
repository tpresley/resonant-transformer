import os
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from resonantTransformer import EnhancedResonantTransformer

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
            input_seq = torch.tensor(generated[-sequence_length:], dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model.recursive_forward(input_seq)
            probs = torch.softmax(logits[0, -1] / 1, dim=0)
            topk_probs, topk_indices = torch.topk(probs, 40)
            next_token = topk_indices[torch.multinomial(topk_probs, 1)].item()
            generated.append(next_token)

    output_text = tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)
    output_text = output_text.replace("Ġ", " ").replace("@@", "").replace("â", "'").strip()
    print("\nGenerated continuation:\n", output_text.strip())

    # --- Diagnostics ---
    if hasattr(model, 'resolution_score'):
        print(f"[resolution score]: {model.resolution_score:.4f}")
    if hasattr(model, 'attention_trajectory') and model.attention_trajectory:
        last_attn = model.attention_trajectory[-1]
        if last_attn.dim() == 4:
            self_attn_mean = last_attn[:, :, 0, 0].mean().item()
            print(f"[self-attn mean]: {self_attn_mean:.4f}")
    if hasattr(model, 'surprisal_trajectory'):
        print(f"[steps to converge]: {len(model.surprisal_trajectory)}")
