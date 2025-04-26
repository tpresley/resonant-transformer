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
resonant_context = state.get("final_resonant_state", None)

# Load tokenizer
tokenizer_dir = "tokenizer-tinystories"
tokenizer = ByteLevelBPETokenizer(
    f"{tokenizer_dir}/vocab.json",
    f"{tokenizer_dir}/merges.txt"
)
tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
pad_id = tokenizer.token_to_id("<pad>")
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")
tokenizer.post_processor = BertProcessing(("<pad>", pad_id), ("<pad>", pad_id))

vocab_size = tokenizer.get_vocab_size()
sequence_length = model_config["sequence_length"]  # Can be inferred/stored if desired

# Instantiate model using saved config
model = EnhancedResonantTransformer(
    baseline=model_config["baseline"] if hasattr(model_config, "baseline") else False,
    vocab_size=model_config["vocab_size"],
    d_model=model_config["d_model"],
    num_heads=model_config["num_heads"],
    num_layers=model_config["num_layers"],
    resonant_token_count=model_config["resonant_token_count"],
    dynamic_resonant_token_count=model_config["dynamic_resonant_token_count"],
    multihead=model_config.get("multihead", False),
    max_recursive_steps=model_config["max_recursive_steps"] if hasattr(model_config, "max_recursive_steps") else 5
)
# Resize saved context to match inference-time batch size
if resonant_context is not None:
    if resonant_context.dim() == 3:
        # resonant_context shape is [batch, num_tokens, d_model]
        resonant_context = resonant_context[0:1]  # Select first batch if needed
        model.global_res = resonant_context.to(DEVICE)
    elif resonant_context.dim() == 2:
        # Already [batch, d_model], fine
        model.global_res = resonant_context.unsqueeze(1).to(DEVICE)
    else:
        raise ValueError(f"[load] Unexpected resonant_context shape: {resonant_context.shape}")
else:
    model.global_res = None

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

    # === Encode with <bos> and <eos> ===
    encoded = tokenizer.encode(user_input)
    tokens = [bos_id] + encoded.ids
    tokens = tokens[:sequence_length]
    input_length = len(tokens)
    if input_length < sequence_length:
        tokens += [pad_id] * (sequence_length - len(tokens))

    generated = tokens[:]
    with torch.no_grad():
        for current in range(100):
            input_seq = torch.tensor(generated[-sequence_length:], dtype=torch.long).unsqueeze(0).to(DEVICE)
            padding_mask = (input_seq == pad_id)

            # === Match training context derivation ===
            if model.global_res is not None:
                context = model.global_res.mean(dim=1).expand(input_seq.size(0), -1).contiguous()
            else:
                context = model.self_token.expand(input_seq.size(0), -1, -1).mean(dim=1)

            if not baseline:
                logits, _ = model.recursive_forward(
                    input_seq,
                    context=context,
                    tol=recursive_convergence_tolerance,
                    padding_mask=padding_mask
                )
            else:
                logits, _, _, _ = model.forward(input_seq, context=context, padding_mask=padding_mask)
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
            if next_token == eos_id:
                print(f"Found EOS at token {current}")
                break


    if eos_id in generated:
        eos_index = generated.index(eos_id)
        print(f"Found EOS in generated string at {eos_id}")
        generated = generated[:eos_index + 1]

    output_text = tokenizer.decode(generated[input_length:], skip_special_tokens=True)
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
