import os
import torch
from torch import nn  # needed for the helper
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from resonantTransformer import EnhancedResonantTransformer
from config import recursive_convergence_tolerance, baseline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_context(model: nn.Module, input_seq: torch.Tensor) -> torch.Tensor:
    """
    Prepare the context vector for inference, matching training behavior.
    Handles global_res or self_token fallback, and ensures dimension correctness.
    """
    if getattr(model, "global_res", None) is not None:
        if model.global_res.dim() == 3:
            context = model.global_res.mean(dim=1)  # [batch, d_model]
        elif model.global_res.dim() == 2:
            context = model.global_res  # [batch, d_model]
        else:
            raise ValueError(f"[prepare_context] Unexpected global_res shape: {model.global_res.shape}")

        # Safety check: context must match d_model
        assert context.size(-1) == model.d_model, \
            f"[prepare_context] Context size mismatch: got {context.size(-1)}, expected {model.d_model}"

        # Expand for batch
        context = context.expand(input_seq.size(0), -1).contiguous()
    else:
        # fallback to self_token mean
        context = model.self_token.expand(input_seq.size(0), -1, -1).mean(dim=1)

    return context

def prepare_padding_mask(input_seq: torch.Tensor, pad_id: int, num_extra_tokens: int) -> torch.Tensor:
    """
    Builds padding mask correctly for input_seq, accounting for self-token and all resonant tokens.
    """
    padding_mask = (input_seq == pad_id)

    if num_extra_tokens > 0:
        extra = torch.zeros((padding_mask.size(0), num_extra_tokens), dtype=torch.bool, device=padding_mask.device)
        padding_mask = torch.cat([extra, padding_mask], dim=1)

    return padding_mask


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
    # sequence_length=model_config["sequence_length"],
    # max_tokens=model_config["max_tokens"],
    # learning_rate=model_config["learning_rate"],
    # batch_size=model_config["batch_size"],
    d_model=model_config["d_model"],
    num_heads=model_config["num_heads"],
    num_layers=model_config["num_layers"],
    resonant_token_count=model_config["resonant_token_count"],
    dynamic_resonant_token_count=model_config["dynamic_resonant_token_count"],
    multihead=model_config.get("multihead", False),
    max_recursive_steps=model_config["max_recursive_steps"] if hasattr(model_config, "max_recursive_steps") else 5,
    # flux_penalty_weight=model_config["flux_penalty_weight"]
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
    tokens = tokens[:sequence_length - 1]  # reserve 1 slot for self-token
    input_length = len(tokens)
    if input_length < (sequence_length - 1):
        tokens += [pad_id] * ((sequence_length - 1) - len(tokens))

    generated = tokens[:]
    with torch.no_grad():
        for current in range(100):
            input_seq = torch.tensor(generated[-(sequence_length-1):], dtype=torch.long).unsqueeze(0).to(DEVICE)

            # Compute number of extra tokens
            static_res_tokens = getattr(model, "static_resonant_token_count", 0)
            dynamic_res_tokens = getattr(model, "dynamic_resonant_token_count", 0)
            multihead_enabled = getattr(model, "multihead", False)
            multihead_tokens = static_res_tokens if multihead_enabled else 0  # If multihead=True, use static_resonant_token_count

            num_extra_tokens = 1 + static_res_tokens + dynamic_res_tokens + multihead_tokens  # 1 for self-token

            # Prepend dummy tokens for all extra tokens
            if num_extra_tokens > 0:
                prepend = torch.full(
                    (input_seq.size(0), num_extra_tokens),
                    pad_id,
                    dtype=input_seq.dtype,
                    device=input_seq.device
                )
                input_seq = torch.cat([prepend, input_seq], dim=1)

            # === Correct padding mask building ===
            padding_mask = prepare_padding_mask(input_seq, pad_id, 0)  # Already included prepended tokens


            # === Use new helper for safe context prep ===
            context = prepare_context(model, input_seq)

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
            logits = logits[0, -1]

            # Safe softmax
            logits = logits / temperature
            logits = logits - logits.max()  # subtract max for numerical stability
            probs = torch.softmax(logits, dim=0)

            # Filter out invalid tokens
            valid_vocab_size = model.embedding.num_embeddings
            probs = probs[:valid_vocab_size]

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            p = 0.9
            mask = cum_probs <= p
            mask[0] = True
            filtered_probs = sorted_probs * mask
            filtered_sum = filtered_probs.sum()

            if filtered_sum <= 0 or torch.isnan(filtered_sum):
                filtered_probs = sorted_probs[:1]
                sorted_indices = sorted_indices[:1]
                filtered_sum = filtered_probs.sum()

            filtered_probs = filtered_probs / filtered_sum
            next_token = sorted_indices[torch.multinomial(filtered_probs, 1)].item()

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
