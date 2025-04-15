# inference.py
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from resonantTransformer import EnhancedResonantTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import d_model, num_heads, num_layers, resonant_token_count, sequence_length

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

# Load model
model = EnhancedResonantTransformer(
    vocab_size, d_model, num_heads, num_layers, resonant_token_count
)
model.load_state_dict(torch.load("enhanced_resonant_model.pt", map_location=DEVICE))
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
            logits, _ = model(input_seq, context=input_seq)
            probs = torch.softmax(logits[0, -1] / 0.8, dim=0)
            topk_probs, topk_indices = torch.topk(probs, 40)
            next_token = topk_indices[torch.multinomial(topk_probs, 1)].item()
            generated.append(next_token)

    output_text = tokenizer.decode(generated[len(tokens):], skip_special_tokens=True)
    print("Generated continuation:", output_text.strip())
