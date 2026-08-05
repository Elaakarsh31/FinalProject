"""
Generate a text sample from the trained checkpoint saved by run_iteration4.py.

Run from the project root:
    python generate_sample.py
"""

import torch
from src.dataloader import load_data
from src.model import GPTLanguageModel

DATA_PATH = "data/input.txt"
CHECKPOINT_PATH = "results/gpt_final.pt"

# Must match the config run_iteration4.py actually trained with.
# If you changed BLOCK_SIZE / N_EMBED there, update these to match --
# the saved weights are shape-locked to whatever config trained them.
BLOCK_SIZE = 256
N_EMBED = 384
N_HEAD = 4      # best config from the grid search (n_layer=4, n_head=4, dropout=0.1)
N_LAYER = 4
DROPOUT = 0.1

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    data = load_data(DATA_PATH, device)
    vocab_size = data["vocab_size"]

    model = GPTLanguageModel(
        vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embed=N_EMBED,
        n_head=N_HEAD, n_layer=N_LAYER, dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully ;)")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=1000)[0].tolist()
    print(data["decode"](generated))


if __name__ == "__main__":
    main()