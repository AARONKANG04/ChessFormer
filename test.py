import torch
from model import ChessFormer

DATASET_PATH = "dataset.pt"   # change to your actual file

def print_first_entries(path, n=10):
    print(f"Loading dataset from: {path}")
    data = torch.load(path)

    print(f"Dataset size: {len(data)}\n")

    for i, entry in enumerate(data[:n]):
        print(f"===== Entry {i} =====")
        
        tokens = entry["tokens"]
        move_idx = entry["move_index"]
        promo_idx = entry["promo_index"]
        is_promo = entry["is_promo"]

        print("Tokens shape:", tokens.shape)        # expect (64,117)
        print("Move index:", move_idx)              # 0..4095
        print("Promo index:", promo_idx)            # -1 or 0..3
        print("Is promo:", is_promo)                # True/False
        print()


    model = ChessFormer()
    sample_entry = data[0]
    tokens = sample_entry["tokens"].unsqueeze(0)  # add batch dim
    move_mask = torch.ones((1, 64, 64))            # dummy mask
    move_logits, promo_logits = model(tokens, move_mask)
    print("Model output shapes:")
    print("Move logits shape:", move_logits.shape)
    print("Promo logits shape:", promo_logits.shape)

print_first_entries(DATASET_PATH)
