import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import ChessFormer   # your model file
import math


# ============================================================
# Dataset wrapper
# ============================================================

class ChessPolicyDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading dataset: {data_path}")
        self.samples = torch.load(data_path)
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        tokens = s["tokens"].float()              # (64,117)
        move_idx = s["move_index"]                # int, 0..4095
        promo_idx = s["promo_index"]              # int in {0,1,2,3} or -1
        is_promo = s["is_promo"]                  # bool

        return (
            tokens,
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(promo_idx, dtype=torch.long),
            torch.tensor(is_promo, dtype=torch.bool)
        )


# ============================================================
# Loss function
# ============================================================

class PolicyLoss(nn.Module):
    """
    Combined move + promotion loss.
    """
    def __init__(self):
        super().__init__()
        self.move_loss = nn.CrossEntropyLoss()
        self.promo_loss = nn.CrossEntropyLoss()

    def forward(self, move_logits, promo_logits, move_idx, promo_idx, is_promo):
        """
        move_logits: (B,4096)
        promo_logits: (B,64,4)
        move_idx: (B,)
        promo_idx: (B,)  -1 if not promotion
        is_promo: (B,) bool
        """

        # ----- Move head loss -----
        L_move = self.move_loss(move_logits, move_idx)

        # ----- Promotion head loss -----
        # Extract the FROM-square of the chosen move
        from_sq = move_idx // 64      # (B,)
        B = promo_logits.shape[0]

        # Gather predicted promotion logits at the (from_sq)
        # promo_logits[b, sq, :] → shape (B,4)
        promo_logits_chosen = promo_logits[
            torch.arange(B), from_sq
        ]  # (B,4)

        # Only compute promo loss for samples where is_promo=True
        if is_promo.any():
            L_promo = self.promo_loss(
                promo_logits_chosen[is_promo],
                promo_idx[is_promo]
            )
        else:
            L_promo = torch.tensor(0.0, device=move_logits.device)

        return L_move + L_promo, L_move, L_promo


# ============================================================
# Training loop
# ============================================================

def train(
        dataset_path="dataset.pt",
        batch_size=64,
        lr=1e-4,
        epochs=5,
        save_path="chessformer_policy.pt"
    ):

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load dataset
    dataset = ChessPolicyDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model
    model = ChessFormer().to(device)
    optimz = optim.Adam(model.parameters(), lr=lr)
    criterion = PolicyLoss()

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        model.train()

        pbar = tqdm(loader)
        total_loss = 0
        total_move_loss = 0
        total_promo_loss = 0

        for tokens, move_idx, promo_idx, is_promo in pbar:
            tokens = tokens.to(device)          # (B,64,117)
            move_idx = move_idx.to(device)      # (B,)
            promo_idx = promo_idx.to(device)    # (B,)
            is_promo = is_promo.to(device)      # (B,)

            # ----- Forward -----
            # Move mask: no masking needed for policy-only model (moves already valid)
            # So produce a full 4096 logits
            move_logits, promo_logits = model(tokens)

            # ----- Loss -----
            loss, L_move, L_promo = criterion(
                move_logits, promo_logits,
                move_idx, promo_idx, is_promo
            )

            # ----- Backprop -----
            optimz.zero_grad()
            loss.backward()
            optimz.step()

            total_loss += loss.item()
            total_move_loss += L_move.item()
            total_promo_loss += L_promo.item()

            pbar.set_description(
                f"Loss: {total_loss:.2f} "
                f"Move: {total_move_loss:.2f} "
                f"Promo: {total_promo_loss:.2f}"
            )

        # Save checkpoint each epoch
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")


# ============================================================
# Launch training
# ============================================================

if __name__ == "__main__":
    train(
        dataset_path="dataset.pt",
        batch_size=32,
        lr=1e-4,
        epochs=30,
        save_path="chessformer_policy.pt"
    )

# ============================================================
# ChessFormer Policy Training
# ============================================================




