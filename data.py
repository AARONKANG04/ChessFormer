import chess
import chess.pgn
import chess.engine
import torch
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
SEARCH_DEPTH = 12
OUTPUT_PATH = "dataset.pt"

# Import your tokenizer and board flip
from model import ChessFormerInputTokenizer, flip_board   # adjust import path as needed

tokenizer = ChessFormerInputTokenizer(history_size=8)


# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def move_to_index(move: chess.Move):
    """Return flat index 0..4095 from from-square and to-square."""
    return move.from_square * 64 + move.to_square


def promo_to_index(move: chess.Move):
    """
    KNIGHT=2 → 0  
    BISHOP=3 → 1  
    ROOK=4   → 2  
    QUEEN=5  → 3  
    Return -1 if not a promotion.
    """
    if move.promotion is None:
        return -1
    return move.promotion - 2

# ---------------------------------------------------
# MAIN EXTRACTION LOGIC
# ---------------------------------------------------

def extract_positions_from_pgn(pgn_path, engine, max_positions=None):
    results = []

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            ply = 0

            history_boards = []

            moves = list(game.mainline_moves())
            for move in tqdm(moves, desc=f"Processing {pgn_path}", leave=False):
                if max_positions is not None and len(results) >= max_positions:
                    return results

                board.push(move)
                ply += 1

                history_boards.append(board.copy())
                if len(history_boards) > 8:
                    history_boards = history_boards[-8:]

                if ply < 16:
                    continue

                flipped_history = [flip_board(b) for b in history_boards]
                tokens = tokenizer.encode(flipped_history)

                analysis_board = flipped_history[0]
                info = engine.analyse(analysis_board, limit=chess.engine.Limit(depth=SEARCH_DEPTH))
                best_move = info["pv"][0]

                results.append({
                    "tokens": tokens,
                    "move_index": move_to_index(best_move),
                    "promo_index": promo_to_index(best_move),
                    "is_promo": best_move.promotion is not None
                })

    return results



# ---------------------------------------------------
# DATASET BUILDER
# ---------------------------------------------------

def build_dataset(pgn_files, output_path=OUTPUT_PATH, max_positions=None):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    samples = []
    remaining = max_positions  # can be None → unlimited

    print(f"\nBuilding dataset from {len(pgn_files)} PGN files...")
    
    # Progress over PGNs
    for pgn in tqdm(pgn_files, desc="PGN Files", unit="file"):
        if remaining is not None and remaining <= 0:
            break

        extracted = extract_positions_from_pgn(
            pgn,
            engine,
            max_positions=remaining
        )

        samples.extend(extracted)

        if remaining is not None:
            remaining -= len(extracted)

    engine.quit()

    print(f"\nSaving dataset to {output_path} (total samples: {len(samples)})")
    torch.save(samples, output_path)
    print("Done.")



# ---------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------

if __name__ == "__main__":

    PGN_FILES = ["/Users/aaronkang/Downloads/games.pgn"]

    build_dataset(PGN_FILES, max_positions=1000)
