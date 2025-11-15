# import chess
# import chess.pgn
# import chess.engine
# import torch
# from pathlib import Path

# # ---------------------------------------------------
# # CONFIG
# # ---------------------------------------------------
# STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
# SEARCH_DEPTH = 12
# OUTPUT_PATH = "dataset.pt"

# # Import your tokenizer and board flip
# from model import ChessFormerInputTokenizer, flip_board   # adjust import path as needed

# tokenizer = ChessFormerInputTokenizer(history_size=8)


# # ---------------------------------------------------
# # HELPER FUNCTIONS
# # ---------------------------------------------------

# def move_to_index(move: chess.Move):
#     return move.from_square * 64 + move.to_square


# def promo_to_index(move: chess.Move):
#     if move.promotion is None:
#         return -1
#     return move.promotion - 2   # Knight=2→0, Bishop=3→1, Rook=4→2, Queen=5→3


# # ---------------------------------------------------
# # MAIN EXTRACTION LOGIC
# # ---------------------------------------------------

# def extract_positions_from_pgn(pgn_path, engine, max_positions=None, print_every=1000):
#     results = []
#     last_print = 0

#     with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
#         while True:
#             game = chess.pgn.read_game(f)
#             if game is None:
#                 break

#             board = game.board()
#             ply = 0
#             history_boards = []

#             for move in game.mainline_moves():

#                 # Stop early if needed
#                 if max_positions is not None and len(results) >= max_positions:
#                     return results

#                 board.push(move)
#                 ply += 1

#                 # Maintain 8-board history
#                 history_boards.append(board.copy())
#                 if len(history_boards) > 8:
#                     history_boards = history_boards[-8:]

#                 if ply < 16:
#                     continue

#                 flipped_history = [flip_board(b) for b in history_boards]
#                 tokens = tokenizer.encode(flipped_history)

#                 # Stockfish best move
#                 analysis_board = flipped_history[0]
#                 info = engine.analyse(analysis_board, limit=chess.engine.Limit(depth=SEARCH_DEPTH))
#                 best_move = info["pv"][0]

#                 # Store training sample
#                 results.append({
#                     "tokens": tokens,
#                     "move_index": move_to_index(best_move),
#                     "promo_index": promo_to_index(best_move),
#                     "is_promo": best_move.promotion is not None
#                 })

#                 # Manual progress print
#                 if len(results) - last_print >= print_every:
#                     last_print = len(results)
#                     print(f"Extracted positions: {len(results)}")

#     return results


# # ---------------------------------------------------
# # DATASET BUILDER
# # ---------------------------------------------------

# def build_dataset(pgn_files, output_path=OUTPUT_PATH, max_positions=None):
#     engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

#     samples = []
#     remaining = max_positions

#     print(f"\nBuilding dataset from {len(pgn_files)} PGN files...")

#     for pgn in pgn_files:

#         if remaining is not None and remaining <= 0:
#             break

#         extracted = extract_positions_from_pgn(
#             pgn,
#             engine,
#             max_positions=remaining,
#             print_every=1000
#         )

#         samples.extend(extracted)

#         if remaining is not None:
#             remaining -= len(extracted)
#             print(f"Remaining limit: {remaining}")

#         if remaining is not None and remaining <= 0:
#             break

#     engine.quit()

#     print(f"\nSaving dataset to {output_path} (total samples: {len(samples)})")
#     torch.save(samples, output_path)
#     print("Done.")


# # ---------------------------------------------------
# # CLI ENTRY POINT
# # ---------------------------------------------------

# if __name__ == "__main__":
#     PGN_FILES = ["/Users/aaronkang/Downloads/games.pgn"]
#     build_dataset(PGN_FILES, max_positions=100000)


import chess
import chess.pgn
import chess.engine
import torch
from multiprocessing import Pool, cpu_count
import os
import io

from model import ChessFormerInputTokenizer, flip_board

# ==============================
# CONFIG
# ==============================
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
SEARCH_DEPTH = 12
OUTPUT_PATH = "dataset.pt"

tokenizer = ChessFormerInputTokenizer(history_size=8)

# ==============================
# HELPERS
# ==============================

def move_to_index(move):
    return move.from_square * 64 + move.to_square

def promo_to_index(move):
    if move.promotion is None:
        return -1
    return move.promotion - 2    # Knight=0, Bishop=1, Rook=2, Queen=3

# ==============================
# WORKER FUNCTION
# ==============================

def process_game_chunk(args):
    """Worker: processes a list of PGN text blocks."""
    chunk_id, game_texts, max_positions = args

    # Each worker must create its own engine instance
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    results = []
    extracted_count = 0

    for game_txt in game_texts:

        game = chess.pgn.read_game(iter(io.StringIO(game_txt)))
        if game is None:
            continue

        board = game.board()
        ply = 0
        history_boards = []

        for move in game.mainline_moves():
            if max_positions is not None and extracted_count >= max_positions:
                engine.quit()
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

            extracted_count += 1

    engine.quit()
    return results


# ==============================
# SPLIT PGN INTO GAME BLOCKS
# ==============================

def split_pgn_into_games(pgn_path):
    """Reads a PGN and returns a list: [game1_text, game2_text, ...]"""
    games = []
    current = []

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event"):
                if current:
                    games.append("".join(current))
                current = [line]
            else:
                current.append(line)

        # Last game
        if current:
            games.append("".join(current))

    return games


# ==============================
# MAIN MULTIPROCESSING BUILDER
# ==============================

def build_dataset_multiprocess(pgn_path, output_path=OUTPUT_PATH, max_positions=250000):
    print("Loading PGN…")
    game_texts = split_pgn_into_games(pgn_path)
    num_games = len(game_texts)
    print(f"Found {num_games} games")

    # Split into worker chunks
    n_workers = min(cpu_count(), 8)  # don't overspawn
    chunk_size = 250

    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(num_games, (i + 1) * chunk_size)
        if start < end:
            chunks.append((i, game_texts[start:end], max_positions))
            print(f"Chunk {i}: games {start} to {end}")

    print(f"Spawning {len(chunks)} workers…")

    # Launch multiprocessing
    pool = Pool(processes=len(chunks))
    results = []

    processed = 0

    for worker_result in pool.imap_unordered(process_game_chunk, chunks):
        results.extend(worker_result)
        processed += len(worker_result)
        print(f"Progress: {processed}/{max_positions}")

        if processed >= max_positions:
            break

    pool.terminate()
    pool.join()

    # Truncate to requested size
    results = results[:max_positions]

    print(f"\nSaving dataset ({len(results)} samples)…")
    torch.save(results, output_path)
    print("Done.")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    build_dataset_multiprocess(
        pgn_path="/Users/aaronkang/Downloads/lichess_elite_2025-07.pgn",
        max_positions=100000,
        output_path="dataset1.pt"
    )
