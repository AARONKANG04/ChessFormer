import torch
import chess

from model import ChessFormer, ChessFormerInputTokenizer, flip_board

def move_to_index(move: chess.Move):
    return move.from_square * 64 + move.to_square

def index_to_move(idx: int):
    return idx // 64, idx % 64

PROMO_MAP = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

def sample_from_model(board, model, tokenizer, temperature=1.0, top_k=None):
    """
    Given a python-chess Board, return a chess.Move chosen by your model.
    Always returns a **legal move**.
    """

    # Build 8-history window
    history = [board.copy()]
    node = board.copy()
    while node.move_stack and len(history) < 8:
        node.pop()
        history.append(node.copy())

    history = list(reversed(history))
    while len(history) < 8:
        history.append(history[-1])

    # Flip so side-to-move = white
    flipped_history = [flip_board(b) for b in history]

    # Tokenize → (1,64,117)
    tokens = tokenizer.encode(flipped_history).unsqueeze(0)

    # Inference
    with torch.no_grad():
        move_logits, promo_logits = model(tokens)
        move_logits = move_logits[0]  # shape (4096,)
        promo_logits = promo_logits[0]  # shape (64,4)

    # Mask illegal moves
    legal = list(board.legal_moves)
    legal_ids = []
    for m in legal:
        legal_ids.append(move_to_index(m))

    logits = move_logits[legal_ids]

    # Top-k
    if top_k is not None and len(logits) > top_k:
        values, idxs = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        for i, v in zip(idxs, values):
            mask[i] = v
        logits = mask

    # Softmax
    probs = torch.softmax(logits / temperature, dim=0)

    # Sample move
    choice = torch.multinomial(probs, 1).item()
    chosen_id = legal_ids[choice]

    # Rebuild move
    fs, ts = index_to_move(chosen_id)
    move = chess.Move(fs, ts)

    # Promotion handling
    piece = board.piece_at(fs)
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(ts)
        if rank == 7:  # promotion rank for white after flip
            promo_probs = torch.softmax(promo_logits[fs] / temperature, dim=0)
            promo_idx = torch.multinomial(promo_probs, 1).item()
            move.promotion = PROMO_MAP[promo_idx]

    return move



def play_game(model, tokenizer, opponent="human", temperature=1.0):
    """
    opponent = "human" or "stockfish"
    """

    board = chess.Board()

    if opponent == "stockfish":
        sf = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")

    while not board.is_game_over():
        print(board)
        print()

        if board.turn == chess.WHITE:
            # MODEL PLAYS WHITE
            move = sample_from_model(board, model, tokenizer, temperature=temperature)
            print("Model plays:", move.uci())
            board.push(move)

        else:
            # OPPONENT MOVE
            if opponent == "human":
                print("Your move (e.g., e2e4): ", end="")
                mv = input().strip()
                try:
                    move = chess.Move.from_uci(mv)
                    if move not in board.legal_moves:
                        print("Illegal move.")
                        continue
                    board.push(move)
                except:
                    print("Invalid format.")
                    continue

            else:
                info = sf.analyse(board, chess.engine.Limit(depth=12))
                move = info["pv"][0]
                print("Stockfish plays:", move.uci())
                board.push(move)

        print()

    print("Game over:", board.result())

    if opponent == "stockfish":
        sf.quit()



model = ChessFormer()
model.load_state_dict(torch.load("chessformer_policy.pt", map_location="cpu"))
model.eval()

tokenizer = ChessFormerInputTokenizer()

play_game(model, tokenizer, opponent="human", temperature=0.8)
