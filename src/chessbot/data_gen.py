import os, time, random, pickle, uuid
import chess, chess.engine
import multiprocessing as mp
import chessbot.encoding as cbe
import chessbot.features as cbf
import chessbot.utils as cbu

SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
OUT_DIR = "C:/Users/Bryan/Data/chessbot_data/training_data/policy_value_data"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 2048   # positions per pickle file


def generate_batch(worker_id, batch_idx):
    """One worker generates a single batch and writes to disk."""
    all_X, all_Y = [], []
    board = cbu.random_init(random.choice([0, 1, 2, 4]))

    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
        while len(all_X) < BATCH_SIZE:
            if board.is_game_over() or not list(board.legal_moves):
                board = cbu.random_init(random.choice([0, 1, 2, 4]))

            to_score = board if board.turn else cbu.mirror_board(board)
            info_list = engine.analyse(
                to_score,
                multipv=50,
                limit=chess.engine.Limit(depth=random.choice([5, 7])),
                info=chess.engine.Info.ALL
            )
            
            info_list = [i for i in info_list if 'pv' in i and i['pv']]
            if not info_list:
                board = cbu.random_init(random.choice([0, 1, 2, 4]))
                continue

            all_X.append(cbe.encode_board(to_score))
            y = cbe.build_training_targets_8x8x73(to_score, info_list)
            y.update(cbf.all_king_exposure_features(to_score))
            y.update(cbf.all_piece_features(to_score))
            y['material'] = cbf.get_piece_value_sum(to_score)
            y['piece_to_move'] = cbe.piece_to_move_target(to_score, y['policy_logits'])
            y['legal_moves'] = 1 * cbe.legal_mask_8x8x73(to_score).astype("float32")
            all_Y.append(y)
            move_to_make = random.choice(info_list[:5])['pv'][0]
            if not board.turn:
                move_to_make = cbu.mirror_move(move_to_make)
            board.push(move_to_make)

    # unique filename: worker id + index + random uuid
    out_path = os.path.join(OUT_DIR, f"w{worker_id}_b{batch_idx}_{uuid.uuid4().hex}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((all_X, all_Y), f)
    print(f"[Worker {worker_id}] Saved {out_path}")


def data_worker(worker_id):
    """Infinite loop: each worker generates batches forever."""
    batch_idx = 0
    while (batch_idx * 2048) < 500000:
        try:
            generate_batch(worker_id, batch_idx)
            batch_idx += 1
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            time.sleep(2)


def main(n_workers=4):
    processes = []
    for wid in range(n_workers):
        p = mp.Process(target=data_worker, args=(wid,))
        p.start()
        processes.append(p)

    # keep main process alive
    for p in processes:
        p.join()


if __name__ == "__main__":
    main(n_workers=4)
