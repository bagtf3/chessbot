def data_worker(worker_id, queue, stop_event):
    SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
    import chess, chess.engine, random
    import chessbot.utils as cbu, chessbot.encoding as cbe, chessbot.features as cbf
    import numpy as np

    board = cbu.random_init(random.choice([0, 1, 2, 3]))
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
        while not stop_event.is_set():
            if board.is_game_over() or not list(board.legal_moves):
                board = cbu.random_init(random.choice([0, 1, 2, 3]))

            to_score = board if board.turn else cbu.mirror_board(board)
            info_list = engine.analyse(
                to_score, multipv=50,
                limit=chess.engine.Limit(depth=random.choice([5, 7, 9])),
                info=chess.engine.Info.ALL
            )
            info_list = [i for i in info_list if 'pv' in i and i['pv']]
            if not info_list:
                board = cbu.random_init(random.choice([0, 1, 2, 3]))
                continue

            X = cbe.encode_board(to_score)
            y = cbe.build_training_targets_8x8x73(to_score, info_list)
            y.update(cbf.all_king_exposure_features(to_score))
            y.update(cbf.all_piece_features(to_score))
            y['material'] = cbf.get_piece_value_sum(to_score)
            y['piece_to_move'] = cbe.piece_to_move_target(to_score, y['policy_logits'])
            y['legal_moves'] = 1 * cbe.legal_mask_8x8x73(to_score).astype(np.float32)

            queue.put((X, y))

            move_to_make = random.choice(info_list[:5])['pv'][0]
            if not board.turn:
                move_to_make = cbu.mirror_move(move_to_make)
            board.push(move_to_make)
