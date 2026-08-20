"""
lc0 replay probe primitives: config, candidate filter, spec construction.

Deliberately shares nothing with blunder_replay.py. xc0_BRP is a testing and
validation-tracking feature with its own pool, ratchet and history; lc0_BRP is
an oracle/teacher training feature. A position reaching lc0_BRP after failing
an xc0_BRP is not a promotion and must not be tracked as one.
"""
import os

from pyfastchess import Board as fastboard

from chessbot.game_utils import STARTPOS_FEN, GameSpec, short_fen

LC0 = "[lc0_brp]"

LC0_BRP_SCENARIO = "lc0_replay_probe"
LC0_BRP_CONFIG_ENV = "LC0_REPLAY_CONFIG"

# plies the probe plays from the blunder position. check_for_terminal tests
# plies > max_game_length, so the config value is one less than this.
LC0_BRP_PLIES = 3


def castling_rights_clear(fen):
    """lc0 and xc0 disagree on castling move indices (king-dest vs king-rook),
    so positions with rights are refused. Rights never regenerate, making a
    root check sufficient for the whole probe window."""
    return fen.split()[2] == "-"


def create_lc0_replay_config(cfg, yaml_file=None):
    """
    Config for the lc0 replay probe worker.

    Unlike the xc0 probe, encoding and backend are pinned to lc0 rather than
    inherited: this worker exists to run the teacher net, not the model under
    test.
    """
    pcfg = cfg.copy()
    p_yaml = yaml_file or os.environ.get(LC0_BRP_CONFIG_ENV, "")
    if not p_yaml or not os.path.exists(p_yaml):
        raise RuntimeError(f"{LC0} no lc0 probe config at {p_yaml!r}")

    pcfg.update_via(p_yaml)
    pcfg.init_paths()

    # set after init_paths, which derives model_path back from run_tag and
    # would otherwise point the worker at the xc0 engine
    pcfg.encoding_type = "lc0"
    pcfg.inference_backend = "lc0_trt"
    pcfg.lc0_distill_model_name = (
        pcfg.lc0_distill_model_name or os.environ.get("LC0_DISTILL_MODEL", "")
    )
    if not pcfg.lc0_distill_model_name:
        raise RuntimeError(f"{LC0} no lc0 model: set lc0_distill_model_name")

    pcfg.max_game_length = LC0_BRP_PLIES - 1
    pcfg.telemetry_kind = "lc0"

    # a 3-ply window is never adjudicated. syzygy still fires on the
    # max_game_length branch in check_for_terminal, so probes can end short.
    pcfg.use_syzygy = False
    pcfg.allow_resignation = False
    pcfg.use_eval_draw = False
    pcfg.use_material_diff = False

    pcfg.add_root_noise = False
    pcfg.sample_moves = False
    pcfg.sampleable = {}

    return pcfg


def traces_to_startpos(prefix, start_fen):
    """Whether replaying prefix from STARTPOS actually reaches start_fen.

    history_uci is relative to the board's own start, so piece_odds,
    piece_training and random_init seed a custom fen and never trace back.
    """
    board = fastboard()
    for uci in prefix:
        if uci not in board.legal_moves():
            return False
        board.push_uci(uci)
    return board.fen() == start_fen


def replay_path(game_data, ply_i):
    """(fen, moves) reaching the position before moves_played[ply_i].

    Prefers the STARTPOS path so lc0's history planes are real, but only once
    the prefix is shown to land on start_fen.
    """
    moves_played = game_data['moves_played']
    history = game_data.get('history_uci') or []
    start_fen = game_data['start_fen']

    prefix_len = len(history) - len(moves_played)
    if prefix_len > 0 and traces_to_startpos(history[:prefix_len], start_fen):
        return STARTPOS_FEN, list(history[:prefix_len + ply_i])

    return start_fen, list(moves_played[:ply_i])


def lc0_replay_spec(cfg, game_data, ply_i, sfen, meta_extra=None):
    """Spec replaying one blunder position for the teacher.

    Returns None when the rebuilt board does not match the sfen recorded at
    that ply -- probing the wrong position is worse than probing none.
    """
    fen, moves = replay_path(game_data, ply_i)

    board = fastboard(fen)
    for uci in moves:
        if uci not in board.legal_moves():
            print(f"{LC0} illegal {uci} rebuilding ply {ply_i}", flush=True)
            return None
        board.push_uci(uci)

    if short_fen(board.fen()) != sfen:
        print(f"{LC0} ply {ply_i} rebuilt to {short_fen(board.fen())!r}, "
              f"expected {sfen!r}", flush=True)
        return None

    meta = {
        "scenario": LC0_BRP_SCENARIO,
        "vs_stockfish": False,
        "stockfish_is_white": False,
        "sfen": sfen,
        "ply_i": ply_i,
        # the result carries only the position, but the run's encoding may
        # need history planes, so the path travels with it
        "replay_fen": fen,
        "replay_moves": moves,
    }
    meta.update(meta_extra or {})
    return GameSpec(fen=fen, moves=moves, meta=meta, cfg=cfg)
