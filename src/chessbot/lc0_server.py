"""
lc0 server: one process owns the lc0 model and a persistent C++ cache, and
searches single positions on request. It plays no games of its own.

Clients (the runner, xc0 loopers) send a position as start_fen + ucis. The
server rebuilds the board, searches it in its forest and replies with the
result. Requests come on two queues: priority (vs_leela moves) and normal
(blunder fixes and other teacher probes).

Request: {"rid", "fen", "moves", "mode", "priority", "meta"}; mode picks the
search config ("teacher" or "vs_leela"), priority the queue, and meta rides
back untouched. The rid is the router's (request_router.py).
Reply:   {"rid", "fen", "moves", "mode", "meta", "record"}; record is the ply
record an xc0 move logs. Clients never ask about a finished position; one
that arrives anyway, or whose root has no moves, is answered with record None.
"""
import os
import sys
import traceback
from queue import Empty

from pyfastchess import Board as fastboard

from chessbot.game_utils import GameSpec
from chessbot.lc0_replay import (VS_LEELA_SECTION, create_lc0_replay_config,
                                 lc0_config_yaml)
from chessbot.looper import GameLooper
from chessbot.mcts_utils import ChessGame

LC0_SCENARIO = "lc0_position"
TEACHER, VS_LEELA = "teacher", VS_LEELA_SECTION
# priority searches may run this far past games_at_once
PRIORITY_OVERFILL = 32
# half-second idle polls the run loop makes before returning to the outer loop
IDLE_WAIT = 600


def lc0_mode_configs(cfg):
    """Search config per request mode: the teacher base, and the base with the
    yaml's vs_leela section on top."""
    base = create_lc0_replay_config(cfg)
    base.telemetry_kind = "lc0"
    vs = base.copy()
    vs.update(lc0_config_yaml(cfg).get(VS_LEELA_SECTION) or {})
    return {TEACHER: base, VS_LEELA: vs}


class PositionSearch(ChessGame):
    """One position, searched once. The move is recorded, never pushed."""

    def __init__(self, req, cfg):
        meta = {"scenario": LC0_SCENARIO, "vs_stockfish": False,
                "stockfish_is_white": False}
        super().__init__(GameSpec(fen=req["fen"], moves=req["moves"],
                                  meta=meta, cfg=cfg))
        self.req = req
        self.priority = bool(req.get("priority"))

    def make_move_from_tree(self):
        if not self.tree.root().is_expanded:
            return False
        mv, method, xc0_mv = self.tree.best()
        if not mv:
            return True
        self.collect_tree_search_data(mv, method, xc0_mv)
        return True


class Lc0Server(GameLooper):
    """GameLooper over PositionSearch items fed from the request queues."""

    def __init__(self, mode_cfgs, telem_q, msg_q, priority_q, normal_q, reply_q):
        self.mode_cfgs = mode_cfgs
        self.priority_q = priority_q
        self.normal_q = normal_q
        self.reply_q = reply_q
        super().__init__(model=None, cfg=mode_cfgs[TEACHER], recent_q=None,
                         telem_q=telem_q, msg_q=msg_q)

    def pull_from_queue(self):
        """Priority requests go in behind the last priority search and may
        overfill by PRIORITY_OVERFILL; normal requests go to the back."""
        cap = self.config.games_at_once
        while len(self.active_games) < cap + PRIORITY_OVERFILL:
            req = take(self.priority_q)
            if req is None:
                break
            self.add_request(req, at=sum(g.priority for g in self.active_games))
        while len(self.active_games) < cap:
            req = take(self.normal_q)
            if req is None:
                break
            self.add_request(req)

    def add_request(self, req, at=None):
        if not is_searchable(req["fen"], req["moves"]):
            self.reply(req, None)
            return
        ps = PositionSearch(req, self.mode_cfgs[req["mode"]])
        if at is None:
            self.active_games.append(ps)
        else:
            self.active_games.insert(at, ps)
        self.forest.add_tree(ps.tree)

    def finalize_game_data(self, game):
        self.games_finished += 1
        self.reply(game.req, game.tree_data.get(0))
        self.update_partial_telemetry()
        game.recents.clear()

    def reply(self, req, record):
        self.reply_q.put({"rid": req["rid"], "fen": req["fen"],
                          "moves": req["moves"], "mode": req["mode"],
                          "meta": req.get("meta"), "record": record})

    def drop_caches(self):
        """The lc0 net never changes, so its cache outlives a pause."""


def is_searchable(fen, moves):
    """False for a finished position: mate, stalemate or a draw by rule."""
    board = fastboard(fen)
    for uci in moves:
        board.push_uci(uci)
    return not board.is_terminal()


def take(q):
    try:
        return q.get_nowait()
    except Empty:
        return None


def lc0_server_main(mode_cfgs, stop_ev, telem_q, msg_q, priority_q, normal_q,
                    reply_q, release_ev=None):
    """mp.Process target. The run loop returns after an idle spell; the
    server goes straight back to waiting until stop_ev, or until a
    drain_and_stop has emptied it.

    Exits with os._exit once the outgoing queues are flushed: a normal
    interpreter exit can hang on CUDA/TRT teardown, which would keep the
    process alive and the run from ending."""
    code = 1
    try:
        server = Lc0Server(mode_cfgs, telem_q, msg_q, priority_q, normal_q,
                           reply_q)
        server.release_ev = release_ev
        with server:
            while not stop_ev.is_set():
                server.run(stop_ev, wait_time=IDLE_WAIT)
                if server.stop_at_empty and not server.active_games:
                    break
        code = 0
    except Exception:
        traceback.print_exc()
    finally:
        # os._exit skips the queues' feeder threads; flush them or the last
        # replies are lost
        for q in (reply_q, telem_q):
            q.close()
            q.join_thread()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
