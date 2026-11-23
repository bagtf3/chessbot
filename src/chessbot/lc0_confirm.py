"""
lc0 confirmation chains, run from inside the runner.

A spec from the rescorer (a blunder or a teacher position) is searched by the lc0 server one ply at a time. Ply 0 always goes.
When lc0 plays xc0's own move there, or the spec is a teacher position, the
chain stops; otherwise lc0 plays on for both sides, one request per ply, up to
LC0_BRP_PLIES. The finished chain comes out as the meta the rescorer's
ingest_lc0_replay reads: moves_played, xerces_uci and tree_search_data keyed
by ply.

Teacher searches never sample: the teacher mode config has root noise and
move sampling off. A chain also ends when lc0's move leaves no legal moves,
so the server is never asked about a finished position.

A thread waits on the replies so each next ply goes out the moment the last
one lands. Finished chains wait on done until the runner's loop drains them:
the rescorer is not thread-safe.
"""
import queue
import threading
import uuid

from chessbot.lc0_replay import LC0_BRP_PLIES
from chessbot.lc0_server import TEACHER, is_searchable


class Lc0Confirmer(object):
    def __init__(self, router, client):
        self.router = router
        self.client = client
        self.replies = queue.Queue()
        router.add_client(client, self.replies)
        self.done = queue.Queue()
        # chain id -> {"spec", "played", "records"}
        self.chains = {}
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.run, daemon=True,
                         name="lc0-confirm").start()

    def submit(self, spec):
        if not is_searchable(spec.fen, spec.moves):
            return
        cid = uuid.uuid4().hex
        with self.lock:
            self.chains[cid] = {"spec": spec, "played": [], "records": {}}
        self.request(cid)

    def request(self, cid):
        chain = self.chains[cid]
        spec = chain["spec"]
        self.router.submit(self.client, {
            "fen": spec.fen, "moves": list(spec.moves) + chain["played"],
            "mode": TEACHER, "priority": False, "meta": {"chain": cid}})

    def run(self):
        while True:
            rep = self.replies.get()
            cid = (rep.get("meta") or {}).get("chain")
            with self.lock:
                chain = self.chains.get(cid)
            if chain is not None:
                self.step(cid, chain, rep["record"])

    def step(self, cid, chain, record):
        """Bank the ply, then ask for the next or finish the chain."""
        meta = chain["spec"].meta
        if record is not None:
            k = len(chain["played"])
            chain["records"][k] = record
            chain["played"].append(record["move_played"])
        played = chain["played"]
        spec = chain["spec"]
        more = (record is not None
                and len(played) < LC0_BRP_PLIES
                and meta.get("stream") != "teacher"
                and played[0] != meta.get("xc0_move")
                and is_searchable(spec.fen, list(spec.moves) + played))
        if more:
            self.request(cid)
            return
        with self.lock:
            del self.chains[cid]
        self.done.put({**meta, "game_id": cid, "moves_played": played,
                       "xerces_uci": played[0] if played else None,
                       "tree_search_data": chain["records"]})

    def take_done(self):
        out = []
        while True:
            try:
                out.append(self.done.get_nowait())
            except queue.Empty:
                return out

    def unfinished_specs(self):
        """Specs of every chain not yet done, to start over after a restart."""
        with self.lock:
            return [c["spec"] for c in self.chains.values()]

    def n_active(self):
        with self.lock:
            return len(self.chains)
