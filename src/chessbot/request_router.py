"""
Request router: many clients, one server process, replies routed back.

The runner owns every queue, so a rebuilt server or a respawned worker attaches
to the same ones and nothing a client holds goes stale. Clients put
{"client", "req"} on in_q (or call submit from inside the runner); each req
gets a router-wide rid and lands on the server's priority or normal queue by
req["priority"]. Replies carry the rid back and go to that client's reply
queue. Every request stays in the outstanding table until its reply lands, so
a server that dies takes nothing with it: resend_outstanding puts the lot back
for its replacement. A reply whose rid is no longer outstanding (a duplicate
from a resend) is dropped.

Two daemon threads, one per direction, so neither waits on the runner loop.
"""
import queue
import threading
import time
from itertools import count


class RequestRouter(object):
    def __init__(self, ctx, name, timeout_s):
        self.name = name
        self.timeout_s = timeout_s
        self.in_q = ctx.Queue()
        self.server_qs = {True: ctx.Queue(), False: ctx.Queue()}
        self.reply_in = ctx.Queue()
        self.ctx = ctx
        self.client_qs = {}
        # rid -> [client, req, sent_at, submitted_at]; sent_at moves on a
        # resend, submitted_at never does
        self.outstanding = {}
        self.lock = threading.Lock()
        self.rids = count()
        # replies delivered per lane (True = priority)
        self.done = {True: 0, False: 0}
        # requests sent again by resend_stale
        self.resends = 0

    @property
    def priority_q(self):
        return self.server_qs[True]

    @property
    def normal_q(self):
        return self.server_qs[False]

    def start(self):
        for target in (self.forward_requests, self.forward_replies):
            threading.Thread(target=target, daemon=True,
                             name=f"{self.name}-{target.__name__}").start()

    def add_client(self, client, reply_q=None):
        """A reply queue for client: an mp queue for a worker process, or any
        queue the caller passes (a thread inside the runner)."""
        q = reply_q if reply_q is not None else self.ctx.Queue()
        with self.lock:
            self.client_qs[client] = q
        return q

    def prune_clients(self, live):
        """Forget clients that are gone, and the requests only they wanted."""
        with self.lock:
            dead = [c for c in self.client_qs if c not in live]
            for c in dead:
                del self.client_qs[c]
            for rid in [r for r, e in self.outstanding.items() if e[0] in dead]:
                del self.outstanding[rid]
        return dead

    def submit(self, client, req):
        with self.lock:
            req["rid"] = next(self.rids)
            t = time.monotonic()
            self.outstanding[req["rid"]] = [client, req, t, t]
        self.server_qs[bool(req.get("priority"))].put(req)

    def forward_requests(self):
        while True:
            msg = self.in_q.get()
            self.submit(msg["client"], msg["req"])

    def forward_replies(self):
        while True:
            rep = self.reply_in.get()
            with self.lock:
                entry = self.outstanding.pop(rep["rid"], None)
                q = self.client_qs.get(entry[0]) if entry else None
                if entry:
                    self.done[bool(entry[1].get("priority"))] += 1
            if q is not None:
                q.put(rep)

    def resend_outstanding(self):
        """For a fresh server: clear what the old one left queued, then queue
        every outstanding request again in rid order."""
        for q in self.server_qs.values():
            drain(q)
        with self.lock:
            entries = sorted(self.outstanding.items())
            now = time.monotonic()
            for _, e in entries:
                e[2] = now
        for _, (_, req, *_) in entries:
            self.server_qs[bool(req.get("priority"))].put(req)
        return len(entries)

    def resend_stale(self):
        """Backstop for a request lost without a server death."""
        now = time.monotonic()
        with self.lock:
            stale = [e for e in self.outstanding.values()
                     if now - e[2] > self.timeout_s]
            for e in stale:
                e[2] = now
            self.resends += len(stale)
        for _, req, *_ in stale:
            self.server_qs[bool(req.get("priority"))].put(req)
        return len(stale)

    def lane_counts(self):
        """{priority: (outstanding, done)} for both lanes."""
        with self.lock:
            out = {True: 0, False: 0}
            for _, req, *_ in self.outstanding.values():
                out[bool(req.get("priority"))] += 1
            return {k: (out[k], self.done[k]) for k in (True, False)}

    def ages(self):
        """(max, mean) seconds since submit over every outstanding request,
        (0, 0) when none are out."""
        now = time.monotonic()
        with self.lock:
            a = [now - e[3] for e in self.outstanding.values()]
        return (max(a), sum(a) / len(a)) if a else (0.0, 0.0)


def drain(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
