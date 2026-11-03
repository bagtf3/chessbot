"""
One batch of games with an lc0 net in the player seat.
Usage: python run_validation_only.py <run_tag> [--model t1-big] [--workers 2]
                                     [--selfplay N]

Default is one paired validation batch vs Stockfish (the run's
validation_config.yaml). --selfplay N plays N selfplay games drawn from the
run's config.yaml game_probs instead. Same harness (GameLooper, MCTS), console
logging and SF rescore as run_selfplay.py; no retrain. Writes the game index
rows, the rescore analysis and, for validation, one validation_history.jsonl
row in the run dir.
"""
import argparse
import json
import multiprocessing as mp
import queue
import sys
import time

from chessbot.game_utils import GameGenerator
from chessbot.rescore import Rescorer, SFRescoreThread
from chessbot.review import RecordKeeper
from chessbot.selfplay_runner import child_looper, drain_queue, parse_paths
from chessbot.utils import make_jsonable
from chessbot.validation import (
    build_validation_summary_from_rows, create_validation_config,
    print_validation_info,
)

PRINT_EVERY = 16
RESCORE_DRAIN_S = 1800


def append_index_row(cfg, meta):
    meta["beat_sf"] = False
    if meta.get("vs_stockfish"):
        result = meta["result"]
        sf_white = meta["stockfish_color"]
        meta["beat_sf"] = bool(
            (result > 0 and not sf_white) or (result < 0 and sf_white))
    with open(cfg.game_index_staging_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(meta), ensure_ascii=False) + "\n")


def tally(rows):
    w = d = 0
    for g in rows:
        res, sf_white = g["result"], g["stockfish_color"]
        if res == 0:
            d += 1
        elif (res > 0) != bool(sf_white):
            w += 1
    return w, d, len(rows) - w - d


def tick_rescorer(rescorer, sf_threads, recorder):
    rescorer.tick()
    rescorer.current_movetime_ms = sf_threads[0].movetime_ms
    recorder.training_queue = rescorer.training_data_size


def rescore_outstanding(rescorer):
    return len(rescorer.intake) + len(rescorer.pending)


def main(run_tag, model, n_workers, n_selfplay):
    cfg, _, val_yaml_path = parse_paths(run_tag)
    cfg.encoding_type = "lc0"
    cfg.inference_backend = "lc0_trt"
    cfg.lc0_distill_model_name = model

    gen = GameGenerator(cfg)
    if n_selfplay:
        specs = [gen.next_game() for _ in range(n_selfplay)]
        print(f"[val-only] {model}: {len(specs)} selfplay games, "
              f"{n_workers} workers", flush=True)
    else:
        val_cfg = create_validation_config(cfg, val_yaml_path)
        n_pairs = max(1, cfg.validation_games_per_batch // 2)
        specs = gen.validation_games(cfg=val_cfg, n_pairs=n_pairs)
        print(f"[val-only] {model}: {len(specs)} games at sf_depth "
              f"{val_cfg.sf_validation_depth}, {n_workers} workers", flush=True)
        print_validation_info(val_cfg)

    recorder = RecordKeeper(0, every_sec=60.0)
    recorder.primary_buffer_dir = cfg.primary_buffer_dir

    sf_game_q, sf_res_q = queue.Queue(), queue.Queue()
    sf_threads = []
    for i in range(max(1, cfg.rescore_n_sf_workers)):
        t = SFRescoreThread(sf_game_q, sf_res_q, cfg)
        t.id = f"sf{i}"
        t.start()
        sf_threads.append(t)
    rescorer = Rescorer(cfg, sf_game_q, sf_res_q)
    # every game keeps the fat record, selfplay included, for GameViewer
    rescorer.decide_reviewable = lambda scenario: True

    ctx = mp.get_context()
    game_q, recent_q, telemetry_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for spec in specs:
        game_q.put(spec)

    workers = []
    for i in range(n_workers):
        c = cfg.copy()
        c.sampleable = None
        c.id = f"w{i}"
        stop_ev = ctx.Event()
        p = ctx.Process(
            target=child_looper,
            args=(c, stop_ev, recent_q, telemetry_q, ctx.Queue(), game_q,
                  ctx.Event()),
            daemon=True,
        )
        p.start()
        workers.append((p, stop_ev))

    rows = []
    n_done = 0
    started = time.monotonic()
    limit_s = cfg.validation_max_minutes * 60
    try:
        # a selfplay game under min_game_length never reports back; the
        # all-workers-exited check ends the batch in that case
        while n_done < len(specs):
            for msg in drain_queue(telemetry_q):
                recorder.ingest_telemetry(msg)
            tick_rescorer(rescorer, sf_threads, recorder)
            recorder.maybe_log_results()
            try:
                game = recent_q.get(timeout=1.0)
            except queue.Empty:
                if time.monotonic() - started > limit_s:
                    print("[val-only] timed out", flush=True)
                    break
                if not any(p.is_alive() for p, _ in workers):
                    print("[val-only] all workers exited", flush=True)
                    break
                continue
            meta = game["meta"]
            n_done += 1
            recorder.ingest_recents(game)
            append_index_row(cfg, meta)
            rescorer.submit(meta["pkl_file"])
            if meta.get("scenario") != "paired_validation":
                continue
            rows.append(meta)
            if len(rows) % PRINT_EVERY == 0:
                w, d, l = tally(rows)
                mins = (time.monotonic() - started) / 60
                print(f"[val-only] {len(rows)}/{len(specs)}  W/D/L {w}/{d}/{l}"
                      f"  score {(w + 0.5 * d) / len(rows):.3f}  {mins:.0f} min",
                      flush=True)
    finally:
        for _, stop_ev in workers:
            stop_ev.set()
        for p, _ in workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

        recorder.maybe_log_results(force=True)
        if rows:
            build_validation_summary_from_rows(rows, val_cfg, model_epoch=0)

        print(f"[val-only] draining rescore, "
              f"{rescore_outstanding(rescorer)} games outstanding", flush=True)
        deadline = time.monotonic() + RESCORE_DRAIN_S
        while rescore_outstanding(rescorer) and time.monotonic() < deadline:
            tick_rescorer(rescorer, sf_threads, recorder)
            time.sleep(1.0)
        if rescore_outstanding(rescorer):
            print(f"[val-only] rescore drain timed out, "
                  f"{rescore_outstanding(rescorer)} games unfinished", flush=True)
        rescorer.push_analyzed(report=True)
        for t in sf_threads:
            t.close()
        rescorer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_tag")
    ap.add_argument("--model", default="t1-big")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--selfplay", type=int, default=0)
    a = ap.parse_args()
    sys.exit(main(a.run_tag, a.model, a.workers, a.selfplay))
