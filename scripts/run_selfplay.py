# run_selfplay.py
import os
import time
import queue as py_queue
import multiprocessing as mp


def now_s():
    return time.time()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def child_selfplay_main(child_id, out_dir, meta_q, stop_ev, cfg):
    """
    Selfplay loop.
    Writes full game artifacts to disk and pushes small pointers to meta_q.

    TODO:
    - create game, play it, compute meta
    - write artifact atomically (tmp then os.replace)
    - q.put({"path": ..., "meta": {...}})
    - obey stop_ev for graceful shutdown
    """
    ensure_dir(out_dir)

    while not stop_ev.is_set():
        time.sleep(0.5)

        # TODO: replace with real IDs and file writing
        game_id = str(int(now_s() * 1000)) + "_c" + str(child_id)
        fin = os.path.join(out_dir, game_id + ".jsonl")

        # TODO: write actual game record; use atomic write
        with open(fin, "w", encoding="utf-8") as f:
            f.write("{}\n")

        meta = {"game_id": game_id, "child_id": child_id, "created_s": now_s()}
        meta_q.put({"path": fin, "meta": meta})

    meta_q.put(None)


def process_game_and_accumulate(msg, accum, cfg):
    """
    Parent-side processing for one game pointer message.

    Returns how many observations were added to accum.

    TODO:
    - load msg["path"]
    - run validation / rescoring / chunk extraction
    - append examples into accum
    """
    _path = msg["path"]
    _meta = msg["meta"]
    _ = (_path, _meta, cfg)

    added = 0
    return added


def save_retrain_shard(accum, shard_dir, gen_idx, shard_idx, cfg):
    """
    Saves retrain data for the retrain worker.

    TODO:
    - serialize accum to disk (pkl or your chosen format)
    - return path to shard
    """
    ensure_dir(shard_dir)
    shard_path = os.path.join(shard_dir, f"gen{gen_idx}_shard{shard_idx}.pkl")
    _ = (accum, cfg)

    with open(shard_path, "wb") as f:
        f.write(b"")
    return shard_path


def retrain_worker_main(run_dir, shard_paths, cfg):
    """
    Heavy GPU retrain work.

    TODO:
    - load shards
    - build dataset
    - train
    - write updated weights to run_dir
    """
    _ = (run_dir, shard_paths, cfg)
    time.sleep(0.5)


def run_one_generation(gen_idx, cfg):
    """
    Single generation:
    - start selfplay children
    - consume messages, build shards
    - run retrain in fresh process (no overlap)
    - stop children cleanly
    """
    run_dir = cfg["run_dir"]
    games_dir = os.path.join(run_dir, "games")
    shards_dir = os.path.join(run_dir, "shards")
    ensure_dir(run_dir)
    ensure_dir(games_dir)
    ensure_dir(shards_dir)

    ctx = mp.get_context("spawn")
    meta_q = ctx.Queue(maxsize=cfg["meta_q_maxsize"])
    stop_ev = ctx.Event()

    procs = []
    for child_id in range(cfg["n_children"]):
        p = ctx.Process(
            target=child_selfplay_main,
            args=(child_id, games_dir, meta_q, stop_ev, cfg),
        )
        p.start()
        procs.append(p)

    done = 0
    shard_idx = 0
    shard_paths = []
    accum = []

    while done < cfg["n_children"]:
        try:
            msg = meta_q.get(timeout=cfg["meta_q_timeout_s"])
        except py_queue.Empty:
            msg = None

        if msg is None:
            done += 1
            continue

        added = process_game_and_accumulate(msg, accum, cfg)

        if added:
            cfg["obs_count"] += added

        if cfg["obs_count"] >= cfg["retrain_min_obs"]:
            shard_path = save_retrain_shard(
                accum, shards_dir, gen_idx, shard_idx, cfg
            )
            shard_paths.append(shard_path)
            shard_idx += 1
            accum = []
            cfg["obs_count"] = 0

            rp = ctx.Process(
                target=retrain_worker_main,
                args=(run_dir, list(shard_paths), cfg),
            )
            rp.start()
            rp.join()
            if rp.exitcode:
                stop_ev.set()
                raise RuntimeError("retrain failed")

    stop_ev.set()
    for p in procs:
        p.join()


def main():
    cfg = {
        "run_dir": os.path.join(os.getcwd(), "run_out"),
        "n_generations": 3,
        "n_children": 4,
        "retrain_min_obs": 100000,
        "meta_q_maxsize": 100000,
        "meta_q_timeout_s": 2.0,
        "obs_count": 0,
    }

    for gen_idx in range(cfg["n_generations"]):
        run_one_generation(gen_idx, cfg)


if __name__ == "__main__":
    main()
