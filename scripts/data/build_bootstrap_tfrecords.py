import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
import pickle
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf

from chessbot import SP_DIR
from chessbot.review import GameViewer, load_game_index, ANALYZE_PKL

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


TARGET_WRITTEN_POSITIONS = 5_000_000
OUT_DIR = os.getenv("BOOTSTRAP_TFREC_DIR", "C:/Users/Bryan/Data/chessbot_data/training_data/wdl_062126")

RUN_TAGS = [
    "precond_run3", "precond_run2",
    ]

MAX_CPL = 30       # game-level CPL filter (mean clipped loss across game)
CPL_THRESHOLD = 20 # per-ply CPL filter passed to generate_training_data
SHARD_GAMES = 100
N_WORKERS = max([1, os.cpu_count() - 2])
Z_BLEND = 0.5      # weight on game result (Z); remainder goes to search WDL

SKIP_OPENING_PLIES = {
    "startpos": 12,
    "pre_opened_mini": 8,
}


def to_wdl(x):
    """Convert scalar x in [-1,1] to soft WDL [w, d, l]."""
    return np.array([max(float(x), 0.0),
                     1.0 - abs(float(x)),
                     max(-float(x), 0.0)], dtype=np.float32)


def bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )


def float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
    )


def check(a):
    if isinstance(a.get("pkl_file"), str):
        return os.path.exists(a["pkl_file"])
    return False


def load_training_games():
    all_games = []
    df_list = []

    for rt in RUN_TAGS:
        rd = os.path.join(SP_DIR, rt)
        ag = load_game_index(rd)
        ag = [a for a in ag if check(a)]

        for g in ag:
            g["run_tag"] = rt

        all_games += ag

        pkl = os.path.join(rd, ANALYZE_PKL)
        with open(pkl, "rb") as f:
            prev_run = pickle.load(f)

        df_all = prev_run["df_all"]
        df_means = prev_run["df_means"]

        df_all["clipped_loss"] = np.clip(df_all["loss"], -1000, 1000)
        clipped_cpl = df_all.groupby("game_id")["clipped_loss"].mean()
        df_means["overall_cpl"] = df_means.game_id.map(clipped_cpl)
        df_means["run_tag"] = rt
        df_list.append(df_means)

    df_trim = pd.concat(df_list).drop_duplicates(["game_id"]).sort_values("ts")
    df_trim = df_trim.query("scenario != 'random_endgame'").copy()

    meta = pd.DataFrame(all_games).drop_duplicates("game_id")
    if "overall_cpl" in meta.columns:
        del meta["overall_cpl"]

    meta = meta.query("plies >= 10")
    meta = meta.merge(
        df_trim[["game_id", "run_tag", "overall_cpl"]],
        on=["game_id", "run_tag"],
    )
    meta = meta.query(f"overall_cpl <= {MAX_CPL}").copy()

    games_by_run = {}
    for rt in RUN_TAGS:
        sub = meta.query("run_tag == @rt").copy()
        games = sub.to_dict("records")
        random.shuffle(games)
        games_by_run[rt] = games

    return games_by_run


def make_example(enc_in, mask, policy, value: np.ndarray, weight):
    enc_bytes = tf.io.serialize_tensor(
        tf.convert_to_tensor(enc_in)
    ).numpy()

    mask_bytes = tf.io.serialize_tensor(
        tf.convert_to_tensor(mask)
    ).numpy()

    pol_bytes = tf.io.serialize_tensor(
        tf.convert_to_tensor(policy)
    ).numpy()

    ex = tf.train.Example(
        features=tf.train.Features(
            feature={
                "enc_in": bytes_feature(enc_bytes),
                "mask": bytes_feature(mask_bytes),
                "policy_logits": bytes_feature(pol_bytes),
                "value_out": tf.train.Feature(
                    float_list=tf.train.FloatList(value=value.tolist())
                ),
                "weight": float_feature(weight),
            }
        )
    )
    return ex.SerializeToString()


def load_sf_df_for_games(run_tag, game_ids):
    df_all = SF_CACHE[run_tag]
    return df_all[df_all["game_id"].isin(game_ids)].copy()


def iter_game_records(game, sf_df):
    game_id = game["game_id"]
    game_sf = sf_df.loc[sf_df["game_id"] == game_id] if sf_df is not None else None

    gv = GameViewer(game["pkl_file"], sf_df=game_sf)

    if len(gv.moves_uci) < 10:
        return

    scenario = game.get("scenario", "")
    min_ply = SKIP_OPENING_PLIES.get(scenario, 0)

    x_list, m_list, p_list, z_list, v_list, wdl_list, r_list = gv.generate_training_data(
        cpl_threshold=CPL_THRESHOLD,
        skip_sf_moves=True,
        min_ply=min_ply,
    )
    if not x_list:
        return

    for enc_in, mask, policy, z, v, wdl in zip(
        x_list, m_list, p_list, z_list, v_list, wdl_list
    ):
        search_wdl = np.array(wdl, dtype=np.float32) if wdl is not None else to_wdl(v)
        wdl_target = Z_BLEND * to_wdl(z) + (1 - Z_BLEND) * search_wdl
        yield make_example(enc_in, mask, policy, wdl_target, 1.0)


def get_shard_start(out_dir):
    if not os.path.isdir(out_dir):
        return 0
    nums = []
    for name in os.listdir(out_dir):
        if not name.endswith(".tfrecord.gz"):
            continue
        stem = name.replace(".tfrecord.gz", "")
        try:
            nums.append(int(stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def make_jobs(games_by_run, shard_start=0):
    all_games = []
    for run_tag, games in games_by_run.items():
        for game in games:
            all_games.append((run_tag, game))
    random.shuffle(all_games)
    jobs = []
    for i in range(0, len(all_games), SHARD_GAMES):
        jobs.append((shard_start + i // SHARD_GAMES, all_games[i:i + SHARD_GAMES]))
    return jobs


SF_CACHE = {}

def worker_init(seed_base):
    seed = seed_base + os.getpid()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    for rt in RUN_TAGS:
        pkl = os.path.join(SP_DIR, rt, ANALYZE_PKL)
        with open(pkl, "rb") as f:
            prev = pickle.load(f)
        SF_CACHE[rt] = prev["df_all"]


def process_shard(job):
    shard_id, tagged_games = job

    by_run = {}
    for run_tag, game in tagged_games:
        by_run.setdefault(run_tag, []).append(game)

    records = []
    n_games = 0
    sf_rows = 0
    for run_tag, games in by_run.items():
        sf_df = load_sf_df_for_games(run_tag, [g["game_id"] for g in games])
        sf_rows += len(sf_df)
        for game in games:
            recs = list(iter_game_records(game, sf_df))
            if recs:
                records.extend(recs)
                n_games += 1

    random.shuffle(records)

    shard_name = f"train_{shard_id:05d}.tfrecord.gz"
    shard_path = os.path.join(OUT_DIR, shard_name)
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(shard_path, options=options) as w:
        for rec in records:
            w.write(rec)

    return {
        "shard_id": shard_id,
        "shard_name": shard_name,
        "n_games": n_games,
        "n_pos": len(records),
        "sf_rows": sf_rows,
    }


def main():
    begin = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    games_by_run = load_training_games()
    total_input_games = sum([len(v) for v in games_by_run.values()])

    shard_start = get_shard_start(OUT_DIR)

    print(f"workers={N_WORKERS}")
    print(f"max_cpl={MAX_CPL}")
    print(f"max_positions={TARGET_WRITTEN_POSITIONS:,}")
    print(f"training_games={total_input_games}")
    print(f"shard_start={shard_start}")
    for rt in RUN_TAGS:
        print(f"  {rt}: {len(games_by_run.get(rt, []))}")

    jobs = make_jobs(games_by_run, shard_start)

    total_games = 0
    total_pos = 0
    jobs_submitted = 0
    stop_submit = False

    max_in_flight = N_WORKERS * 2
    futures = set()

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=N_WORKERS,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(12345,),
    ) as ex:
        while len(futures) < max_in_flight and jobs_submitted < len(jobs):
            fut = ex.submit(process_shard, jobs[jobs_submitted])
            futures.add(fut)
            jobs_submitted += 1

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)

            for fut in done:
                res = fut.result()

                total_games += res["n_games"]
                total_pos += res["n_pos"]

                elapsed = time.time() - begin
                pps = total_pos / max([1e-9, elapsed])
                remain = max([0, TARGET_WRITTEN_POSITIONS - total_pos])
                eta_sec = remain / max([1e-9, pps])
                pct = 100.0 * total_pos / TARGET_WRITTEN_POSITIONS
                pct = min([100.0, pct])

                print(
                    f"[{res['shard_id']:05d}]  "
                    f"games={res['n_games']}  "
                    f"positions={res['n_pos']}  "
                    f"sf_rows={res['sf_rows']}  "
                    f"file={res['shard_name']}  "
                    f"total_pos={total_pos:,}  "
                    f"pct={pct:.1f}%  "
                    f"pps={pps:,.0f}  "
                    f"eta={eta_sec/60:.1f}m"
                )

                if total_pos >= TARGET_WRITTEN_POSITIONS:
                    stop_submit = True

            while (
                not stop_submit
                and len(futures) < max_in_flight
                and jobs_submitted < len(jobs)
            ):
                fut = ex.submit(process_shard, jobs[jobs_submitted])
                futures.add(fut)
                jobs_submitted += 1

            if stop_submit and not futures:
                break

    print()
    print(f"jobs_submitted={jobs_submitted}")
    print(f"total_games={total_games}")
    print(f"total_positions={total_pos}")


if __name__ == "__main__":
    mp.freeze_support()
    main()