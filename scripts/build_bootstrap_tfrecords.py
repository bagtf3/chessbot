import os, time
import pickle
import random
import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf

from chessbot import SP_DIR
from chessbot.review import GameViewer, load_game_index, ANALYZE_PKL

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


TARGET_WRITTEN_POSITIONS = 3_500_000
OUT_DIR = "C:/Users/Bryan/Data/chessbot_data/bootstrap_tfrecords"
#RUN_TAGS = [
#    "cf_10x256x5_full_sims3",
#    "cf_10x256x5_full_sims4",
#    "cf_10x256x5_deep_sims1",
#    "film_16M_low_sims1",
#]

RUN_TAGS = ["cf_10x256x5_full_sims2", "film_16M_low_sims1"]
DRAW_RATE = 0.5
MAX_CPL = 45       # game-level CPL filter (mean clipped loss across game)
CPL_THRESHOLD = 60 # per-ply CPL filter passed to generate_training_data
SHARD_GAMES = 250
N_WORKERS = max([1, os.cpu_count() - 2])


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


def make_example(enc_in, mask, policy, value, weight):
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
                "value_out": float_feature(value),
                "weight": float_feature(weight),
            }
        )
    )
    return ex.SerializeToString()


def load_sf_df_for_games(run_tag, game_ids):
    pkl = os.path.join(SP_DIR, run_tag, ANALYZE_PKL)
    with open(pkl, "rb") as f:
        prev_run = pickle.load(f)

    df_all = prev_run["df_all"]
    return df_all[df_all["game_id"].isin(game_ids)].copy()


def iter_game_records(game, sf_df):
    # Pre-filter to this game's rows so GameViewer.init works on a tiny frame
    game_id = game["game_id"]
    game_sf = sf_df.loc[sf_df["game_id"] == game_id] if sf_df is not None else None

    gv = GameViewer(game["pkl_file"], sf_df=game_sf)

    if gv.result == 0 and random.random() > DRAW_RATE:
        return

    if len(gv.moves_uci) < 10:
        return

    x_list, m_list, p_list, z_list, v_list, r_list = gv.generate_training_data(
        cpl_threshold=CPL_THRESHOLD
    )
    if not x_list:
        return

    weight = 0.5 if gv.result == 0 else 1.0

    for enc_in, mask, policy, z, v in zip(x_list, m_list, p_list, z_list, v_list):
        value = 0.9 * z + 0.1 * v
        yield make_example(enc_in, mask, policy, value, weight)


def write_shard(shard_path, games, sf_df):
    n_games = 0
    n_pos = 0

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(shard_path, options=options) as w:
        for game in games:
            wrote_any = False

            for rec in iter_game_records(game, sf_df):
                w.write(rec)
                n_pos += 1
                wrote_any = True

            if wrote_any:
                n_games += 1

    return n_games, n_pos


def make_jobs(games_by_run):
    jobs = []
    shard_id = 0

    for run_tag, games in games_by_run.items():
        for i in range(0, len(games), SHARD_GAMES):
            jobs.append((shard_id, run_tag, games[i:i + SHARD_GAMES]))
            shard_id += 1

    random.shuffle(jobs)
    return jobs


def worker_init(seed_base):
    seed = seed_base + os.getpid()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def process_shard(job):
    shard_id, run_tag, games = job
    game_ids = [g["game_id"] for g in games]
    sf_df = load_sf_df_for_games(run_tag, game_ids)

    shard_name = f"{run_tag}_train_{shard_id:05d}.tfrecord.gz"
    shard_path = os.path.join(OUT_DIR, shard_name)

    n_games, n_pos = write_shard(shard_path, games, sf_df)

    return {
        "shard_id": shard_id,
        "run_tag": run_tag,
        "shard_name": shard_name,
        "n_games": n_games,
        "n_pos": n_pos,
        "sf_rows": len(sf_df),
    }


def main():
    begin = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    games_by_run = load_training_games()
    total_input_games = sum([len(v) for v in games_by_run.values()])

    print(f"workers={N_WORKERS}")
    print(f"max_cpl={MAX_CPL}")
    print(f"max_positions={TARGET_WRITTEN_POSITIONS:,}")
    print(f"training_games={total_input_games}")
    for rt in RUN_TAGS:
        print(f"  {rt}: {len(games_by_run.get(rt, []))}")

    jobs = make_jobs(games_by_run)
    random.shuffle(jobs)

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
                    f"[{res['shard_id']:05d}] {res['run_tag']}  "
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