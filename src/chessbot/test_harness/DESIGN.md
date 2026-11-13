# Search test harness

Replays mined positions through xc0 search under configurable arms and records
the search at every check-in. Workers share the xc0 inference server; each
worker owns its own NN caches.

## Experiments it serves

1. Early stopping revisited: full inventory of check-in data, every check-in
   reached.
2. Find the SF-best: does search reach an SF-good move, how fast, and does it
   ever if it misses at first.
3. RSC, Qema, Qmm, QdS: which signals earn their keep, can RSC be tuned.
4. MateLock++: scout depth, node limits, trigger points.
5. Mistakes and refutations: what a refuted PV looks like in the tree, how fast
   search moves off it, which signals justify more budget.
6. pv_extender: first pluck of each `collect_many_leaves` skips PUCT (still
   obeys must_visit and penalty) and walks most-visited children to terminal.
7. Later: dynamic mbs [2, 4, 8, 16], in-cache WDL / prior updates, others.

The harness core (corpus, labels, runner, store) stays fixed across these;
an experiment is a set of arms plus a controller plus probes.

## Run pass option (RPO) controller

One job = one position x one arm. Check-ins land every 100 sims from sim 100.

At each check-in:
- snapshot (see Probes)
- `pick` = what xc0's own selection logic returns under the arm's settings;
  most-visited and RSC argmax are both recorded
- `would_stop` = the tier1 / tier2 rules fired at this check-in
- `good` = pick is within `GOOD_CP` of SF best, with labels at `MIN_LABEL_DEPTH`+

Then:
- `would_stop and good`: stop. Job done.
- `would_stop and not good`: record a wrong-stop event, keep searching.
- not `would_stop`: keep searching.
- budget exhausted: new attempt from a bare root (warm cache) with double the
  budget, attempt index on every row. Ends at `SIMS_CEILING` (32000) as
  "never found".

A pick with no label (or one under `MIN_LABEL_DEPTH`) goes to the SF pool
asynchronously; search continues and the stop decision waits on the label.

## C++ changes (pyfastchess)

All behind harness-only knobs; selfplay behaviour is unchanged with them off.

1. `es_record_all`: check-ins from sim 100 (`es_collect_start_ = 0`) instead of
   `min_sims - span`.
2. `es_observe_only`: tier rules are evaluated and the result recorded
   (`would_stop`, reason) but `es_tripped_` is never set. The budget ("full")
   stop still applies. Python owns the stop.
3. `break_on_checkin`: `collect_many_leaves` returns right after a check-in, so
   Python snapshots at exact sim counts and cache-hit runs can't blow through
   a check-in.
4. `EsCheckinRow` gains per-condition pass/fail for tier1 (conds 1-7) and tier2,
   RSC argmax, gap_block veto, `would_stop` and reason.
5. `Qdema` per node: EMA of the signed backup delta `(v - Q_pre) * pov`,
   updated next to `Qdelta_sign` (`mcts.cpp` backup at the root) and exposed
   on `ChildDetail`. Qdelta_sign keeps only the sign; Qdema keeps the size,
   reacts fast to a refutation and shrinks as Q catches up.
6. Per-experiment knobs as they are built: `pv_extender`, MateLock++ scout
   params.

## Cache policy

The NN caches (`raw_policy_cache`, `priors_cache`) are per worker process.
Each arm declares a cache group:

- `shared`: the arm runs alongside other `shared` arms on the same workers,
  job order shuffled so warm-cache advantage balances out across arms.
  Default for parameter sweeps that don't depend on cache state.
- `isolated`: the arm gets its own workers. For anything whose result depends
  on cache state: mbs sweeps, in-cache WDL / prior updates.

Trees are always fresh per job (`reuse_tree` off).

## Corpus

Records use the BRP shape: `fen`, `uci_path` (full history for xc0h),
`deep_evals`, `deep_depths`, `deep_best_move`, plus source fields (run tag,
game id, ply, model epoch, scenario) and stored stats (CPL, KL, CE, played
move).

Mined from `game_index.json` + `analyze_results_combined.pkl` (rebuilt first),
opening only sampled game logs. Stratified:
- high KL, high CE, high CPL, critical positions, random baseline
- each record keeps its stratum and sampling weight, so results can be
  reweighted to representative while hard positions are overrepresented

Built once per corpus, saved, reused across experiments.

## Labels

Wraps the BRP SF store (`ratchet_store`, `cached_deep_score`). Stored SF evals
are used as they are; SF runs only for a move with no eval or one under
`MIN_LABEL_DEPTH`. New evals ratchet into the corpus record, so every
experiment leaves the corpus better labelled.

## Probes

Snapshot functions run at each check-in; an experiment lists the ones it
wants. Initial set:
- `checkin`: the C++ `EsCheckinRow` (jsd, delta_12, dQ12, Q1, Qema1, per-cond
  flags, would_stop)
- `root`: root child table top-k (uci, N, Q, Qema, Qmm, Qdelta_sign, Qdema,
  rcs, prior, raw_prior, visit_share, boosted_N, forced_N)
- `pv`: principal variation with per-node N, P, Q
- `pick`: most-visited, RSC argmax, xc0 pick, label and `good` for each
- `cache`: cache-hit fraction since the last check-in (`count_cached`)

## Modules

```
test_harness/
  corpus.py       mine, stratify, weight, save/load
  labels.py       SF label lookup + async on-need SF pool
  arms.py         arm = config overlay + C++ knobs + cache group
  controllers.py  RPO (more later)
  probes.py       check-in snapshot functions
  runner.py       worker loop + launcher
  store.py        row writer, per-experiment merge
  analysis.py     shared readers and plots
```

Runner: worker loop modelled on `GameLooper.run` with many trees per worker
batched through the shared xc0 server (`make_remote_infer`). The launcher
expands (positions x arms) into jobs, assigns them to workers by cache group,
and shuffles order within each worker.

## Outputs

`D:\Data\chessbot_data\sidetests\<experiment>\`:
- `experiment.yaml`: arms, controller, probes, corpus path, model hash
- `rows_w<k>.pkl`: one row per (position, arm, attempt, check-in), per worker
- `jobs_w<k>.pkl`: one row per job: outcome, sims to first good, sims to stop,
  wrong-stop count, attempts, final pick
- merged after the run for analysis

## Later experiments

### Tree purity: original vs rebuilt

Search to N, then rebuild a bare root to the same N on the warm cache.
- cost: rebuild wall time cold vs warm, all-hit descents/sec, at several N
- difference: pick agreement, root Q / Qmm delta, KL between visit
  distributions, PV agreement
- as an arm: rebuild at a check-in and continue; does the clean tree reach
  `good` sooner than the original would have

Controller action: `rebuild` (bare root, cache kept), callable at any check-in.

### Refutation -> cache correction -> rebuild

Multi-step, since refutation detection doesn't exist yet. Oracle first, so
the payoff is measured before a detector is built.

1. Ground truth: label refutation events offline from RPO rows (a PV move
   that led, lost the lead, and SF says was bad). Feeds experiment 5.
2. Oracle correction: at the refutation, overwrite the cache WDL of the
   critical nodes (PV nodes on the refuted line), then rebuild and compare to
   the uncorrected rebuild. Two value sources as arms:
   - search-informed: the node's backed-up subtree WDL (no SF)
   - SF-informed: WDL from the SF label (upper bound)
3. Detector: replace the oracle trigger with signals found in step 1, using
   the search-informed source only.

Cache group `isolated`. Writes go through `raw_cache_lookup` +
`raw_cache_bulk_insert` (policy kept, WDL replaced); `priors_cache` entries
for corrected keys must be invalidated or updated to match.

### Refutation signals and Qdema

End-of-search snapshots of refuted leaders (top raw prior, heavy visits,
`visit_share` near 0, lost to a lower-prior pick) show Qema and Qdelta_sign
separating leader from pick; Q and Qmm barely do. The open question is lead
time: do the signals turn before visits move.

1. Observe: record Q, Qema, Qmm, Qdelta_sign, Qdema, rcs, visit_share per
   root child at every check-in on a refutation corpus; measure each
   signal's lead before the switch. Model which signals are needed.
2. Qdema as a selection term (arm): bonus/penalty `k * Qdema`, capped. The
   EMA only updates on visits, so a penalty that starves a node would freeze
   it; Qdema also decays by parent visits since `last_visit`.

Workflow: mine a corpus, form a hypothesis, test it on a freshly drawn corpus.

### Other

- policy blind vs value blind misses: was the SF-good move barely visited or
  visited with a wrong Q
- forced-visit probe: force visits into the SF-good move, does its Q ever
  overtake the pick
- stop frontier: offline replay of stop rules over RPO rows, sims saved vs
  wrong-stop rate
- criticality predictor: early check-in signals that predict a bad pick
- lc0 on the same corpus as a reference
- standing benchmark: fixed corpus + arm across model checkpoints
- wall time and nps per arm

## Fixed per experiment

- one pinned model checkpoint, hash recorded
- TRT fp16 as in selfplay
- no root noise, no move sampling, c_puct from the arm (no per-game draw)
- selfplay paused while the harness owns the GPU
