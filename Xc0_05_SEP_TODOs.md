# Xc0 Data + Retrain Overhaul TODOs (2026-09-05)

Corpus snapshot: 13,081 pretrain shards = ~133.95M records (13,081 x 10,240)
feeding `18m_gen2_pretrain` (running, ~33h ETA).

## 1. Looper data-save -> new protocol
- Make selfplay write the lean per-game sparse pkl.gz (migrated format), not the
  fat game log.
- Open Qs: write per-game or straight-to-shard? keep the fat `game_logs` log at
  all or replace it? which fields survive (raw_visits, sel_method, cpl, ...)?

## 2. Retrain buffer sizing -> 12 / 12 / 4
- primary 12, replay 12, historic 4 shards per retrain draw.
- Reconcile with the current replay_buffer constants (144/144, live 204,800) --
  these are sample counts, not buffer caps.

## 3. Rescorer: remove tscaler
- Strip the temperature-scaler path out of the rescore / target build.

## 4. Rescorer: mirror the migration script
- Restructure rescorer target build to match `migrate_game_logs_mp.py` shape
  (sparse policy, 50/50 result+search WDL, cpl, sel_method, ...).

## 5. Unified prior_clip_max <-> uniform_eps (one-stop shop)
- `uniform_eps = max(uniform_eps, eps_needed_so_max_prior <= prior_clip_max)`.
- Single function used at MCTS prior-blend time.

## 6. Apply #5 to retrain targets too
- Same clip-via-eps blend on stored policy targets so retrain doesn't sharpen
  the model.

## 7. Historic -> cold storage drain (HDD)
- After 4 historic shards are sampled, move them to cold storage on the HDD.
- Draining historic = the signal that it's time for a new re-pretrain.
- Overturns the "NEVER delete/modify historic" rule in CLAUDE.md notes + memory
  -- update those when this lands.
