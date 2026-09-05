# Xc0 Data + Retrain Overhaul TODOs (2026-09-05)

Corpus snapshot: 13,081 pretrain shards = ~133.95M records (13,081 x 10,240)
feeding `18m_gen2_pretrain` (running, ~33h ETA).

Ordered by dependency. Take sequentially top-to-bottom.

## 1. Rescorer: remove tscaler
- Strip the temperature-scaler path out of the rescore / target build.
- First because it clears the deck inside the rescorer before #2 reshapes it.

## 2. Rescorer: mirror the migration script
- Restructure rescorer target build to match `migrate_game_logs_mp.py` shape
  (sparse policy, 50/50 result+search WDL, cpl, sel_method, ...).
- The spine: defines the lean per-game format everything downstream consumes.

## 3. Looper data-save -> new protocol
- Make selfplay write the lean per-game sparse pkl.gz (migrated format), not the
  fat game log.
- Depends on #2 (the lean write = whatever the rescorer now emits).
- Open Qs: write per-game or straight-to-shard? keep the fat `game_logs` log at
  all or replace it? which fields survive (raw_visits, sel_method, cpl, ...)?

## 4. Unified prior_clip_max <-> uniform_eps (one-stop shop)
- `uniform_eps = max(uniform_eps, eps_needed_so_max_prior <= prior_clip_max)`.
- Single function used at MCTS prior-blend time.
- Independent (MCTS-side); must precede #5.

## 5. Apply #4 to retrain targets too
- Same clip-via-eps blend on stored policy targets so retrain doesn't sharpen
  the model.
- Depends on #4 (the function) and #2 (applied inside the reshaped target build).

## 6. Retrain buffer sizing -> 12 / 12 / 4
- primary 12, replay 12, historic 4 shards per retrain draw.
- Reconcile with the current replay_buffer constants (144/144, live 204,800) --
  these are sample counts, not buffer caps.

## 7. sp-retrainment consumes 4-tuple natively
- Don't open + convert each historic shard to a 7-tuple at load time.
- sp-retrainment should read the 4-tuple `(xc0h, policy_sparse, wdl_target,
  "pretraining")` directly and just apply weights -- no per-shard rewrite.
- Precedes #8 (read natively before draining historic).

## 8. Historic -> cold storage drain (HDD)
- After 4 historic shards are sampled, move them to cold storage on the HDD.
- Draining historic = the signal that it's time for a new re-pretrain.
- Overturns the "NEVER delete/modify historic" rule in CLAUDE.md notes + memory
  -- update those when this lands.
- Budget: 13,081 shards / 4 per retrain = ~3,270 retrains before depletion.
- Last: riskiest (deletes/moves historic); depends on #6 (4-historic draw) and #7.
