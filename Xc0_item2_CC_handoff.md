# Item #2 handoff to CC

## User's clarified design (authoritative)

Finish the native LEAN/FATTY game-record pipeline. Every played ply is saved
in the final per-game `.pkl.gz`, with both `raw_visits` and a baseline `policy`.
Rescoring separately decides what enters `live_buffer`: that training sample
may use a modified policy, different weights, teacher routing, or be skipped.

**The saved policy does NOT need to equal the live training policy.** The
earlier Codex proposal to force equality was wrong. Preserve the game as a
reusable data source; do not overwrite its baseline with online SF corrections.

Read this alongside `Xc0_05_SEP_TODOs.md` and
`Xc0_05_SEP_item2_native_writer.md`. This clarification supersedes any earlier
proposal in chat to persist the live-buffer target back onto the saved ply.

## Desired flow

1. Looper captures every played ply before pushing its move: encoding, raw
   visits, STM search WDL, raw NN WDL, and review/search details. It writes the
   intermediate `{header, plies}` `.pkl`.
2. Rescorer reconstructs the game ONCE, walking its moves to prepare SF requests
   and any board-dependent data. Reuse the captured encoding and raw visits.
   Do not re-encode or reconstruct a board when the necessary data is already
   present. Asynchronous SF results can be finalized in a later iteration over
   the prepared plies; iterating data is fine, replaying the boards again is not.
3. Prepare the archival baseline policy once from raw visits, with the agreed
   legal-move floor and unified `blend_to_uniform` rule. Keep raw visits intact.
   Record result/value targets in STM perspective, plus available CPL/depth.
4. Derive a separate live-training sample from the prepared data. Apply the
   existing CPL decisions, SF policy corrections, sample weights, and LC0/BRP
   routing there. Modifications must not mutate the archival policy or values.
5. Writer only selects LEAN/FATTY fields and serializes prepared data. No board
   creation, move replay, encoding, or policy reconstruction in the writer.
6. Write completed `<gid>.pkl.gz`, append its official index row, then remove
   the intermediate `.pkl`. Keep staging and final-index roles distinct.

## Archival versus training data

| Data | Saved ply | Live-buffer sample |
| --- | --- | --- |
| raw_visits | Original sparse counts, unchanged | Input for target construction |
| policy | Baseline normalized/blended search policy | May be corrected toward SF |
| search/NN WDL | STM perspective | Consume in STM perspective |
| result/value target | Preserve the agreed baseline target | Follow the training path's target rule |
| CPL/depth | Store when actually analyzed | Drives acceptance/correction/routing |
| rejected/teacher-routed ply | Still saved | May produce no XC0 live sample |
| review details | Retained for FATTY only | Not needed by trainer |

Keep every played ply, including SF and validation moves. A move excluded from
training must not disappear from the game record. Where a search observation
is truly unavailable, preserve an explicit missing/empty observation rather
than fabricate visits, a zero CPL, or a populated encoding. Check the looper's
fallback ply creation: it currently can emit a move with `stm=None` and no
search fields. The single board pass can recover STM; assess which other fields
must be captured upstream to meet the per-ply schema.

## Partial Codex changes already in the working tree

These are unfinished edits, not a verified implementation. Preserve unrelated
CC/user edits. Inspect current code before changing anything; edits happened
during concurrent work and interrupted turns.

- `start_game` prepares `z_wdl`, `wdl_target`, baseline `policy`, KL, and
  `training_route` on the input plies. It uses captured `xc0h` for normal games.
- `policy_from_raw_visits` was added to create the baseline without converting
  candidate UCI moves back to indices. It floors counts to one, normalizes,
  then calls `blend_to_uniform(counts, 1e-6, prior_clip_max)`.
- `write_lean_record` no longer replays the board. It reads prepared fields.
  Gzip output now uses `.tmp` followed by `os.replace`.
- `finalize_game` now adds `games_seen` after the writer returns.
- `blend_wdl_stm` was added. Regular-game Q and CE no longer flip Black twice.
- LC0/BRP WDL consumers had extra Black flips removed because the common
  collector emits STM WDL. Recheck producers and consumers, not just comments.
- The equiv replay policy's clip/renormalize was replaced with unified blending.
- Codex removed the normal-game visit-count swap that promoted `xerces_uci`
  over the most-visited move. If this is still wanted for live training,
  restore it only on the training copy; never change archival raw visits.

### Correct this known wrong edit first

`finalize_game` currently contains:

```python
node['policy'] = list(zip(entry[2][0].tolist(), entry[2][1].tolist()))
```

This overwrites the saved baseline with the corrected live-buffer policy.
REMOVE that write-back. Keep separate local training policy data. Ensure no
other live-path modification aliases an archival array.

`training_route` was introduced by Codex, not requested by the user. Keep it
only if useful as metadata; do not make core serialization depend on a new
field unnecessarily. It must not control whether a ply is saved.

An interrupted patch intended to make `training_data_from_sf` return its
live record did NOT land in the last inspected code. It is unnecessary to
copy that record back into the saved ply under the clarified design.

## Remaining cleanup and correctness work

- Finish separating archival preparation from online training decisions.
  Avoid maintaining two baseline-policy implementations. Reuse computed data;
  do not add another dense/sparse round trip unless a consumer requires it.
- Keep float32 for probability computation; preserve the intended on-disk
  schema and precision at serialization. Do not accidentally write a tuple of
  arrays where downstream per-game readers expect a list of `(idx, value)`.
- Audit `build_raw_visits_sparse` in `mcts_utils.py`: counts currently use int16.
  Establish whether allowed visit budgets can overflow 32767. Counts must stay
  pristine; use an adequate count dtype if needed (indices may remain int16).
- Verify STM end to end for regular, SF, LC0, and BRP observations. The old
  `blend_wdl` helper still accepts white-POV input. Adapt or remove it only after
  checking actual remaining callers. Update misleading POV comments.
- Existing mirror fixtures in `tests/test_rescore_math.py` still feed reversed
  Black WDL for replay paths. Those describe the old schema and need updating
  if retained; do not reintroduce flips to satisfy stale fixtures.
- Make finalization/restart order coherent. Final index is the completion
  marker. `get_unprocessed` currently also skips `games_seen`; check whether
  analysis history can suppress an unfinished staged game. A partially written
  gzip must not be treated as completion. Do not delete the intermediate before
  the final artifact and index are successfully written.
- Avoid duplicate reviewability sampling between runner and rescorer. Ensure
  paired validation stays reviewable and non-reviewable games lose only FATTY
  extras, never baseline fields. Update consumers of the official game index
  and lean records as needed for the major-version schema.
- Check the ordinary reader and in-memory BRP reader against their respective
  producers. Probes still travel as `tree_search_data` through queues, not as
  normal indexed game files. Do not force them through the disk workflow.
- Remove dead locals, obsolete comments, and unused helpers from this refactor.
  Update #2/#5 status honestly when the integration is finished.

## Scope and working preferences

- User authorized completing the refactor; no further per-line approval needed.
- User explicitly said NOT to run tests. Do not launch tests, selfplay, SF,
  training, or GPU jobs unless separately authorized. Static inspection is fine.
- No tests were successfully run by Codex. Two small STM helper tests were
  added earlier, but do not establish integration correctness.
- Do not touch live run configuration, historic storage, or unrelated TODOs.
- Keep required `os._exit(0)` behavior. This is not permission to redesign
  shutdown or worker orchestration.
- Prefer direct Python failures over default-heavy config handling or custom
  validation wrappers. Keep the implementation small and readable.
- Do not claim measured speedups. The concrete improvement is removing the
  second board replay and redundant target preparation; SF cost is unchanged.

## Completion criteria (inspection, not an instruction to run tests)

- Exactly one board reconstruction/replay in the normal rescore workflow.
- Every input move appears in the saved game in order.
- Raw visits remain unchanged; saved policy stays the baseline even when the
  live policy is corrected or the sample is skipped.
- Writer serializes prepared fields only.
- STM conventions agree across all newly produced observations and consumers.
- Final artifact/index/source-file lifecycle is consistent with restart logic.
- Report precisely what is finished and what remains unverified.
