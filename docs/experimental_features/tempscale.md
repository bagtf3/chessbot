# Prior Temperature Scaling

After priors are built for a node (softmax, `uniform_eps` blending, `prior_clip_max` clamping), Xerces optionally sharpens them further if the resulting distribution is still too spread out. The intuition: a high-entropy prior over legal moves — several moves rated near-equally plausible — is a weak signal for PUCT to explore with, even when the position itself is not actually balanced. Sharpening toward a target entropy gives search a clearer starting point without hand-picking a move.

## When It Fires

Two gates control activation, both config-driven and off by default:

```cpp
if (tempscale_entropy_target_ > 0.0f && k >= 5.0f && q_stm >= tempscale_trigger_q_) {
```

- `tempscale_entropy_target` (`0` = disabled) — the normed-entropy ceiling; distributions already below it are left alone.
- `tempscale_trigger_q` — a STM-POV Q floor (`-2` = always eligible, `0.5` = only sharpen when already winning).
- At least 5 legal moves — not worth the search for near-forced positions.

## Binary Search on Temperature

If the node's normed entropy exceeds the target, Xerces binary-searches for the sharpening exponent that lands closest to the target without ever increasing entropy:

```cpp
float lo = 0.3f, hi = 1.0f, best = 0.75f, best_dist = 1e30f;

for (int iter = 0; iter < 5; ++iter) {
    const float mid = (lo + hi) * 0.5f;
    const float inv_mid = 1.0f / mid;
    // raw_prior ** (1/mid), renormalize, re-blend uniform_eps + prior_clip_max, recompute entropy
    ...
    if (ne_mid < ne) {
        const float dist = std::abs(ne_mid - tempscale_entropy_target_);
        if (dist < best_dist) { best = mid; best_dist = dist; }
        if (dist < 0.05f) break;
    }
    if (ne_mid < tempscale_entropy_target_) lo = mid;
    else hi = mid;
}
```

Five iterations, bounds `[0.3, 1.0]` on the temperature `T` (exponent applied is `1/T`, so the search only ever sharpens, never flattens). A candidate is only accepted as `best` if it actually reduced entropy relative to the untouched distribution — a candidate that overshoots back up is rejected even if its distance to the target looks close. The final chosen temperature is reapplied through the same pipeline (raw prior power, renormalize, `uniform_eps` blend, `prior_clip_max` clamp) so the sharpened distribution obeys the same invariants as any other prior.

## Relationship to the Rescorer's Version

The same target-entropy/bisection shape is used a second time, independently, in `Rescorer`/`build_bootstrap_records.py` on the Python side — there it reshapes the final MCTS **visit distribution** into a policy training target (gated on zero centipawn loss and enough legal moves), not the root **prior** before search even starts. The two serve different purposes and are configured separately: this page is about the live, in-search C++ version gated by `tempscale_entropy_target`/`tempscale_trigger_q`.
