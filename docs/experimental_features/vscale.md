# VScale

VScale compresses NN value estimates during backpropagation while letting terminal outcomes propagate at full strength. The idea is that true terminal states with value e.g. `(1, 0, 0)` are difficult to distinquish from "completely winning" NN evals e.g. `(0.99, 0.01, 0.0)`. Downscaling NN values allows true terminals to stand out better and influence search more effectively.

## How It Works

During backprop, the scalar used to update `W` (the accumulator for Q) is scaled by `vscale` for NN-evaluated nodes and left at full power for terminals:

```cpp
const float v_scalar = (wdl.win - wdl.loss) * (is_terminal ? 1.0f : vscale_);
...
n->W += v_scalar;
n->Q = n->W / static_cast<float>(n->visit_count());
```

Only `W` and `Q` are affected. The `p_win`, `p_draw`, `p_loss` accumulators always receive the full WDL — so the tree's WDL statistics remain uncompressed. VScale only affects the value signal that PUCT uses for move selection.

At `vscale=1.0` the behavior is standard. Values below 1.0 flatten Q differences between children, making visit counts carry more relative weight in selection. Values above 1.0 amplify the value signal. The active config uses `vscale=0.9`.
