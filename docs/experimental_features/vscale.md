# VScale

VScale compresses NN value estimates during backpropagation while letting terminal outcomes propagate at full strength. The idea is that a model's learned value is a noisy estimate — search should be able to correct it — while a known checkmate or draw is a hard fact that should dominate the tree unconditionally.

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

## Effect

With a compressed value signal, the search is less likely to latch onto a subtree purely because the NN rates it highly. Positions need to earn visits through both a good value estimate *and* accumulated evidence from exploration. This is particularly useful early in training when the value head is still poorly calibrated — a slightly wrong value won't dominate the tree as aggressively.
