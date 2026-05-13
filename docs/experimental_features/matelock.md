# MateLock

When a forced mate is found anywhere in the tree during backpropagation, Xerces locks all future descents through the mating line — bypassing PUCT entirely until the game ends. Simultaneously, children that lead to the engine losing receive a decaying penalty on their PUCT score to discourage revisiting them.

## How It Works

During backprop, when a terminal node resolves as a win for the side to move, the parent records the winning move via an atomic compare-and-swap:

```cpp
// on terminal win during backprop
if (stm_wins) {
    p->set_must_visit_uci(n->uci);
} else if (v_scalar != 0.0f) {
    n->performance_penalty.fetch_add(1);
}
```

`set_must_visit_uci` uses a three-state machine (`0=empty`, `2=writing`, `1=ready`) so only one thread wins the write under concurrent descent. On subsequent descents into that node, the forced move is checked before any PUCT scoring:

```cpp
// selection — checked before PUCT
char forced_uci[16];
if (take_must_visit_uci(forced_uci)) {
    ++cc->count_must_visit;
    // walk directly to the mating child, no scoring
}
```

Losing children accumulate `performance_penalty`. Each time selection evaluates that child, the penalty is subtracted from its PUCT score and then decremented by one — decaying naturally as visits accumulate rather than permanently suppressing the move.

```cpp
int pen = ch->performance_penalty.load();
if (pen > 0) {
    score -= static_cast<float>(pen);
    ch->performance_penalty.fetch_sub(1);
}
```

Once a mate is locked, the remaining sim budget deepens and confirms the mating line rather than exploring alternatives. Xerces won't second-guess a known forced win just because a different move has an attractive prior.
