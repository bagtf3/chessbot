"""Single owner of the centipawn-loss rules.

The rescorer, the teacher audit and the blunder pool each used to compute
their own loss, so their CPL numbers were never comparable. A scorer is built
per call site from one ScoringRules, which is what keeps them in step.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringRules:
    """equiv_frac of 0 disables the dynamic band entirely."""
    equiv_frac: float = 0.0
    equiv_min: float = 0.0
    equiv_max: float = 0.0
    zero_inside_equiv: bool = False
    swap_on_better: bool = False
    mate_cap: float = None
    require_positive_z: bool = True


@dataclass
class MoveScore:
    loss: float
    raw_delta: float
    best_uci: str
    best_cp: float
    swapped: bool
    missed_mate: bool
    equiv: float


class MoveScorer:
    def __init__(self, rules):
        self.rules = rules

    def score(self, best_uci, best_cp, played_uci, played_cp,
              blunder_cp=None, z_stm=None):
        """Loss for one played move against the engine's own best.

        Returns the possibly-swapped best alongside the loss; `swapped` is
        reported rather than applied so a caller can update whatever extra
        fields it tracks off the same decision.
        """
        r = self.rules
        raw = best_cp - played_cp

        equiv = 0.0
        if r.equiv_frac:
            equiv = min(max(abs(best_cp) * r.equiv_frac, r.equiv_min),
                        r.equiv_max)

        loss = raw
        out_uci, out_cp, swapped = best_uci, best_cp, False
        if r.zero_inside_equiv and abs(raw) <= equiv:
            loss = 0.0
        elif r.swap_on_better and raw <= -equiv:
            # the played move beat the engine's best; adopt it as the bar
            out_uci, out_cp, swapped = played_uci, played_cp, True

        # tested against the post-swap best and the post-equiv loss, matching
        # the original inline rule
        missed_mate = False
        if r.mate_cap is not None and blunder_cp is not None:
            z_ok = True if not r.require_positive_z else (
                z_stm is not None and z_stm > 0)
            missed_mate = (out_cp >= 1200 and played_cp >= 500 and z_ok
                           and loss >= blunder_cp)
            if missed_mate:
                loss = min(r.mate_cap, loss)

        return MoveScore(loss=loss, raw_delta=raw, best_uci=out_uci,
                         best_cp=out_cp, swapped=swapped,
                         missed_mate=missed_mate, equiv=equiv)


def make_scorer(kind, cfg=None):
    """kind: 'rescore' | 'lc0' | 'brp'.

    'rescore' and 'lc0' share one rule set so their CPLs stay comparable; they
    differ only in the missed-mate test, which needs a game outcome the
    teacher's probe positions do not have.
    """
    if kind == 'brp':
        # raw delta: probe_cpl drives eviction, the sf_ms ratchet and the
        # escalation bar, and those must stay continuous with pool history
        return MoveScorer(ScoringRules())

    if kind not in ('rescore', 'lc0'):
        raise ValueError(f"unknown scorer kind {kind!r}")

    return MoveScorer(ScoringRules(
        equiv_frac=0.1,
        equiv_min=cfg.rescore_equiv_min,
        equiv_max=cfg.rescore_equiv_max,
        zero_inside_equiv=True,
        swap_on_better=True,
        mate_cap=100.0,
        # a probe position carries no result, and out_cp >= 1200 already
        # asserts the side to move is winning
        require_positive_z=(kind == 'rescore'),
    ))
