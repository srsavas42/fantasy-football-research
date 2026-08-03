"""Compare two walk-forward runs and say whether the candidate is acceptable.

The gate this replaces was a convention applied by hand: read the two JSON
files, put the metrics you care about in a table, and promote if the candidate
wins two holdouts of three. Three things were wrong with that.

**It only watched what the author tabulated.** The team model's 1.0177 R-hat sat
in the validation output of every run for weeks without appearing in any
comparison table, because no table had a column for it. A gate that depends on
remembering to look is not a gate. This one enumerates every metric and every
component present in both runs and reports the whole surface.

**It counted fold wins as if the folds were exchangeable.** They are not. The
2023 holdout trains on a history ending in the second-largest year-over-year
pass-rate move in the window (-0.0132 in 2022) and then scores a season that
partially reverts. "Wins two of three" treats that fold as one independent vote,
when what it actually is is one draw from a different regime. With three folds
there is no honest way to estimate a standard error, so this reports the pooled
change against the fold-to-fold spread and is willing to return
``inconclusive`` — which, for a change worth a few tenths of a percent across
three correlated seasons, is usually the true answer.

**It gated R-hat at a bright line sitting inside R-hat's own noise.** The team
model's residual 1.0107 was chased for a while before a reseed showed the
statistic itself moves: seed 42 gives 1.01, seed 7 gives 1.00, and both settle
at 1.00 by 2000 draws. A threshold at 1.01 is therefore a coin flip on the seed,
not a convergence fact. Divergences are gated hard because a divergence is a
real event; R-hat is gated with headroom at 1.02 and anything in between is
flagged for a reseed rather than failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

# Metrics where a smaller number is better, and the nominal target for the
# coverage metrics, where what matters is distance from nominal in either
# direction — an 88% interval that covers 95% of outcomes is as wrong as one
# that covers 81%, and only one of those looks like an improvement to a rule
# that treats "higher is better".
LOWER_IS_BETTER = ("mae", "crps", "brier", "rmse")
# The volume and scoring walk-forwards spell coverage differently — ``cov80``
# and ``coverage_80``. Both belong here. Missing one meant the scoring runs, the
# ones the final promotion decisions are actually made on, had their coverage
# scored as "higher is better" rather than "closer to nominal", silently and in
# the wrong direction.
COVERAGE_NOMINAL = {
    "cov80": 0.80,
    "cov95": 0.95,
    "coverage_80": 0.80,
    "coverage_95": 0.95,
}
NOT_A_METRIC = ("n", "seconds", "scoring")
NOT_A_STREAM = ("seconds", "n", "diagnostics")

# How large a move has to be before it is worth a verdict. Coverage is measured
# in coverage points because its distance from nominal is often near zero, and a
# relative change against a near-zero baseline explodes: a 95% interval moving
# from 0.960 to 0.966 is six tenths of a coverage point, and reporting it as
# "+62%" would have the gate blocking promotions over nothing. Everything else
# is relative, where a floor of a quarter of a percent keeps the gate from
# ruling on differences smaller than the thing being measured.
COVERAGE_MATERIAL = 0.01
RELATIVE_MATERIAL = 0.0025

# R-hat below this is accepted; above it fails. The band between the classic
# 1.01 and this bound is reported as needing a reseed rather than treated as a
# verdict, because the statistic's own Monte Carlo error is that wide at this
# draw count.
RHAT_ACCEPT = 1.02
RHAT_WATCH = 1.01
MIN_BULK_ESS = 400.0


@dataclass
class MetricComparison:
    """One metric's behaviour across every holdout."""

    stream: str
    metric: str
    baseline: dict[str, float]
    candidate: dict[str, float]
    protected: bool = False

    @property
    def folds(self) -> list[str]:
        return sorted(set(self.baseline) & set(self.candidate))

    @property
    def is_coverage(self) -> bool:
        return self.metric in COVERAGE_NOMINAL

    @property
    def material(self) -> float:
        """Smallest move in this metric's own units worth calling a change."""
        return COVERAGE_MATERIAL if self.is_coverage else RELATIVE_MATERIAL

    @property
    def units(self) -> str:
        return "coverage points" if self.is_coverage else "relative"

    @property
    def recognized(self) -> bool:
        """Whether the gate knows which direction is better for this metric.

        It refuses to guess. The original code assumed anything it did not
        recognise as an error metric was one where higher is better, which is
        how ``coverage_80`` came to be scored upside down for every scoring-run
        comparison. An unrecognised metric is now surfaced instead of silently
        given a direction.

        A run may also report a bare number rather than a metric block —
        ``carry_eligibility_brier`` is one. There the direction is carried by
        the stream's own name, so that is what gets checked. Reading a metric
        literally called ``value`` as an error would be the same guess in
        different clothing.
        """
        name = self.stream if self.metric == "value" else self.metric
        return self.is_coverage or any(name.endswith(m) for m in LOWER_IS_BETTER)

    @property
    def relative_change(self) -> np.ndarray:
        """Per-fold change, signed so that negative is always better.

        Coverage is returned in coverage points rather than as a ratio. Its
        baseline distance from nominal is routinely near zero, and dividing by
        that turns a rounding-scale move into a headline percentage.
        """
        out = []
        for fold in self.folds:
            base, cand = self.baseline[fold], self.candidate[fold]
            if self.is_coverage:
                nominal = COVERAGE_NOMINAL[self.metric]
                out.append(abs(cand - nominal) - abs(base - nominal))
                continue
            if base == 0:
                out.append(0.0 if cand == 0 else np.inf)
            else:
                out.append((cand - base) / abs(base))
        return np.asarray(out, dtype=float)

    @property
    def pooled(self) -> float:
        change = self.relative_change
        return float(change.mean()) if len(change) else float("nan")

    @property
    def spread(self) -> float:
        """Standard error of the pooled change across folds.

        Three folds make this a crude estimate and it is not pretending
        otherwise. Its job is to stop a 0.05% pooled move that swings from -3%
        to +3% across seasons from being reported as an improvement.
        """
        change = self.relative_change
        if len(change) < 2:
            return float("inf")
        return float(change.std(ddof=1) / np.sqrt(len(change)))

    @property
    def consistent(self) -> bool:
        change = self.relative_change
        return bool(len(change) and (np.sign(change) == np.sign(change[0])).all())

    @property
    def verdict(self) -> str:
        """Where this metric lands, on two independent hurdles.

        A move has to clear both to earn a verdict: it must be larger than the
        fold-to-fold spread, so a change that flips sign across seasons is not
        read as a trend, and it must be larger than the materiality floor, so
        the gate does not rule on differences below the resolution of what is
        being measured.
        """
        if not self.recognized:
            return "unrecognized"
        pooled, spread = self.pooled, self.spread
        if not np.isfinite(pooled):
            return "unknown"
        if abs(pooled) < 1e-9:
            return "unchanged"
        if abs(pooled) < self.material:
            return "negligible"
        if not np.isfinite(spread) or abs(pooled) <= spread:
            return "inconclusive"
        return "improved" if pooled < 0 else "regressed"

    @property
    def wins(self) -> int:
        return int((self.relative_change < 0).sum())


@dataclass
class AcceptanceReport:
    metrics: list[MetricComparison]
    diagnostics: list[dict[str, object]]
    protected_tolerance: float
    blockers: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Whether the candidate damages nothing. Not whether it is worth doing.

        Keep the two apart. The matched ``oof_*`` estimator regressed nothing
        and improved nothing either — every metric below the materiality floor —
        at roughly forty times the fit cost. "Accepted" was never the question
        for that change; ``worthwhile`` is.
        """
        return not self.blockers

    @property
    def worthwhile(self) -> bool:
        return self.accepted and bool(self.by_verdict("improved"))

    def by_verdict(self, verdict: str) -> list[MetricComparison]:
        return [m for m in self.metrics if m.verdict == verdict]


def _flatten(run: Mapping[str, object]) -> dict[tuple[str, str], dict[str, float]]:
    """Pull every (stream, metric) series out of a walk-forward run JSON."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for fold, payload in run.items():
        if not isinstance(payload, Mapping):
            continue
        for stream, value in payload.items():
            if stream in NOT_A_STREAM:
                continue
            if isinstance(value, Mapping):
                for metric, number in value.items():
                    if metric in NOT_A_METRIC or not isinstance(number, (int, float)):
                        continue
                    out.setdefault((stream, metric), {})[fold] = float(number)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out.setdefault((stream, "value"), {})[fold] = float(value)
    return out


def _diagnostic_problems(run: Mapping[str, object], label: str) -> list[dict[str, object]]:
    """Sampler health for every component, not only the ones someone tabulated."""
    found: list[dict[str, object]] = []
    for fold, payload in run.items():
        diagnostics = payload.get("diagnostics") if isinstance(payload, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        for component, values in diagnostics.items():
            if not isinstance(values, Mapping):
                continue
            rhat = float(values.get("max_rhat", float("nan")))
            ess = float(values.get("min_ess", float("nan")))
            divergences = int(values.get("divergences", 0) or 0)
            issues = []
            if divergences > 0:
                issues.append(f"{divergences} divergences")
            if np.isfinite(rhat) and rhat >= RHAT_ACCEPT:
                issues.append(f"R-hat {rhat:.4f}")
            if np.isfinite(ess) and ess < MIN_BULK_ESS:
                issues.append(f"bulk ESS {ess:.0f}")
            watch = (
                np.isfinite(rhat) and RHAT_WATCH <= rhat < RHAT_ACCEPT and not issues
            )
            if issues or watch:
                found.append(
                    {
                        "run": label,
                        "fold": fold,
                        "component": component,
                        "max_rhat": rhat,
                        "min_ess": ess,
                        "divergences": divergences,
                        "blocking": bool(issues),
                        "detail": ", ".join(issues)
                        or f"R-hat {rhat:.4f} is inside the reseed band",
                    }
                )
    return found


def compare_runs(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    protected: Iterable[str] = ("pass_qb", "qb_workload"),
    protected_tolerance: float = 0.005,
) -> AcceptanceReport:
    """Full-surface comparison of a candidate walk-forward against a baseline.

    ``protected`` names streams a change is not allowed to damage even in
    exchange for gains elsewhere, and ``protected_tolerance`` is how much
    relative regression they may absorb before that becomes a blocker.
    """
    base_metrics, cand_metrics = _flatten(baseline), _flatten(candidate)
    protected = set(protected)

    metrics = [
        MetricComparison(
            stream=stream,
            metric=metric,
            baseline=base_metrics[(stream, metric)],
            candidate=cand_metrics[(stream, metric)],
            protected=stream in protected,
        )
        for (stream, metric) in sorted(base_metrics.keys() & cand_metrics.keys())
    ]

    diagnostics = _diagnostic_problems(candidate, "candidate")
    blockers: list[str] = []

    # A metric the baseline reports and the candidate does not is a silently
    # narrowed comparison, which is the failure mode this whole module exists
    # to prevent.
    dropped = sorted(base_metrics.keys() - cand_metrics.keys())
    for stream, metric in dropped:
        blockers.append(f"candidate does not report {stream}/{metric}")

    unrecognized = sorted(
        f"{m.stream}/{m.metric}" for m in metrics if not m.recognized
    )
    if unrecognized:
        blockers.append(
            "the gate does not know which direction is better for "
            + ", ".join(unrecognized)
            + " — add it to LOWER_IS_BETTER or COVERAGE_NOMINAL rather than "
            "letting it be scored by guess"
        )

    for comparison in metrics:
        if not comparison.recognized:
            continue
        if (
            comparison.protected
            and not comparison.is_coverage
            and comparison.pooled > protected_tolerance
        ):
            blockers.append(
                f"protected stream {comparison.stream}/{comparison.metric} "
                f"regresses {comparison.pooled:+.2%}, beyond the "
                f"{protected_tolerance:.1%} allowance"
            )
        elif comparison.verdict == "regressed":
            blockers.append(
                f"{comparison.stream}/{comparison.metric} regresses "
                f"{_render(comparison, comparison.pooled)} consistently across folds"
            )

    for problem in diagnostics:
        if problem["blocking"]:
            blockers.append(
                f"{problem['component']} on the {problem['fold']} fold: "
                f"{problem['detail']}"
            )

    return AcceptanceReport(
        metrics=metrics,
        diagnostics=diagnostics,
        protected_tolerance=protected_tolerance,
        blockers=blockers,
    )


def _render(comparison: MetricComparison, value: float) -> str:
    """Coverage in percentage points, everything else as a percentage.

    Coverage is carried internally as a proportion, so it is scaled by 100 to
    print. Labelling the raw 0.005 as "0.005pp" understates it by two orders of
    magnitude — it is half a coverage point, and half a point is a number
    somebody might act on.
    """
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100:+.2f}pp" if comparison.is_coverage else f"{value:+.2%}"


def format_report(report: AcceptanceReport, *, baseline: str, candidate: str) -> str:
    """A readable summary; the same content the docs tables were written by hand."""
    lines = [
        f"{candidate} vs {baseline}",
        "",
        f"{'stream/metric':30s} {'pooled':>10s} {'±se':>9s} {'folds':>6s}  verdict",
        "-" * 76,
    ]
    order = {
        "regressed": 0,
        "improved": 1,
        "inconclusive": 2,
        "negligible": 3,
        "unchanged": 4,
        "unknown": 5,
    }
    for m in sorted(
        report.metrics, key=lambda m: (order.get(m.verdict, 9), m.pooled)
    ):
        name = f"{m.stream}/{m.metric}"
        flag = " *" if m.protected else ""
        consistency = "" if m.consistent else " (sign varies)"
        lines.append(
            f"{name:30s} {_render(m, m.pooled):>10s} {_render(m, m.spread):>9s} "
            f"{m.wins:>3d}/{len(m.folds):<2d}  {m.verdict}{flag}{consistency}"
        )

    if report.diagnostics:
        lines += ["", "sampler diagnostics"]
        for problem in report.diagnostics:
            mark = "BLOCK" if problem["blocking"] else "watch"
            lines.append(
                f"  [{mark}] {problem['component']} @ {problem['fold']}: {problem['detail']}"
            )

    if not report.accepted:
        verdict = "NOT ACCEPTED"
    elif report.worthwhile:
        verdict = "ACCEPTED"
    else:
        # Safe and pointless. Worth saying out loud, because the cost of a
        # change is not in this table and a reader who sees "ACCEPTED" alone
        # will not weigh it.
        verdict = "ACCEPTED — but nothing improved materially; weigh the cost"
    lines += ["", verdict]
    for blocker in report.blockers:
        lines.append(f"  - {blocker}")
    return "\n".join(lines)
