"""The promotion gate, as code rather than as a convention.

Three defects in the hand-applied version are what these tests pin down: it only
watched the metrics somebody remembered to tabulate, it counted fold wins as if
the three holdouts were exchangeable when one of them straddles a pass-rate
regime shift, and it gated R-hat at a bright line sitting inside R-hat's own
Monte Carlo error.
"""

import pytest

from ffmodel.evaluation.acceptance import (
    RELATIVE_MATERIAL,
    compare_runs,
    format_report,
)


def _run(mae_by_fold, *, rhat=1.002, divergences=0, extra=None):
    out = {}
    for fold, mae in mae_by_fold.items():
        out[fold] = {
            "target": {"mae": mae, "crps": mae * 0.75, "n": 600},
            "seconds": 120.0,
            "diagnostics": {
                "target": {"max_rhat": rhat, "min_ess": 2000.0,
                           "divergences": divergences},
            },
        }
        if extra:
            out[fold].update(extra(fold))
    return out


def test_a_consistent_material_gain_is_accepted():
    base = _run({"2022": 1.00, "2023": 1.00, "2024": 1.00})
    cand = _run({"2022": 0.95, "2023": 0.96, "2024": 0.94})

    report = compare_runs(base, cand)

    assert report.accepted
    improved = {(m.stream, m.metric) for m in report.by_verdict("improved")}
    assert ("target", "mae") in improved


def test_a_change_that_flips_sign_across_folds_is_inconclusive_not_a_win():
    """Two wins and a large loss used to promote. It should not.

    This is the shape the 2023 holdout produces: it trains on a history ending
    in the second-largest pass-rate move in the window and scores a season that
    partially reverts, so it is not a third independent vote.
    """
    base = _run({"2022": 1.00, "2023": 1.00, "2024": 1.00})
    cand = _run({"2022": 0.96, "2023": 1.09, "2024": 0.97})

    report = compare_runs(base, cand)
    mae = next(m for m in report.metrics if (m.stream, m.metric) == ("target", "mae"))

    assert mae.wins == 2  # the old rule's "two of three" would have promoted
    assert not mae.consistent
    assert mae.verdict == "inconclusive"
    assert report.accepted  # inconclusive does not block; it also does not sell


def test_a_consistent_material_regression_blocks():
    base = _run({"2022": 1.00, "2023": 1.00, "2024": 1.00})
    cand = _run({"2022": 1.03, "2023": 1.04, "2024": 1.02})

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("target/mae regresses" in b for b in report.blockers)


def test_a_move_below_the_materiality_floor_is_not_a_verdict():
    # Consistent across all three folds, and far too small to act on. The gate
    # blocked a promotion on exactly this before the floor existed.
    base = _run({"2022": 1.00, "2023": 1.00, "2024": 1.00})
    cand = _run({"2022": 1.0006, "2023": 1.0006, "2024": 1.0006})

    report = compare_runs(base, cand)
    mae = next(m for m in report.metrics if (m.stream, m.metric) == ("target", "mae"))

    assert 0 < mae.pooled < RELATIVE_MATERIAL
    assert mae.verdict == "negligible"
    assert report.accepted


def _coverage(value):
    return lambda fold: {"snap": {"cov95": value, "mae": 1.0}}


def test_coverage_is_scored_in_points_not_as_a_ratio():
    """A ratio against a near-nominal baseline is a divide-by-almost-zero.

    0.960 to 0.966 against a 0.95 nominal is six tenths of a coverage point. As
    a relative change it reads +62%, and the gate refused a good promotion over
    it.
    """
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.960))
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.966))

    report = compare_runs(base, cand)
    cov = next(m for m in report.metrics if (m.stream, m.metric) == ("snap", "cov95"))

    assert cov.is_coverage
    assert cov.pooled == pytest.approx(0.006, abs=1e-9)
    assert cov.verdict == "negligible"
    assert report.accepted


def test_coverage_drifting_a_full_point_off_nominal_does_register():
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.950))
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.980))

    report = compare_runs(base, cand)
    cov = next(m for m in report.metrics if (m.stream, m.metric) == ("snap", "cov95"))

    assert cov.verdict == "regressed"
    assert not report.accepted


def test_divergences_block_because_a_divergence_is_a_real_event():
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9}, divergences=4)

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("divergences" in b for b in report.blockers)


def test_an_rhat_inside_its_own_noise_is_watched_rather_than_failed():
    """1.0107 was chased as a convergence defect. It is a reseed artifact.

    Seed 42 gives 1.01 and seed 7 gives 1.00 on the same model and data, and
    both settle at 1.00 by 2000 draws. A threshold at 1.01 is a coin flip on the
    seed, so the band up to 1.02 reports rather than rules.
    """
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9}, rhat=1.0107)

    report = compare_runs(base, cand)

    assert report.accepted
    watched = [d for d in report.diagnostics if not d["blocking"]]
    assert watched and "reseed band" in watched[0]["detail"]


def test_an_rhat_past_the_headroom_still_fails():
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9}, rhat=1.06)

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("R-hat" in b for b in report.blockers)


def test_every_component_is_watched_not_only_the_tabulated_ones():
    # The team model's R-hat sat in the JSON of every run for weeks without
    # appearing in a comparison table, because no table had a column for it.
    def team_only(fold):
        return {
            "diagnostics": {
                "target": {"max_rhat": 1.001, "min_ess": 2000.0, "divergences": 0},
                "team": {"max_rhat": 1.001, "min_ess": 12.0, "divergences": 0},
            }
        }

    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9}, extra=team_only)

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("team" in b and "ESS" in b for b in report.blockers)


def test_a_protected_stream_may_not_be_traded_away():
    def pass_stream(value):
        return lambda fold: {"pass_qb": {"mae": value}}

    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=pass_stream(5.0))
    cand = _run({"2022": 0.5, "2023": 0.5, "2024": 0.5}, extra=pass_stream(5.2))

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("protected stream pass_qb/mae" in b for b in report.blockers)


def test_a_candidate_that_stopped_reporting_a_metric_blocks():
    # Silently narrowing the comparison is the failure this module exists to
    # prevent, so it is a blocker rather than a missing row.
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9})
    for fold in cand:
        del cand[fold]["target"]["crps"]

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("does not report target/crps" in b for b in report.blockers)


def test_wall_clock_is_not_scored_as_a_metric():
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})

    report = compare_runs(base, cand)

    assert not any(m.stream == "seconds" for m in report.metrics)


def test_the_report_renders_both_unit_scales():
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.95))
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9}, extra=_coverage(0.97))

    text = format_report(compare_runs(base, cand), baseline="b", candidate="c")

    assert "pp" in text  # coverage in points
    assert "%" in text  # everything else relative
    assert "ACCEPTED" in text


def _scoring_run(coverage_by_fold, *, rmse=50.0):
    """The scoring walk-forward's schema, which differs from the volume one."""
    return {
        fold: {
            "ppr": {
                "mae": 40.0,
                "crps": 30.0,
                "rmse": rmse,
                "coverage_80": value,
                "coverage_95": 0.95,
                "n": 500,
                "scoring": "ppr",
            }
        }
        for fold, value in coverage_by_fold.items()
    }


def test_the_scoring_schemas_coverage_spelling_is_recognised():
    """``coverage_80`` and ``cov80`` are the same quantity.

    Only ``cov80`` was known, so on every scoring-run comparison — the ones the
    final promotion decisions are made on — coverage was scored as "higher is
    better" instead of "closer to nominal", in the wrong direction and silently.
    """
    base = _scoring_run({"2022": 0.80, "2023": 0.80, "2024": 0.80})
    cand = _scoring_run({"2022": 0.90, "2023": 0.90, "2024": 0.90})

    report = compare_runs(base, cand, protected=())
    cov = next(m for m in report.metrics if m.metric == "coverage_80")

    assert cov.is_coverage
    assert cov.pooled == pytest.approx(0.10)
    assert cov.verdict == "regressed"  # not an improvement, whatever its sign


def test_rmse_is_an_error_metric():
    base = _scoring_run({"2022": 0.80, "2023": 0.80, "2024": 0.80}, rmse=50.0)
    cand = _scoring_run({"2022": 0.80, "2023": 0.80, "2024": 0.80}, rmse=45.0)

    report = compare_runs(base, cand, protected=())
    rmse = next(m for m in report.metrics if m.metric == "rmse")

    assert rmse.pooled < 0
    assert rmse.verdict == "improved"


def test_an_unrecognised_metric_is_surfaced_rather_than_guessed():
    """The gate must not assign a direction it does not know.

    Assuming "not an error metric, therefore higher is better" is exactly how
    coverage came to be scored upside down.
    """
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0})
    cand = _run({"2022": 0.9, "2023": 0.9, "2024": 0.9})
    for fold in (base, cand):
        for payload in fold.values():
            payload["target"]["sharpness_index"] = 0.5

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("does not know which direction" in b for b in report.blockers)
    assert any(m.verdict == "unrecognized" for m in report.metrics)


def test_accepted_and_worthwhile_are_different_questions():
    """A change can be safe and pointless.

    The matched oof_* estimator regressed nothing and improved nothing either,
    at roughly forty times the fit cost. Reporting only "ACCEPTED" would hide
    the judgement that decision actually turned on.
    """
    base = _run({"2022": 1.0000, "2023": 1.0000, "2024": 1.0000})
    cand = _run({"2022": 0.9994, "2023": 0.9994, "2024": 0.9994})

    report = compare_runs(base, cand)

    assert report.accepted
    assert not report.worthwhile
    assert "weigh the cost" in format_report(report, baseline="b", candidate="c")
