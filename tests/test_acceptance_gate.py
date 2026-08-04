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


def test_a_bare_scalar_takes_its_direction_from_the_stream_name():
    """``carry_eligibility_brier`` is a number, not a metric block.

    Its direction lives in the stream's own name. Reading a metric literally
    called ``value`` as an error metric would be the same guess the gate refuses
    to make elsewhere, just in different clothing.
    """
    def brier(value):
        return lambda fold: {"carry_eligibility_brier": value}

    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=brier(0.150))
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=brier(0.140))

    report = compare_runs(base, cand)
    scalar = next(
        m for m in report.metrics if m.stream == "carry_eligibility_brier"
    )

    assert scalar.recognized
    assert scalar.pooled < 0
    assert scalar.verdict == "improved"


def test_a_bare_scalar_the_gate_cannot_name_is_still_refused():
    def mystery(value):
        return lambda fold: {"some_index": value}

    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=mystery(0.5))
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=mystery(0.4))

    report = compare_runs(base, cand)

    assert not report.accepted
    assert any("some_index" in b for b in report.blockers)


def test_coverage_prints_in_percentage_points_not_raw_proportions():
    """Half a coverage point must not print as "0.005pp".

    Coverage is carried internally as a proportion. Rendering it without
    scaling understated every coverage move by two orders of magnitude, in a
    column headed "pp" — and half a point is a number somebody might act on.
    """
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.950))
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.955))

    report = compare_runs(base, cand)
    cov = next(m for m in report.metrics if m.metric == "cov95")
    text = format_report(report, baseline="b", candidate="c")

    assert cov.pooled == pytest.approx(0.005)
    assert "+0.50pp" in text
    assert "+0.005pp" not in text


def _sized(cov_by_fold, n, metric="cov95"):
    return {
        fold: {"snap": {metric: value, "mae": 1.0, "n": n}}
        for fold, value in cov_by_fold.items()
    }


def test_the_coverage_floor_scales_with_the_sample():
    """One coverage point is 2.5 rows on one stream and 17 on another.

    A fixed floor calibrated for the 253-row quarterback stream is far too
    coarse for the 1,754-row target stream, and far too fine the other way.
    """
    small = compare_runs(
        _sized({"2022": 0.95, "2023": 0.95, "2024": 0.95}, n=84),
        _sized({"2022": 0.96, "2023": 0.96, "2024": 0.96}, n=84),
        protected=(),
    )
    large = compare_runs(
        _sized({"2022": 0.95, "2023": 0.95, "2024": 0.95}, n=1754),
        _sized({"2022": 0.96, "2023": 0.96, "2024": 0.96}, n=1754),
        protected=(),
    )

    tiny = next(m for m in small.metrics if m.is_coverage)
    big = next(m for m in large.metrics if m.is_coverage)

    assert tiny.material > big.material
    # The same one-point move is noise on 84 rows and a verdict on 1,754.
    assert tiny.verdict == "negligible"
    assert big.verdict == "regressed"


def test_a_run_without_row_counts_falls_back_to_the_fixed_floor():
    base = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.95))
    cand = _run({"2022": 1.0, "2023": 1.0, "2024": 1.0}, extra=_coverage(0.96))

    report = compare_runs(base, cand)
    cov = next(m for m in report.metrics if m.metric == "cov95")

    # Conservative, not sensitive: without the sample there is no basis for
    # scaling, so the gate declines to rule aggressively on a number it cannot
    # size.
    assert cov.sample_size == 0
    assert cov.material == pytest.approx(0.01)


def test_the_floor_never_collapses_to_zero_on_a_huge_fold():
    huge = compare_runs(
        _sized({"2022": 0.95}, n=10_000_000),
        _sized({"2022": 0.9501}, n=10_000_000),
        protected=(),
    )
    cov = next(m for m in huge.metrics if m.is_coverage)

    assert cov.material >= 0.002


def test_the_gate_refuses_runs_from_different_data_pulls():
    """Two runs on the same seasons are not automatically comparable.

    nflverse revises history, so a rebuilt cache changes values under an
    unchanged row count. Comparing across one attributes the rebuild's
    differences to the change under test, which this repo did twice before the
    fingerprint existed.
    """
    from ffmodel.evaluation.acceptance import frames_mismatch

    left = {"_frames": {"digest": "aaaa", "cache_dir": "a", "player_rows": 7234}}
    right = {"_frames": {"digest": "bbbb", "cache_dir": "b", "player_rows": 7234}}

    problem = frames_mismatch(left, right)

    assert problem is not None
    assert "different frames" in problem
    # The row counts matching is the trap, so the message has to say so.
    assert "Equal row counts do not mean" in problem


def test_matching_fingerprints_are_comparable():
    from ffmodel.evaluation.acceptance import frames_mismatch

    frames = {"digest": "aaaa", "cache_dir": "a", "player_rows": 7234}

    assert frames_mismatch({"_frames": frames}, {"_frames": dict(frames)}) is None


def test_an_unfingerprinted_run_is_reported_rather_than_assumed_fine():
    from ffmodel.evaluation.acceptance import frames_mismatch

    frames = {"digest": "aaaa", "cache_dir": "a", "player_rows": 7234}

    assert "predate" in frames_mismatch({}, {"_frames": frames})
    assert "predate" in frames_mismatch({"_frames": frames}, {})


def test_the_fingerprint_is_not_scored_as_a_fold():
    """It sits at the same level as the folds, so the parser has to skip it."""
    from ffmodel.evaluation.acceptance import compare_runs

    run = {
        "_frames": {"digest": "aaaa", "player_rows": 10},
        "2023": {"carry": {"n": 100, "mae": 1.0}},
    }
    other = {
        "_frames": {"digest": "aaaa", "player_rows": 10},
        "2023": {"carry": {"n": 100, "mae": 0.9}},
    }

    report = compare_runs(run, other)

    assert {c.stream for c in report.metrics} == {"carry"}


def test_a_version_change_is_reported_as_method_not_data():
    """A rehash must not read as a data difference.

    The digest changed when the combine was made non-commutative. Without a
    version, every run fingerprinted before that would look like it came from
    a different cache, and the gate would block comparisons that are fine.
    """
    from ffmodel.evaluation.acceptance import frames_mismatch

    old = {"_frames": {"version": 1, "digest": "aaaa", "cache_dir": "a"}}
    new = {"_frames": {"version": 2, "digest": "bbbb", "cache_dir": "a"}}

    problem = frames_mismatch(old, new)

    assert problem is not None
    assert "different versions of the hash" in problem
    assert "Re-fingerprint" in problem
