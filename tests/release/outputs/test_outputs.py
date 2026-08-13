import csv
import io
from pathlib import Path

import pytest

from ffmodel.release import (
    EvidenceRecord,
    PlayerPredictionRecord,
    Position,
    PredictionQuantiles,
    ReleaseContract,
    ReleaseContractError,
    ReleaseIdentity,
    SchemaValidationError,
    ScoringFormat,
)
from ffmodel.release.outputs import (
    CanonicalPredictionSet,
    OutputPlayer,
    OutputRenderConfig,
    RankingBasis,
    ranked_players,
    render_consumer_outputs,
)

DIGEST = "b" * 64
FIXTURES = Path(__file__).parent / "fixtures"


def _quantiles(mean):
    return PredictionQuantiles(mean, mean - 10, mean - 2, mean + 10)


def _canonical():
    contract = ReleaseContract(ReleaseIdentity(2026, "attempt-01"), (ScoringFormat.PPR, ScoringFormat.STANDARD))
    evidence = (EvidenceRecord("projection", "a" * 64, "fixture", contract.identity.cutoff),)
    players = (
        OutputPlayer(
            PlayerPredictionRecord("id-b", "Zoë Runner", Position.RB, {ScoringFormat.PPR: _quantiles(250.125), ScoringFormat.STANDARD: _quantiles(220.0)}),
            "nyj", True, evidence,
        ),
        OutputPlayer(
            PlayerPredictionRecord("id-a", "Álvaro Passer", Position.QB, {ScoringFormat.PPR: _quantiles(250.125), ScoringFormat.STANDARD: _quantiles(270.5)}),
            "dal", False, evidence,
        ),
    )
    return CanonicalPredictionSet(contract, DIGEST, players)


def test_canonical_json_contains_every_configured_format_and_t01_records():
    canonical = _canonical()
    restored = CanonicalPredictionSet.from_dict(canonical.to_dict(), contract=canonical.contract)
    assert restored == canonical
    assert canonical.to_bytes() == canonical.to_bytes()
    assert all(set(player["scoring"]) == {"ppr", "standard"} for player in canonical.to_dict()["players"])
    prediction_fields = {"schema_version", "player_id", "player_name", "position", "scoring"}
    assert all(PlayerPredictionRecord.from_dict({key: player[key] for key in prediction_fields}) for player in canonical.to_dict()["players"])


def test_multi_format_outputs_are_byte_stable_and_projections_of_canonical():
    canonical = _canonical()
    config = OutputRenderConfig(RankingBasis.MEAN, decimal_places=3)
    first = render_consumer_outputs(canonical, config)
    second = render_consumer_outputs(CanonicalPredictionSet.from_dict(canonical.to_dict(), contract=canonical.contract), config)
    assert first.files == second.files
    assert first.files["predictions.json"] == canonical.to_bytes()
    assert set(first.files) == {"predictions.json", "rankings_ppr.csv", "rankings_ppr.txt", "rankings_standard.csv", "rankings_standard.txt"}
    assert all(b"\r\n" not in content for content in first.files.values())
    assert first.files["rankings_ppr.csv"] == (FIXTURES / "rankings_ppr.csv").read_bytes()
    assert first.files["rankings_standard.csv"] == (FIXTURES / "rankings_standard.csv").read_bytes()
    assert first.files["rankings_ppr.txt"] == (FIXTURES / "rankings_ppr.txt").read_bytes()
    assert first.files["rankings_standard.txt"] == (FIXTURES / "rankings_standard.txt").read_bytes()
    assert first.files["rankings_ppr.csv"].decode("utf-8").startswith("target_season,attempt,package_digest")
    assert "provenance_digests" in first.files["rankings_ppr.csv"].decode("utf-8")
    assert "Zoë Runner" in first.files["rankings_ppr.txt"].decode("utf-8")


def test_csv_and_text_use_same_canonical_rank_order_with_explicit_tie_break():
    canonical = _canonical()
    config = OutputRenderConfig(RankingBasis.MEAN, decimal_places=2)
    expected_ids = [player.prediction.player_id for player in ranked_players(canonical, ScoringFormat.PPR, RankingBasis.MEAN)]
    assert expected_ids == ["id-a", "id-b"]
    outputs = render_consumer_outputs(canonical, config)
    csv_rows = list(csv.DictReader(io.StringIO(outputs.files["rankings_ppr.csv"].decode("utf-8"))))
    assert [row["player_id"] for row in csv_rows] == expected_ids
    assert [row["rank"] for row in csv_rows] == ["1", "2"]
    text = outputs.files["rankings_ppr.txt"].decode("utf-8")
    assert text.index("1. Álvaro Passer") < text.index("2. Zoë Runner")
    assert "mean=250.12" in text
    assert f"provenance={'a' * 64}" in text


def test_every_csv_value_is_a_formatted_projection_of_the_canonical_json():
    canonical = _canonical()
    canonical_players = {item["player_id"]: item for item in canonical.to_dict()["players"]}
    for basis in RankingBasis:
        outputs = render_consumer_outputs(canonical, OutputRenderConfig(basis, decimal_places=2))
        for scoring_format in canonical.contract.scoring_formats:
            rows = list(csv.DictReader(io.StringIO(outputs.files[f"rankings_{scoring_format.value}.csv"].decode("utf-8"))))
            expected_order = [item.prediction.player_id for item in ranked_players(canonical, scoring_format, basis)]
            assert [row["player_id"] for row in rows] == expected_order
            text = outputs.files[f"rankings_{scoring_format.value}.txt"].decode("utf-8")
            assert f"Ranking basis: {basis.value}" in text
            assert text.index(f"1. {canonical_players[expected_order[0]]['player_name']}") < text.index(f"2. {canonical_players[expected_order[1]]['player_name']}")
            for row in rows:
                source = canonical_players[row["player_id"]]
                assert row["target_season"] == str(canonical.contract.identity.target_season)
                assert row["attempt"] == canonical.contract.identity.attempt
                assert row["package_digest"] == canonical.package_digest
                assert row["scoring_format"] == scoring_format.value
                assert row["ranking_basis"] == basis.value
                assert row["position"] == source["position"]
                assert row["team"] == source["team"]
                assert row["cold_start"] == str(source["cold_start"]).lower()
                assert row["provenance_digests"] == "|".join(item["digest"] for item in source["provenance"])
                for field in ("mean", "p10", "p50", "p90"):
                    assert row[field] == f"{source['scoring'][scoring_format.value][field]:.2f}"


def test_missing_unknown_and_empty_configurations_fail_closed():
    canonical = _canonical()
    with pytest.raises(ReleaseContractError, match="ScoringFormat"):
        ranked_players(canonical, "auction", RankingBasis.MEAN)
    with pytest.raises(ReleaseContractError, match="RankingBasis"):
        ranked_players(canonical, ScoringFormat.PPR, "mean")
    with pytest.raises(ReleaseContractError, match="explicit"):
        OutputRenderConfig("mean")
    with pytest.raises(ReleaseContractError, match="at least one"):
        CanonicalPredictionSet(canonical.contract, DIGEST, ())
    broken = canonical.to_dict()
    broken["scoring_formats"] = ["ppr"]
    with pytest.raises(SchemaValidationError, match="exactly match"):
        CanonicalPredictionSet.from_dict(broken, contract=canonical.contract)
    incomplete = OutputPlayer(
        PlayerPredictionRecord("id-c", "Missing Format", Position.TE, {ScoringFormat.PPR: _quantiles(100)}),
        "buf", False, canonical.players[0].provenance,
    )
    with pytest.raises(ReleaseContractError, match="exactly match"):
        CanonicalPredictionSet(canonical.contract, DIGEST, (incomplete,))


def test_canonical_schema_rejects_identity_and_unknown_field_drift():
    canonical = _canonical()
    data = canonical.to_dict()
    data["unexpected"] = True
    with pytest.raises(SchemaValidationError, match="unknown"):
        CanonicalPredictionSet.from_dict(data, contract=canonical.contract)
    data = canonical.to_dict()
    data["attempt"] = "wrong"
    with pytest.raises(SchemaValidationError, match="identity"):
        CanonicalPredictionSet.from_dict(data, contract=canonical.contract)
