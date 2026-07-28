"""Conditional lagged-efficiency screens for season-average volume."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_season_average import (
    VOLUME_OUTPUTS,
    add_walk_forward_volume_features,
)
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.features.season_efficiency import (
    CONDITIONAL_EFFICIENCY_SIGNALS,
    add_conditional_volume_efficiency_features,
)
from ffmodel.models.volume_season_average import DIRECT_SHARE_EFFICIENCY_FEATURES


def _interaction(stream: str, modifier: str) -> tuple[str, ...]:
    return tuple(
        f"{signal}_x_{modifier}"
        for signal in CONDITIONAL_EFFICIENCY_SIGNALS[stream]
    )


CONDITIONAL_VOLUME_CONFIGURATIONS = {
    "pass": {
        "none": (),
        "unconditional": DIRECT_SHARE_EFFICIENCY_FEATURES["pass"],
        "quality_only": ("prior_pass_quality_signal",),
        "room": _interaction("pass", "room"),
        "team": _interaction("pass", "team"),
        "uncertainty": _interaction("pass", "uncertainty"),
        "returning": _interaction("pass", "returning"),
        "changer": _interaction("pass", "changer"),
        "room_returning": _interaction("pass", "room_returning"),
        "unconditional_plus_room": (
            DIRECT_SHARE_EFFICIENCY_FEATURES["pass"]
            + _interaction("pass", "room")
        ),
        "unconditional_plus_continuity": (
            DIRECT_SHARE_EFFICIENCY_FEATURES["pass"]
            + _interaction("pass", "returning")
        ),
    },
    "target": {
        "none": (),
        "quality_only": ("prior_rec_quality_signal",),
        "quality_epa": CONDITIONAL_EFFICIENCY_SIGNALS["target"],
        "room": _interaction("target", "room"),
        "team": _interaction("target", "team"),
        "uncertainty": _interaction("target", "uncertainty"),
        "returning": _interaction("target", "returning"),
        "changer": _interaction("target", "changer"),
        "room_returning": _interaction("target", "room_returning"),
        "quality_plus_room": (
            ("prior_rec_quality_signal",) + _interaction("target", "room")
        ),
    },
    "carry": {
        "none": (),
        "unconditional": DIRECT_SHARE_EFFICIENCY_FEATURES["carry"],
        "centered": CONDITIONAL_EFFICIENCY_SIGNALS["carry"],
        "quality_only": ("prior_rush_quality_signal",),
        "room": _interaction("carry", "room"),
        "team": _interaction("carry", "team"),
        "uncertainty": _interaction("carry", "uncertainty"),
        "returning": _interaction("carry", "returning"),
        "changer": _interaction("carry", "changer"),
        "room_returning": _interaction("carry", "room_returning"),
        "unconditional_plus_room": (
            DIRECT_SHARE_EFFICIENCY_FEATURES["carry"]
            + _interaction("carry", "room")
        ),
        "unconditional_plus_continuity": (
            DIRECT_SHARE_EFFICIENCY_FEATURES["carry"]
            + _interaction("carry", "returning")
        ),
    },
}

REFERENCE_CONFIGURATION = {
    "pass": "unconditional",
    "target": "quality_only",
    "carry": "unconditional",
}


def conditional_volume_fold_metrics(
    data: SeasonAverageData, *, alpha: float = 300.0
) -> pd.DataFrame:
    """Screen efficiency/context interactions in leak-free season folds."""
    enriched = SeasonAverageData(
        data.team_rows,
        add_conditional_volume_efficiency_features(data.player_rows),
    )
    records = []
    for stream, configurations in CONDITIONAL_VOLUME_CONFIGURATIONS.items():
        output, _, observed_count = VOLUME_OUTPUTS[stream]
        for model, features in configurations.items():
            predicted = add_walk_forward_volume_features(
                enriched,
                include_efficiency=True,
                feature_overrides={stream: features},
                alpha=alpha,
            )
            team_games = pd.to_numeric(
                predicted["team_games"], errors="coerce"
            ).clip(lower=1)
            observed = (
                pd.to_numeric(predicted[observed_count], errors="coerce")
                / team_games
            )
            estimate = pd.to_numeric(predicted[output], errors="coerce")
            replacement = pd.to_numeric(
                predicted.get(
                    "is_replacement_player", pd.Series(0, index=predicted.index)
                ),
                errors="coerce",
            ).fillna(0).eq(1)
            valid = (~replacement) & observed.notna() & estimate.notna()
            if stream == "pass":
                valid &= predicted["position"].eq("QB")
            scored = pd.DataFrame(
                {
                    "season": predicted.loc[valid, "season"].to_numpy(dtype=int),
                    "error": (estimate[valid] - observed[valid]).to_numpy(dtype=float),
                }
            )
            for season, fold in scored.groupby("season"):
                records.append(
                    {
                        "stream": stream,
                        "model": model,
                        "season": int(season),
                        "n": int(len(fold)),
                        "absolute_error": float(np.abs(fold["error"]).sum()),
                        "squared_error": float(np.square(fold["error"]).sum()),
                        "mae": float(np.abs(fold["error"]).mean()),
                        "rmse": float(np.sqrt(np.square(fold["error"]).mean())),
                    }
                )
    return pd.DataFrame(records)


def conditional_volume_metrics(
    folds: pd.DataFrame, *, recent_start: int = 2019
) -> pd.DataFrame:
    """Pool conditional folds and compare stability to both reference models."""
    if folds.empty:
        return pd.DataFrame()
    records = []
    for stream, stream_folds in folds.groupby("stream"):
        baseline = stream_folds[stream_folds["model"].eq("none")].set_index(
            "season"
        )
        reference_name = REFERENCE_CONFIGURATION[stream]
        reference = stream_folds[
            stream_folds["model"].eq(reference_name)
        ].set_index("season")
        for model, group in stream_folds.groupby("model"):
            group = group.set_index("season")
            n = int(group["n"].sum())
            common = group.index.intersection(baseline.index)
            recent = common[common >= recent_start]
            records.append(
                {
                    "stream": stream,
                    "model": model,
                    "n": n,
                    "mae": float(group["absolute_error"].sum() / n),
                    "rmse": float(np.sqrt(group["squared_error"].sum() / n)),
                    "wins_vs_none": int(
                        (group.loc[common, "mae"] < baseline.loc[common, "mae"]).sum()
                    ),
                    "folds": int(len(common)),
                    "recent_wins_vs_none": int(
                        (group.loc[recent, "mae"] < baseline.loc[recent, "mae"]).sum()
                    ),
                    "recent_folds": int(len(recent)),
                    "reference_model": reference_name,
                    "wins_vs_reference": int(
                        (group.loc[common, "mae"] < reference.loc[common, "mae"]).sum()
                    ),
                    "recent_wins_vs_reference": int(
                        (group.loc[recent, "mae"] < reference.loc[recent, "mae"]).sum()
                    ),
                }
            )
    return pd.DataFrame(records).sort_values(["stream", "mae"]).reset_index(drop=True)
