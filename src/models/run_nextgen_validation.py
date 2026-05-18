"""Validate next-gen ensemble promotion readiness across reliability regimes."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.ensemble import ENSEMBLE_DIR, load_model_predictions
from src.models.run_nextgen_ensemble import NEXTGEN_DIR, load_enriched_matchup_dataset
from src.models.run_production_showdown import evaluate_probabilities
from src.models.tabular_model import split_by_season
from src.utils.logging import setup_logging
from src.utils.paths import DOCS_DIR

logger = logging.getLogger(__name__)

VALIDATION_RESULTS_PATH = DOCS_DIR / "experiments" / "nextgen_promotion_gate.json"
VALIDATION_SUMMARY_PATH = DOCS_DIR / "experiments" / "nextgen_promotion_gate.md"

MIN_OVERALL_LOG_LOSS_GAIN = 0.001
MIN_PLAYOFF_GAMES = 100
MIN_MISSING_PLAYER_GAMES = 100
MAX_ALLOWED_SLICE_LOG_LOSS_REGRESSION = 0.005

CRITICAL_SLICE_NAMES = {
    "season_2023_24",
    "season_2024_25",
    "high_context_confidence",
    "low_context_confidence",
    "home_back_to_back",
    "away_back_to_back",
    "dense_schedule",
}


def _as_bool_mask(mask: pd.Series | np.ndarray, index: pd.Series) -> pd.Series:
    """Normalize a row-order mask into a game-id-indexed boolean series."""
    return pd.Series(np.asarray(mask, dtype=bool), index=index.astype(str))


def _zero_float_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=frame.index)


def _column_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return _zero_float_series(frame)
    return frame[existing].fillna(0.0).sum(axis=1)


def _column_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return _zero_float_series(frame)
    return frame[existing].fillna(0.0).mean(axis=1)


def _top_quantile_mask(values: pd.Series, quantile: float) -> pd.Series:
    """Return a top-quantile mask, or an empty mask when the signal has no variance."""
    values = values.fillna(0.0)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(False, index=values.index)
    return values >= float(values.quantile(quantile))


def _bottom_quantile_mask(values: pd.Series, quantile: float) -> pd.Series:
    """Return a bottom-quantile mask, or an empty mask when the signal has no variance."""
    values = values.fillna(0.0)
    if values.nunique(dropna=True) <= 1:
        return pd.Series(False, index=values.index)
    return values <= float(values.quantile(quantile))


def build_validation_slices(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build promotion-gate slices from the held-out test frame."""
    frame = test_df.reset_index(drop=True).copy()
    game_ids = frame["game_id"].astype(str).reset_index(drop=True)
    slices: dict[str, pd.Series] = {}

    slices["all_test"] = _as_bool_mask(np.ones(len(frame), dtype=bool), game_ids)

    if "season_type" in frame.columns:
        season_type = frame["season_type"].astype(str).str.lower()
        is_regular_season = season_type.eq("regular season")
        is_playoffs = season_type.eq("playoffs")
    else:
        game_prefix = game_ids.str[:3]
        is_regular_season = game_prefix == "002"
        is_playoffs = game_prefix == "004"
    slices["regular_season"] = _as_bool_mask(is_regular_season, game_ids)
    slices["playoffs"] = _as_bool_mask(is_playoffs, game_ids)

    if "season" in frame.columns:
        for season in sorted(frame["season"].dropna().astype(str).unique()):
            slice_name = f"season_{season.replace('-', '_')}"
            slices[slice_name] = _as_bool_mask(frame["season"].astype(str) == season, game_ids)

    missing_value = _column_sum(
        frame,
        [
            "home_projected_player_value_missing",
            "away_projected_player_value_missing",
        ],
    )
    slices["missing_player_impact"] = _as_bool_mask(missing_value > 0, game_ids)
    slices["high_missing_value"] = _as_bool_mask(_top_quantile_mask(missing_value, 0.75), game_ids)

    context_confidence = _column_mean(
        frame,
        [
            "home_projected_top8_confidence_mean",
            "away_projected_top8_confidence_mean",
        ],
    )
    slices["high_context_confidence"] = _as_bool_mask(
        _top_quantile_mask(context_confidence, 0.75),
        game_ids,
    )
    slices["low_context_confidence"] = _as_bool_mask(
        _bottom_quantile_mask(context_confidence, 0.25),
        game_ids,
    )

    home_back_to_back = frame.get("home_back_to_back", pd.Series(0, index=frame.index)).fillna(0)
    away_back_to_back = frame.get("away_back_to_back", pd.Series(0, index=frame.index)).fillna(0)
    slices["home_back_to_back"] = _as_bool_mask(home_back_to_back.astype(float) > 0, game_ids)
    slices["away_back_to_back"] = _as_bool_mask(away_back_to_back.astype(float) > 0, game_ids)

    dense_schedule = _column_sum(
        frame,
        [
            "home_games_last_7",
            "away_games_last_7",
        ],
    )
    slices["dense_schedule"] = _as_bool_mask(_top_quantile_mask(dense_schedule, 0.75), game_ids)
    return slices


def _load_production_probabilities(test_df: pd.DataFrame) -> pd.DataFrame:
    """Load saved production ensemble probabilities aligned to the enriched test split."""
    test_game_ids = test_df["game_id"].astype(str).reset_index(drop=True)
    test_preds = load_model_predictions("test").copy()
    test_preds["game_id"] = test_preds["game_id"].astype(str)
    prob_cols = [column for column in test_preds.columns if column.endswith("_prob")]
    if not prob_cols:
        raise ValueError("No production probability columns were found for validation.")

    meta_model = joblib.load(ENSEMBLE_DIR / "meta_model.joblib")
    calibrator = joblib.load(ENSEMBLE_DIR / "calibrator.joblib")

    aligned = test_game_ids.to_frame(name="game_id").merge(
        test_preds[["game_id", "actual_home_win", *prob_cols]],
        on="game_id",
        how="left",
    )
    if aligned[prob_cols].isna().any().any():
        raise ValueError("Production ensemble inputs do not align with the enriched test split.")

    raw_probs = meta_model.predict_proba(aligned[prob_cols].fillna(0.5).to_numpy())[:, 1]
    calibrated_probs = calibrator.predict_proba(raw_probs)
    return pd.DataFrame(
        {
            "game_id": aligned["game_id"].astype(str),
            "actual_home_win": aligned["actual_home_win"].astype(int),
            "production_raw_prob": raw_probs,
            "production_calibrated_prob": calibrated_probs,
        }
    )


def _load_nextgen_probabilities() -> pd.DataFrame:
    """Load cached next-gen probabilities produced by run_nextgen_ensemble."""
    path = NEXTGEN_DIR / "nextgen_predictions_test.parquet"
    if not path.exists():
        msg = (
            f"Missing next-gen test predictions at {path}. "
            "Run `python -m src.models.run_nextgen_ensemble` first."
        )
        raise FileNotFoundError(msg)
    predictions = pd.read_parquet(path).copy()
    predictions["game_id"] = predictions["game_id"].astype(str)
    return predictions


def load_validation_frame(test_df: pd.DataFrame) -> pd.DataFrame:
    """Load production and next-gen probabilities in one aligned validation frame."""
    production = _load_production_probabilities(test_df)
    nextgen = _load_nextgen_probabilities()
    frame = production.merge(
        nextgen[
            [
                "game_id",
                "actual_home_win",
                "nextgen_raw_prob",
                "nextgen_calibrated_prob",
            ]
        ],
        on="game_id",
        how="left",
        suffixes=("_production", "_nextgen"),
    )
    if frame["nextgen_raw_prob"].isna().any():
        raise ValueError("Next-gen predictions do not align with the enriched test split.")
    if not (
        frame["actual_home_win_production"].astype(int)
        == frame["actual_home_win_nextgen"].astype(int)
    ).all():
        raise ValueError("Production and next-gen labels disagree after alignment.")
    frame = frame.rename(columns={"actual_home_win_production": "actual_home_win"})
    return frame.drop(columns=["actual_home_win_nextgen"])


def evaluate_model_comparison(
    scored_frame: pd.DataFrame,
    slice_masks: dict[str, pd.Series],
    *,
    production_col: str = "production_raw_prob",
    nextgen_col: str = "nextgen_raw_prob",
) -> dict[str, dict[str, Any]]:
    """Score production vs next-gen probabilities for every validation slice."""
    game_ids = scored_frame["game_id"].astype(str).reset_index(drop=True)
    y_true = scored_frame["actual_home_win"].to_numpy(dtype=int)
    production_probs = scored_frame[production_col].to_numpy(dtype=float)
    nextgen_probs = scored_frame[nextgen_col].to_numpy(dtype=float)

    results: dict[str, dict[str, Any]] = {}
    for slice_name, mask_lookup in slice_masks.items():
        aligned_mask = game_ids.map(mask_lookup).fillna(False).to_numpy(dtype=bool)
        game_count = int(aligned_mask.sum())
        if game_count == 0:
            results[slice_name] = {
                "status": "no_data",
                "game_count": 0,
                "reason": "No held-out games matched this validation slice.",
            }
            continue

        production_metrics = evaluate_probabilities(
            y_true[aligned_mask],
            production_probs[aligned_mask],
        )
        nextgen_metrics = evaluate_probabilities(
            y_true[aligned_mask],
            nextgen_probs[aligned_mask],
        )
        results[slice_name] = {
            "status": "scored",
            "game_count": game_count,
            "positive_rate": float(y_true[aligned_mask].mean()),
            "production": production_metrics,
            "nextgen": nextgen_metrics,
            "delta": {
                "log_loss": float(
                    nextgen_metrics["log_loss"] - production_metrics["log_loss"]
                ),
                "accuracy": float(
                    nextgen_metrics["accuracy"] - production_metrics["accuracy"]
                ),
                "brier_score": float(
                    nextgen_metrics["brier_score"] - production_metrics["brier_score"]
                ),
                "calibration_error": float(
                    nextgen_metrics["calibration_error"]
                    - production_metrics["calibration_error"]
                ),
            },
        }
    return results


def _gate_check(name: str, passed: bool, detail: str) -> dict[str, str | bool]:
    return {
        "name": name,
        "status": "pass" if passed else "block",
        "passed": passed,
        "detail": detail,
    }


def build_promotion_verdict(slice_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Decide whether the next-gen ensemble is ready for production promotion."""
    checks: list[dict[str, str | bool]] = []

    overall = slice_results.get("all_test", {})
    if overall.get("status") == "scored":
        overall_delta = float(overall["delta"]["log_loss"])
        checks.append(
            _gate_check(
                "overall_log_loss_gain",
                overall_delta <= -MIN_OVERALL_LOG_LOSS_GAIN,
                (
                    "Next-gen must beat production by at least "
                    f"{MIN_OVERALL_LOG_LOSS_GAIN:.3f} log loss; observed "
                    f"{overall_delta:+.4f}."
                ),
            )
        )
    else:
        checks.append(
            _gate_check(
                "overall_log_loss_gain",
                False,
                "The all-test slice was not scoreable.",
            )
        )

    playoff_count = int(slice_results.get("playoffs", {}).get("game_count", 0))
    checks.append(
        _gate_check(
            "playoff_coverage",
            playoff_count >= MIN_PLAYOFF_GAMES,
            (
                f"Need at least {MIN_PLAYOFF_GAMES} held-out playoff games; "
                f"found {playoff_count}."
            ),
        )
    )

    missing_count = int(slice_results.get("missing_player_impact", {}).get("game_count", 0))
    checks.append(
        _gate_check(
            "missing_player_coverage",
            missing_count >= MIN_MISSING_PLAYER_GAMES,
            (
                f"Need at least {MIN_MISSING_PLAYER_GAMES} held-out games with "
                f"nonzero missing-player value; found {missing_count}."
            ),
        )
    )

    regressions = []
    for slice_name in sorted(CRITICAL_SLICE_NAMES):
        result = slice_results.get(slice_name)
        if not result or result.get("status") != "scored":
            continue
        log_loss_delta = float(result["delta"]["log_loss"])
        if log_loss_delta > MAX_ALLOWED_SLICE_LOG_LOSS_REGRESSION:
            regressions.append(
                f"{slice_name} regressed by {log_loss_delta:+.4f} log loss"
            )

    checks.append(
        _gate_check(
            "critical_slice_regression",
            not regressions,
            (
                "No critical slice exceeded the allowed regression threshold."
                if not regressions
                else "; ".join(regressions)
            ),
        )
    )

    status = "ready" if all(bool(check["passed"]) for check in checks) else "blocked"
    failed_checks = [str(check["name"]) for check in checks if not bool(check["passed"])]
    if status == "ready":
        recommendation = "Candidate is ready for a shadow/live promotion review."
    elif failed_checks == ["missing_player_coverage"]:
        recommendation = "Promote only after real missing-player coverage is validated."
    elif failed_checks == ["playoff_coverage"]:
        recommendation = "Promote only after playoff coverage is validated."
    else:
        recommendation = "Promote only after all blocked coverage and regression gates pass."
    return {
        "status": status,
        "checks": checks,
        "recommendation": recommendation,
    }


def _fmt_metric(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def build_summary_markdown(results: dict[str, Any]) -> str:
    """Render a human-readable promotion gate report."""
    verdict = results["verdict"]
    lines = [
        "# Next-Gen Promotion Gate",
        "",
        f"Generated at `{results['generated_at_utc']}`.",
        "",
        "## Promotion Status",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Recommendation: {verdict['recommendation']}",
        "",
        "## Gate Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in verdict["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['detail']} |")

    lines.extend(
        [
            "",
            "## Slice Comparison",
            "",
            "| Slice | Games | Production Log Loss | Next-Gen Log Loss | Delta | "
            "Production Acc | Next-Gen Acc | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for slice_name, result in results["slice_results"].items():
        if result["status"] != "scored":
            lines.append(
                f"| {slice_name} | {result['game_count']} | n/a | n/a | n/a | "
                f"n/a | n/a | {result['status']} |"
            )
            continue
        lines.append(
            "| "
            f"{slice_name} | {result['game_count']} | "
            f"{_fmt_metric(result['production']['log_loss'])} | "
            f"{_fmt_metric(result['nextgen']['log_loss'])} | "
            f"{_fmt_metric(result['delta']['log_loss'])} | "
            f"{_fmt_metric(result['production']['accuracy'])} | "
            f"{_fmt_metric(result['nextgen']['accuracy'])} | "
            f"{result['status']} |"
        )

    playoff_count = int(results["slice_results"].get("playoffs", {}).get("game_count", 0))
    missing_count = int(
        results["slice_results"].get("missing_player_impact", {}).get("game_count", 0)
    )
    lines.extend(["", "## What This Means", ""])
    if verdict["status"] == "ready":
        lines.append("- The next-gen candidate clears the current promotion gates.")
    else:
        lines.append("- The aggregate next-gen gain is useful, but promotion remains gated.")
    if playoff_count > 0:
        lines.append(f"- Playoff coverage is now present with `{playoff_count}` held-out games.")
    else:
        lines.append("- Current enriched historical evaluation has no held-out playoff rows.")
    if missing_count > 0:
        lines.append(
            f"- Missing-player coverage is present with `{missing_count}` held-out games."
        )
    else:
        lines.append(
            "- Current missing-player value is zero throughout the held-out set, so "
            "availability-aware claims are not yet validated."
        )
    if verdict["status"] == "ready":
        lines.append(
            "- The next production step is a shadow/live promotion review with "
            "monitoring on playoff, calibration, and missing-player slices."
        )
        lines.append(
            "- Longer-term availability work should replace the historical absence "
            "proxy with richer official inactive history when available."
        )
    else:
        lines.append(
            "- The next production step is to improve the blocked coverage or "
            "regression checks, then rerun this gate."
        )
    return "\n".join(lines) + "\n"


def run_nextgen_validation() -> dict[str, Any]:
    """Run the cached next-gen promotion validation gate."""
    enriched_df = load_enriched_matchup_dataset()
    _, _, test_df = split_by_season(enriched_df)
    scored_frame = load_validation_frame(test_df)
    slice_masks = build_validation_slices(test_df)
    slice_results = evaluate_model_comparison(scored_frame, slice_masks)
    verdict = build_promotion_verdict(slice_results)

    results = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "test_rows": int(len(test_df)),
        "criteria": {
            "min_overall_log_loss_gain": MIN_OVERALL_LOG_LOSS_GAIN,
            "min_playoff_games": MIN_PLAYOFF_GAMES,
            "min_missing_player_games": MIN_MISSING_PLAYER_GAMES,
            "max_allowed_slice_log_loss_regression": (
                MAX_ALLOWED_SLICE_LOG_LOSS_REGRESSION
            ),
        },
        "slice_results": slice_results,
        "verdict": verdict,
    }

    VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    VALIDATION_SUMMARY_PATH.write_text(build_summary_markdown(results), encoding="utf-8")
    return results


def main() -> None:
    """Run and persist the next-gen promotion validation gate."""
    setup_logging()
    results = run_nextgen_validation()
    logger.info(
        "Next-gen promotion gate status: %s",
        results["verdict"]["status"],
    )
    logger.info("Saved next-gen promotion gate report to %s", VALIDATION_SUMMARY_PATH)


if __name__ == "__main__":
    main()
