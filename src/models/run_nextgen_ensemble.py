"""Train a next-generation ensemble with enriched player/lineup model inputs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.calibrate import PlattCalibrator
from src.models.ensemble import load_model_predictions
from src.models.run_enriched_experiments import train_catboost, train_lightgbm
from src.models.run_production_showdown import (
    _evaluate_slice_metrics as evaluate_slice_metrics,
)
from src.models.run_production_showdown import (
    evaluate_probabilities,
    load_production_stack_predictions,
)
from src.models.tabular_model import get_feature_columns, split_by_season
from src.utils.logging import setup_logging
from src.utils.paths import DOCS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

NEXTGEN_DIR = MODELS_DIR / "ensembles_nextgen"
NEXTGEN_RESULTS_PATH = DOCS_DIR / "experiments" / "nextgen_ensemble_results.json"
NEXTGEN_SUMMARY_PATH = DOCS_DIR / "experiments" / "nextgen_ensemble_results.md"

PRODUCTION_INPUT_COLS = ["neural_prob", "xgboost_prob", "elo_prob"]
ENRICHED_INPUT_COLS = ["enriched_catboost_prob", "enriched_lightgbm_prob"]


def load_enriched_matchup_dataset() -> pd.DataFrame:
    """Load the enriched matchup table used for next-gen ensemble inputs."""
    path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset_enriched.parquet"
    if not path.exists():
        msg = (
            f"Missing enriched matchup dataset at {path}. "
            "Run `python -m src.models.run_enriched_experiments` first."
        )
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def train_enriched_input_models(enriched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train enriched CatBoost/LightGBM inputs and return validation/test probabilities."""
    feature_cols = get_feature_columns(enriched_df)
    train_df, val_df, test_df = split_by_season(enriched_df)

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target_home_win"].to_numpy()
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["target_home_win"].to_numpy()
    X_test = test_df[feature_cols].fillna(0)

    logger.info("Training enriched CatBoost ensemble input...")
    catboost = train_catboost(
        X_train.to_numpy(),
        y_train,
        X_val.to_numpy(),
        y_val,
    )
    logger.info("Training enriched LightGBM ensemble input...")
    lightgbm = train_lightgbm(X_train, y_train, X_val, y_val)

    NEXTGEN_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(catboost, NEXTGEN_DIR / "enriched_catboost.joblib")
    joblib.dump(lightgbm, NEXTGEN_DIR / "enriched_lightgbm.joblib")
    (NEXTGEN_DIR / "enriched_feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2),
        encoding="utf-8",
    )

    val_probs = pd.DataFrame(
        {
            "game_id": val_df["game_id"].astype(str).to_numpy(),
            "actual_home_win": y_val,
            "enriched_catboost_prob": catboost.predict_proba(X_val.to_numpy())[:, 1],
            "enriched_lightgbm_prob": lightgbm.predict_proba(X_val)[:, 1],
        }
    )
    test_probs = pd.DataFrame(
        {
            "game_id": test_df["game_id"].astype(str).to_numpy(),
            "actual_home_win": test_df["target_home_win"].to_numpy(),
            "enriched_catboost_prob": catboost.predict_proba(X_test.to_numpy())[:, 1],
            "enriched_lightgbm_prob": lightgbm.predict_proba(X_test)[:, 1],
        }
    )
    val_probs.to_parquet(NEXTGEN_DIR / "enriched_input_predictions_val.parquet", index=False)
    test_probs.to_parquet(NEXTGEN_DIR / "enriched_input_predictions_test.parquet", index=False)
    return val_probs, test_probs


def load_or_train_enriched_input_predictions(
    enriched_df: pd.DataFrame,
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached enriched input probabilities or train them if needed."""
    val_path = NEXTGEN_DIR / "enriched_input_predictions_val.parquet"
    test_path = NEXTGEN_DIR / "enriched_input_predictions_test.parquet"
    if not refresh and val_path.exists() and test_path.exists():
        logger.info("Loading cached enriched ensemble input predictions from %s", NEXTGEN_DIR)
        return pd.read_parquet(val_path), pd.read_parquet(test_path)
    return train_enriched_input_models(enriched_df)


def assemble_meta_frames(
    production_val: pd.DataFrame,
    production_test: pd.DataFrame,
    enriched_val: pd.DataFrame,
    enriched_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge production and enriched probability inputs on game id."""
    val_frame = production_val.copy()
    test_frame = production_test.copy()
    for frame in (val_frame, test_frame, enriched_val, enriched_test):
        frame["game_id"] = frame["game_id"].astype(str)

    val_frame = val_frame.merge(
        enriched_val[["game_id", *ENRICHED_INPUT_COLS]],
        on="game_id",
        how="left",
    )
    test_frame = test_frame.merge(
        enriched_test[["game_id", *ENRICHED_INPUT_COLS]],
        on="game_id",
        how="left",
    )

    required_cols = [*PRODUCTION_INPUT_COLS, *ENRICHED_INPUT_COLS]
    if val_frame[required_cols].isna().any().any():
        raise ValueError("Validation meta-frame has missing ensemble input probabilities.")
    if test_frame[required_cols].isna().any().any():
        raise ValueError("Test meta-frame has missing ensemble input probabilities.")
    return val_frame, test_frame


def train_meta_variant(
    name: str,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    input_cols: list[str],
    slice_masks: dict[str, pd.Series],
) -> dict[str, Any]:
    """Train one logistic meta-model variant on validation predictions."""
    X_val = val_frame[input_cols].fillna(0.5).to_numpy()
    y_val = val_frame["actual_home_win"].to_numpy(dtype=int)
    X_test = test_frame[input_cols].fillna(0.5).to_numpy()
    y_test = test_frame["actual_home_win"].to_numpy(dtype=int)
    test_game_ids = test_frame["game_id"].astype(str).reset_index(drop=True)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_val, y_val)
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    calibrator = PlattCalibrator()
    calibrator.fit(val_probs, y_val)
    calibrated_test_probs = calibrator.predict_proba(test_probs)

    result: dict[str, Any] = {
        "status": "trained",
        "input_cols": input_cols,
        "raw": {
            "metrics": evaluate_probabilities(y_test, test_probs),
            "test_slices": evaluate_slice_metrics(
                y_test,
                test_probs,
                test_game_ids,
                slice_masks,
            ),
        },
        "calibrated": {
            "metrics": evaluate_probabilities(y_test, calibrated_test_probs),
            "test_slices": evaluate_slice_metrics(
                y_test,
                calibrated_test_probs,
                test_game_ids,
                slice_masks,
            ),
        },
        "model_weights": {
            col: float(weight) for col, weight in zip(input_cols, model.coef_[0], strict=False)
        },
        "model_intercept": float(model.intercept_[0]),
    }

    if name == "nextgen_full":
        joblib.dump(model, NEXTGEN_DIR / "meta_model.joblib")
        joblib.dump(calibrator, NEXTGEN_DIR / "calibrator.joblib")
        predictions = test_frame[["game_id", "actual_home_win"]].copy()
        predictions["nextgen_raw_prob"] = test_probs
        predictions["nextgen_calibrated_prob"] = calibrated_test_probs
        predictions.to_parquet(NEXTGEN_DIR / "nextgen_predictions_test.parquet", index=False)

    return result


def build_variant_leaderboard(
    production_results: dict[str, Any],
    variant_results: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a combined leaderboard for production and next-gen variants."""
    rows: list[dict[str, Any]] = []

    production_labels = {
        "production_neural_full_fusion": "production neural full fusion",
        "production_ensemble_raw": "production ensemble raw",
        "production_ensemble_calibrated": "production ensemble calibrated",
    }
    for model_name, result in production_results.items():
        rows.append(
            {
                "family": "production_stack",
                "label": production_labels[model_name],
                "variant": model_name,
                "mode": "saved",
                "metrics": result["metrics"],
            }
        )

    for variant_name, result in variant_results.items():
        for mode_name in ["raw", "calibrated"]:
            rows.append(
                {
                    "family": "nextgen_ensemble",
                    "label": f"{variant_name} / {mode_name}",
                    "variant": variant_name,
                    "mode": mode_name,
                    "metrics": result[mode_name]["metrics"],
                }
            )

    return sorted(
        rows,
        key=lambda row: (row["metrics"]["log_loss"], -row["metrics"]["accuracy"]),
    )


def build_summary_markdown(results: dict[str, Any]) -> str:
    """Render the next-gen ensemble results into a compact markdown report."""
    lines = [
        "# Next-Generation Ensemble Results",
        "",
        f"Generated at `{results['generated_at_utc']}`.",
        "",
        "## Leaderboard",
        "",
        "| Rank | Family | Label | Accuracy | Log Loss | Brier | ROC-AUC | ECE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(results["leaderboard"], start=1):
        metrics = row["metrics"]
        roc_auc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
        lines.append(
            "| "
            f"{rank} | {row['family']} | {row['label']} | "
            f"{metrics['accuracy']:.4f} | {metrics['log_loss']:.4f} | "
            f"{metrics['brier_score']:.4f} | {roc_auc} | "
            f"{metrics['calibration_error']:.4f} |"
        )

    verdict = results["verdict"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Best overall: `{verdict['best_overall']['label']}`",
            f"- Best next-gen entry: `{verdict['best_nextgen']['label']}`",
            f"- Production baseline: `{verdict['production_baseline']['label']}`",
            "- Next-gen minus production log-loss gap: "
            f"`{verdict['nextgen_minus_production_log_loss']:+.4f}`",
            "",
            "## Next-Gen Full Weights",
            "",
        ]
    )

    full_weights = results["variants"]["nextgen_full"]["model_weights"]
    for input_name, weight in sorted(
        full_weights.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):
        lines.append(f"- `{input_name}`: `{weight:.4f}`")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `production ensemble raw` is the current saved production meta-model output.",
            "- `production_retrained` retrains the same logistic meta-model on the same "
            "validation split, using only `neural + xgboost + elo`.",
            "- `nextgen_full` adds enriched CatBoost and LightGBM probabilities to the "
            "production input stack.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_nextgen_ensemble(*, refresh_enriched_inputs: bool = False) -> dict[str, Any]:
    """Train and evaluate the next-generation enriched ensemble variants."""
    enriched_df = load_enriched_matchup_dataset()
    _, _, test_df = split_by_season(enriched_df)
    from src.models.run_enriched_experiments import build_test_slice_masks

    slice_masks = build_test_slice_masks(enriched_df)
    enriched_val, enriched_test = load_or_train_enriched_input_predictions(
        enriched_df,
        refresh=refresh_enriched_inputs,
    )
    production_val = load_model_predictions("val")
    production_test = load_model_predictions("test")
    meta_val, meta_test = assemble_meta_frames(
        production_val,
        production_test,
        enriched_val,
        enriched_test,
    )

    variant_inputs = {
        "production_retrained": PRODUCTION_INPUT_COLS,
        "nextgen_catboost": [*PRODUCTION_INPUT_COLS, "enriched_catboost_prob"],
        "nextgen_lightgbm": [*PRODUCTION_INPUT_COLS, "enriched_lightgbm_prob"],
        "nextgen_full": [*PRODUCTION_INPUT_COLS, *ENRICHED_INPUT_COLS],
    }
    variants = {
        name: train_meta_variant(name, meta_val, meta_test, cols, slice_masks)
        for name, cols in variant_inputs.items()
    }

    production_results = load_production_stack_predictions()
    leaderboard = build_variant_leaderboard(production_results, variants)
    best_overall = leaderboard[0]
    best_nextgen = next(row for row in leaderboard if row["family"] == "nextgen_ensemble")
    production_baseline = next(
        row for row in leaderboard if row["variant"] == "production_ensemble_raw"
    )

    results = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "test_rows": int(len(test_df)),
        "production_results": production_results,
        "variants": variants,
        "leaderboard": leaderboard,
        "verdict": {
            "best_overall": best_overall,
            "best_nextgen": best_nextgen,
            "production_baseline": production_baseline,
            "nextgen_minus_production_log_loss": float(
                best_nextgen["metrics"]["log_loss"]
                - production_baseline["metrics"]["log_loss"]
            ),
        },
    }

    NEXTGEN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEXTGEN_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    NEXTGEN_SUMMARY_PATH.write_text(build_summary_markdown(results), encoding="utf-8")
    (NEXTGEN_DIR / "nextgen_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return results


def main() -> None:
    """Run and persist the next-generation ensemble report."""
    setup_logging()
    results = run_nextgen_ensemble()
    best = results["verdict"]["best_overall"]
    logger.info(
        "Next-gen ensemble winner: %s with log_loss=%.4f accuracy=%.4f",
        best["label"],
        best["metrics"]["log_loss"],
        best["metrics"]["accuracy"],
    )
    logger.info("Saved next-gen ensemble report to %s", NEXTGEN_SUMMARY_PATH)


if __name__ == "__main__":
    main()
