"""Compare enriched benchmark winners against the current production stack."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.models.ensemble import ENSEMBLE_DIR, load_model_predictions
from src.models.run_enriched_experiments import (
    RESULTS_PATH as ENRICHED_RESULTS_PATH,
)
from src.models.run_enriched_experiments import (
    build_test_slice_masks,
)
from src.utils.logging import setup_logging
from src.utils.paths import DOCS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

SHOWDOWN_RESULTS_PATH = DOCS_DIR / "experiments" / "production_stack_showdown.json"
SHOWDOWN_SUMMARY_PATH = DOCS_DIR / "experiments" / "production_stack_showdown.md"
NEURAL_DIR = MODELS_DIR / "neural"


def evaluate_probabilities(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
) -> dict[str, float | None]:
    """Compute probability metrics for a binary classifier."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted_class = (probabilities >= 0.5).astype(int)

    bin_edges = np.linspace(0, 1, 11)
    calibration_error = 0.0
    total_count = 0
    for idx in range(len(bin_edges) - 1):
        mask = (probabilities >= bin_edges[idx]) & (probabilities < bin_edges[idx + 1])
        if idx == len(bin_edges) - 2:
            mask = (probabilities >= bin_edges[idx]) & (probabilities <= bin_edges[idx + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(probabilities[mask].mean()) - float(y_true[mask].mean()))
        calibration_error += gap * count
        total_count += count

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predicted_class)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "calibration_error": float(calibration_error / total_count) if total_count > 0 else 0.0,
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None
    )
    return metrics


def _evaluate_slice_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    test_game_ids: pd.Series,
    slice_masks: dict[str, pd.Series],
) -> dict[str, dict[str, float | int | None]]:
    slice_results: dict[str, dict[str, float | int | None]] = {}
    game_ids = test_game_ids.astype(str).reset_index(drop=True)

    for slice_name, mask_lookup in slice_masks.items():
        aligned_mask = game_ids.map(mask_lookup).fillna(False).to_numpy(dtype=bool)
        selected_count = int(aligned_mask.sum())
        if selected_count == 0 or selected_count == len(game_ids):
            continue

        metrics = evaluate_probabilities(y_true[aligned_mask], probabilities[aligned_mask])
        slice_results[slice_name] = {
            "game_count": selected_count,
            "positive_rate": float(y_true[aligned_mask].mean()),
            **metrics,
        }

    return slice_results


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_enriched_results() -> dict[str, Any]:
    if not ENRICHED_RESULTS_PATH.exists():
        msg = (
            f"Missing enriched experiment report at {ENRICHED_RESULTS_PATH}. "
            "Run `python -m src.models.run_enriched_experiments` first."
        )
        raise FileNotFoundError(msg)
    return _load_json(ENRICHED_RESULTS_PATH)


def _load_test_slice_masks() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    enriched_path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset_enriched.parquet"
    if not enriched_path.exists():
        msg = (
            f"Missing enriched dataset at {enriched_path}. "
            "Run the enriched experiment suite first."
        )
        raise FileNotFoundError(msg)
    enriched_df = pd.read_parquet(enriched_path)
    _, _, test_df = __import__(
        "src.models.tabular_model",
        fromlist=["split_by_season"],
    ).split_by_season(enriched_df)
    return test_df.reset_index(drop=True), build_test_slice_masks(enriched_df)


def _covered_test_game_ids(
    expected_game_ids: pd.Series,
    *prediction_frames: pd.DataFrame,
) -> pd.Series:
    """Return expected game ids covered by all supplied prediction frames."""
    covered_ids = set(expected_game_ids.astype(str))
    for frame in prediction_frames:
        if "game_id" not in frame.columns:
            return pd.Series(dtype=str)
        covered_ids &= set(frame["game_id"].astype(str))
    return expected_game_ids[expected_game_ids.astype(str).isin(covered_ids)].reset_index(
        drop=True
    )


def _coverage_summary(expected_game_ids: pd.Series, covered_game_ids: pd.Series) -> dict[str, int]:
    return {
        "expected_test_game_count": int(len(expected_game_ids)),
        "scored_game_count": int(len(covered_game_ids)),
        "missing_game_count": int(len(expected_game_ids) - len(covered_game_ids)),
    }


def load_production_stack_predictions() -> dict[str, dict[str, Any]]:
    """Load saved production model probabilities and score them on test slices."""
    test_df, slice_masks = _load_test_slice_masks()
    expected_test_game_ids = test_df["game_id"].astype(str).reset_index(drop=True)

    neural_path = NEURAL_DIR / "neural_predictions_test.parquet"
    if not neural_path.exists():
        raise FileNotFoundError(
            f"Missing neural predictions at {neural_path}. Train the production fusion model first."
        )

    neural_df = pd.read_parquet(neural_path).copy()
    neural_df["game_id"] = neural_df["game_id"].astype(str)
    neural_df = neural_df.rename(columns={"pred_home_win": "probability"})
    neural_df = neural_df.sort_values("game_id").reset_index(drop=True)

    val_preds = load_model_predictions("val")
    test_preds = load_model_predictions("test")
    test_preds["game_id"] = test_preds["game_id"].astype(str)
    test_game_ids = _covered_test_game_ids(expected_test_game_ids, neural_df, test_preds)
    coverage = _coverage_summary(expected_test_game_ids, test_game_ids)
    if test_game_ids.empty:
        raise ValueError("No enriched test rows are covered by saved production predictions.")
    if coverage["missing_game_count"] > 0:
        logger.warning(
            "Saved production artifacts cover %d/%d enriched test games; %d rows are skipped.",
            coverage["scored_game_count"],
            coverage["expected_test_game_count"],
            coverage["missing_game_count"],
        )

    production: dict[str, dict[str, Any]] = {}
    aligned_neural = test_game_ids.to_frame(name="game_id").merge(
        neural_df[["game_id", "probability", "actual_home_win"]],
        on="game_id",
        how="left",
    )
    if aligned_neural["probability"].isna().any():
        raise ValueError("Neural predictions do not fully align with the enriched test split.")

    neural_probs = aligned_neural["probability"].to_numpy()
    neural_actuals = aligned_neural["actual_home_win"].to_numpy(dtype=int)
    production["production_neural_full_fusion"] = {
        "status": "scored",
        "source": "models/neural/neural_predictions_test.parquet",
        "coverage": coverage,
        "metrics": evaluate_probabilities(neural_actuals, neural_probs),
        "test_slices": _evaluate_slice_metrics(
            neural_actuals,
            neural_probs,
            test_game_ids,
            slice_masks,
        ),
    }

    prob_cols = [column for column in val_preds.columns if column.endswith("_prob")]
    meta_model = joblib.load(ENSEMBLE_DIR / "meta_model.joblib")
    calibrator = joblib.load(ENSEMBLE_DIR / "calibrator.joblib")

    ensemble_test = test_game_ids.to_frame(name="game_id").merge(
        test_preds[["game_id", "actual_home_win", *prob_cols]],
        on="game_id",
        how="left",
    )
    if ensemble_test[prob_cols].isna().any().any():
        raise ValueError("Ensemble inputs do not fully align with the enriched test split.")

    X_test = ensemble_test[prob_cols].fillna(0.5).to_numpy()
    y_test = ensemble_test["actual_home_win"].to_numpy(dtype=int)
    raw_probs = meta_model.predict_proba(X_test)[:, 1]
    calibrated_probs = calibrator.predict_proba(raw_probs)

    for model_name, probabilities, source in [
        (
            "production_ensemble_raw",
            raw_probs,
            "models/ensembles/meta_model.joblib",
        ),
        (
            "production_ensemble_calibrated",
            calibrated_probs,
            "models/ensembles/calibrator.joblib",
        ),
    ]:
        production[model_name] = {
            "status": "scored",
            "source": source,
            "coverage": coverage,
            "metrics": evaluate_probabilities(y_test, probabilities),
            "test_slices": _evaluate_slice_metrics(
                y_test,
                probabilities,
                test_game_ids,
                slice_masks,
            ),
        }

    return production


def build_combined_leaderboard(
    enriched_results: dict[str, Any],
    production_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine enriched experiment rows with production stack rows."""
    rows: list[dict[str, Any]] = []

    for row in enriched_results.get("leaderboard", []):
        rows.append(
            {
                "family": "enriched_benchmark",
                "label": f"{row['feature_set']} / {row['model_name']}",
                "feature_set": row["feature_set"],
                "model_name": row["model_name"],
                "metrics": row["metrics"],
            }
        )

    production_labels = {
        "production_neural_full_fusion": "production neural full fusion",
        "production_ensemble_raw": "production ensemble raw",
        "production_ensemble_calibrated": "production ensemble calibrated",
    }
    for model_name, model_result in production_results.items():
        rows.append(
            {
                "family": "production_stack",
                "label": production_labels[model_name],
                "feature_set": "production_stack",
                "model_name": model_name,
                "metrics": model_result["metrics"],
            }
        )

    return sorted(
        rows,
        key=lambda row: (row["metrics"]["log_loss"], -row["metrics"]["accuracy"]),
    )


def _best_row(rows: list[dict[str, Any]], family: str | None = None) -> dict[str, Any]:
    filtered = [row for row in rows if family is None or row["family"] == family]
    if not filtered:
        raise ValueError("No leaderboard rows available for selection.")
    return filtered[0]


def _best_calibrated_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (row["metrics"]["calibration_error"], row["metrics"]["log_loss"]),
    )


def build_showdown_markdown(results: dict[str, Any]) -> str:
    """Render the production showdown JSON into a compact markdown summary."""
    lines = [
        "# Production Stack Showdown",
        "",
        f"Generated at `{results['generated_at_utc']}`.",
        "",
        "## Combined Leaderboard",
        "",
        "| Rank | Family | Label | Accuracy | Log Loss | Brier | ROC-AUC | ECE |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    for rank, row in enumerate(results["combined_leaderboard"], start=1):
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
            f"- Best production stack entry: `{verdict['best_production']['label']}`",
            f"- Best enriched challenger: `{verdict['best_enriched']['label']}`",
            f"- Best calibrated entry: `{verdict['best_calibrated']['label']}`",
            "- Production minus enriched log-loss gap: "
            f"`{verdict['production_minus_enriched_log_loss']:+.4f}`",
            "",
            "## Production Slice Comparison",
            "",
            "| Model | Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for model_name, model_result in results["production_results"].items():
        for slice_name, metrics in model_result["test_slices"].items():
            roc_auc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
            lines.append(
                "| "
                f"{model_name} | {slice_name} | {metrics['game_count']} | "
                f"{metrics['accuracy']:.4f} | {metrics['log_loss']:.4f} | "
                f"{metrics['brier_score']:.4f} | {roc_auc} | "
                f"{metrics['calibration_error']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The enriched rows are loaded from the saved benchmark report rather than "
            "retrained here.",
            "- The production rows are recomputed from saved neural and ensemble artifacts "
            "so they use the same held-out test split and slice masks.",
            "- Lower log loss is treated as the primary rank metric because the product "
            "goal is calibrated probability quality, not just hard-pick accuracy.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_showdown() -> dict[str, Any]:
    """Compare saved enriched experiment winners against production model artifacts."""
    enriched_results = _load_enriched_results()
    production_results = load_production_stack_predictions()
    combined_leaderboard = build_combined_leaderboard(enriched_results, production_results)

    best_overall = _best_row(combined_leaderboard)
    best_production = _best_row(combined_leaderboard, family="production_stack")
    best_enriched = _best_row(combined_leaderboard, family="enriched_benchmark")
    best_calibrated = _best_calibrated_row(combined_leaderboard)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_path": str(ENRICHED_RESULTS_PATH),
        "production_results": production_results,
        "combined_leaderboard": combined_leaderboard,
        "verdict": {
            "best_overall": best_overall,
            "best_production": best_production,
            "best_enriched": best_enriched,
            "best_calibrated": best_calibrated,
            "production_minus_enriched_log_loss": float(
                best_production["metrics"]["log_loss"] - best_enriched["metrics"]["log_loss"]
            ),
        },
    }


def main() -> None:
    """Run and persist the production-vs-enriched showdown report."""
    setup_logging()
    SHOWDOWN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = run_showdown()
    SHOWDOWN_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    SHOWDOWN_SUMMARY_PATH.write_text(build_showdown_markdown(results), encoding="utf-8")

    best_overall = results["verdict"]["best_overall"]
    logger.info(
        "Showdown winner: %s with log_loss=%.4f accuracy=%.4f",
        best_overall["label"],
        best_overall["metrics"]["log_loss"],
        best_overall["metrics"]["accuracy"],
    )
    logger.info("Saved showdown report to %s", SHOWDOWN_SUMMARY_PATH)


if __name__ == "__main__":
    main()
