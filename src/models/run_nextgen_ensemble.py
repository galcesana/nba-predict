"""Train a next-generation ensemble with enriched player/lineup model inputs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.calibrate import PlattCalibrator
from src.models.ensemble import load_model_predictions
from src.models.run_enriched_experiments import (
    get_experiment_feature_sets,
    train_catboost,
    train_lightgbm,
)
from src.models.run_production_showdown import (
    _evaluate_slice_metrics as evaluate_slice_metrics,
)
from src.models.run_production_showdown import (
    evaluate_probabilities,
    load_production_stack_predictions,
)
from src.models.tabular_model import split_by_season
from src.utils.logging import setup_logging
from src.utils.paths import DOCS_DIR, MODELS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

NEXTGEN_DIR = MODELS_DIR / "ensembles_nextgen"
NEXTGEN_RESULTS_PATH = DOCS_DIR / "experiments" / "nextgen_ensemble_results.json"
NEXTGEN_SUMMARY_PATH = DOCS_DIR / "experiments" / "nextgen_ensemble_results.md"

PRODUCTION_INPUT_COLS = ["neural_prob", "xgboost_prob", "elo_prob"]
ENRICHED_INPUT_CONFIG_VERSION = "value_tuned_inputs_v1"
ENRICHED_INPUT_MODEL_CONFIGS = {
    "catboost": {
        "prob_col": "enriched_catboost_prob",
        "feature_set": "enriched_value_only",
    },
    "lightgbm": {
        "prob_col": "enriched_lightgbm_prob",
        "feature_set": "enriched_all",
    },
}
ENRICHED_INPUT_COLS = [
    config["prob_col"] for config in ENRICHED_INPUT_MODEL_CONFIGS.values()
]
ENRICHED_FEATURE_COLUMNS_PATH = NEXTGEN_DIR / "enriched_feature_columns.json"


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


def load_legacy_matchup_dataset() -> pd.DataFrame:
    """Load the legacy matchup table used to derive M1 ablation feature sets."""
    path = PROCESSED_DIR / "matchup_rows" / "matchup_dataset.parquet"
    if not path.exists():
        msg = (
            f"Missing legacy matchup dataset at {path}. "
            "Run `python -m src.features.build_matchup_dataset` first."
        )
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def build_enriched_input_feature_config(
    enriched_df: pd.DataFrame,
    *,
    legacy_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Build model-specific enriched input feature selections.

    The next-gen meta-model consumes probabilities, not raw columns. Each probability
    producer can therefore use the feature family that scored best for that learner.
    """
    legacy = legacy_df if legacy_df is not None else load_legacy_matchup_dataset()
    feature_sets = get_experiment_feature_sets(legacy, enriched_df)
    feature_config: dict[str, dict[str, Any]] = {}
    for model_name, model_config in ENRICHED_INPUT_MODEL_CONFIGS.items():
        feature_set_name = model_config["feature_set"]
        if feature_set_name not in feature_sets:
            msg = f"Unknown enriched feature set for {model_name}: {feature_set_name}"
            raise KeyError(msg)
        _, feature_cols = feature_sets[feature_set_name]
        feature_config[model_name] = {
            "feature_set": feature_set_name,
            "prob_col": model_config["prob_col"],
            "feature_cols": feature_cols,
        }
    return feature_config


def _feature_config_payload(feature_config: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Render model-specific feature columns as a stable artifact payload."""
    return {
        "schema_version": ENRICHED_INPUT_CONFIG_VERSION,
        "models": {
            model_name: {
                "feature_set": config["feature_set"],
                "prob_col": config["prob_col"],
                "feature_count": len(config["feature_cols"]),
                "columns": config["feature_cols"],
            }
            for model_name, config in feature_config.items()
        },
    }


def _feature_column_cache_matches(feature_config: dict[str, dict[str, Any]]) -> bool:
    """Return whether saved feature-column metadata matches the current input config."""
    if not ENRICHED_FEATURE_COLUMNS_PATH.exists():
        logger.info("Cached enriched feature-column metadata is missing.")
        return False
    try:
        payload = json.loads(ENRICHED_FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.info("Cached enriched feature-column metadata is not valid JSON.")
        return False
    if not isinstance(payload, dict):
        logger.info("Cached enriched feature-column metadata uses the legacy shared schema.")
        return False
    if payload.get("schema_version") != ENRICHED_INPUT_CONFIG_VERSION:
        logger.info(
            "Cached enriched feature-column metadata is stale: expected %s, found %s.",
            ENRICHED_INPUT_CONFIG_VERSION,
            payload.get("schema_version"),
        )
        return False

    cached_models = payload.get("models", {})
    for model_name, config in feature_config.items():
        cached = cached_models.get(model_name)
        if not isinstance(cached, dict):
            logger.info("Cached enriched metadata is missing model config for %s.", model_name)
            return False
        if cached.get("feature_set") != config["feature_set"]:
            logger.info("Cached %s feature set is stale.", model_name)
            return False
        if cached.get("prob_col") != config["prob_col"]:
            logger.info("Cached %s probability column is stale.", model_name)
            return False
        if cached.get("columns") != config["feature_cols"]:
            logger.info("Cached %s feature columns are stale.", model_name)
            return False
    return True


def _write_feature_column_metadata(feature_config: dict[str, dict[str, Any]]) -> None:
    ENRICHED_FEATURE_COLUMNS_PATH.write_text(
        json.dumps(_feature_config_payload(feature_config), indent=2),
        encoding="utf-8",
    )


def train_enriched_input_models(
    enriched_df: pd.DataFrame,
    *,
    feature_config: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train enriched CatBoost/LightGBM inputs and return validation/test probabilities."""
    feature_config = feature_config or build_enriched_input_feature_config(enriched_df)
    train_df, val_df, test_df = split_by_season(enriched_df)

    y_train = train_df["target_home_win"].to_numpy()
    y_val = val_df["target_home_win"].to_numpy()

    catboost_cols = feature_config["catboost"]["feature_cols"]
    lightgbm_cols = feature_config["lightgbm"]["feature_cols"]
    X_train_catboost = train_df[catboost_cols].fillna(0)
    X_val_catboost = val_df[catboost_cols].fillna(0)
    X_test_catboost = test_df[catboost_cols].fillna(0)
    X_train_lightgbm = train_df[lightgbm_cols].fillna(0)
    X_val_lightgbm = val_df[lightgbm_cols].fillna(0)
    X_test_lightgbm = test_df[lightgbm_cols].fillna(0)

    logger.info(
        "Training enriched CatBoost ensemble input on %s (%d features)...",
        feature_config["catboost"]["feature_set"],
        len(catboost_cols),
    )
    catboost = train_catboost(
        X_train_catboost.to_numpy(),
        y_train,
        X_val_catboost.to_numpy(),
        y_val,
    )
    logger.info(
        "Training enriched LightGBM ensemble input on %s (%d features)...",
        feature_config["lightgbm"]["feature_set"],
        len(lightgbm_cols),
    )
    lightgbm = train_lightgbm(X_train_lightgbm, y_train, X_val_lightgbm, y_val)

    NEXTGEN_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(catboost, NEXTGEN_DIR / "enriched_catboost.joblib")
    joblib.dump(lightgbm, NEXTGEN_DIR / "enriched_lightgbm.joblib")
    _write_feature_column_metadata(feature_config)

    val_probs = pd.DataFrame(
        {
            "game_id": val_df["game_id"].astype(str).to_numpy(),
            "actual_home_win": y_val,
            "enriched_catboost_prob": catboost.predict_proba(
                X_val_catboost.to_numpy()
            )[:, 1],
            "enriched_lightgbm_prob": lightgbm.predict_proba(X_val_lightgbm)[:, 1],
        }
    )
    test_probs = pd.DataFrame(
        {
            "game_id": test_df["game_id"].astype(str).to_numpy(),
            "actual_home_win": test_df["target_home_win"].to_numpy(),
            "enriched_catboost_prob": catboost.predict_proba(
                X_test_catboost.to_numpy()
            )[:, 1],
            "enriched_lightgbm_prob": lightgbm.predict_proba(X_test_lightgbm)[:, 1],
        }
    )
    val_probs.to_parquet(NEXTGEN_DIR / "enriched_input_predictions_val.parquet", index=False)
    test_probs.to_parquet(NEXTGEN_DIR / "enriched_input_predictions_test.parquet", index=False)
    return val_probs, test_probs


def load_or_train_enriched_input_predictions(
    enriched_df: pd.DataFrame,
    *,
    refresh: bool = False,
    feature_config: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached enriched input probabilities or train them if needed."""
    feature_config = feature_config or build_enriched_input_feature_config(enriched_df)
    val_path = NEXTGEN_DIR / "enriched_input_predictions_val.parquet"
    test_path = NEXTGEN_DIR / "enriched_input_predictions_test.parquet"
    if not refresh and val_path.exists() and test_path.exists():
        val_predictions = pd.read_parquet(val_path)
        test_predictions = pd.read_parquet(test_path)
        _, val_df, test_df = split_by_season(enriched_df)
        if _prediction_cache_matches_split(
            val_predictions,
            val_df,
            label="validation enriched input predictions",
        ) and _prediction_cache_matches_split(
            test_predictions,
            test_df,
            label="test enriched input predictions",
        ) and _feature_column_cache_matches(feature_config):
            logger.info("Loading cached enriched ensemble input predictions from %s", NEXTGEN_DIR)
            return val_predictions, test_predictions
        logger.info("Cached enriched input predictions are stale; retraining input models.")
    return train_enriched_input_models(enriched_df, feature_config=feature_config)


def _prediction_cache_matches_split(
    predictions: pd.DataFrame,
    split_df: pd.DataFrame,
    *,
    label: str,
) -> bool:
    """Return whether cached prediction rows match the current split game ids."""
    if predictions.empty or "game_id" not in predictions.columns:
        logger.info("Cached %s are stale because no game_id rows exist.", label)
        return False
    expected_ids = set(split_df["game_id"].astype(str))
    actual_ids = set(predictions["game_id"].astype(str))
    if actual_ids != expected_ids:
        logger.info(
            "Cached %s are stale: expected %d game ids, found %d.",
            label,
            len(expected_ids),
            len(actual_ids),
        )
        return False
    missing_cols = [col for col in ENRICHED_INPUT_COLS if col not in predictions.columns]
    if missing_cols:
        logger.info("Cached %s are stale; missing columns: %s", label, missing_cols)
        return False
    return True


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
            "## Enriched Input Feature Sets",
            "",
        ]
    )
    enriched_input_models = results.get("enriched_input_config", {}).get("models", {})
    if not enriched_input_models:
        lines.append("- Not recorded for this run.")
    for model_name, config in enriched_input_models.items():
        lines.append(
            f"- `{model_name}` uses `{config['feature_set']}` "
            f"({config['feature_count']} features) -> `{config['prob_col']}`"
        )

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
    feature_config = build_enriched_input_feature_config(enriched_df)
    enriched_val, enriched_test = load_or_train_enriched_input_predictions(
        enriched_df,
        refresh=refresh_enriched_inputs,
        feature_config=feature_config,
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(test_df)),
        "enriched_input_config": _feature_config_payload(feature_config),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-enriched-inputs",
        action="store_true",
        help="Retrain enriched input models even if cached probabilities exist.",
    )
    args = parser.parse_args()
    setup_logging()
    results = run_nextgen_ensemble(refresh_enriched_inputs=args.refresh_enriched_inputs)
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
