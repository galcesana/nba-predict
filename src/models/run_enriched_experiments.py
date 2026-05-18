"""Run round-two enriched matchup experiments across model and feature families."""

from __future__ import annotations

import importlib.util
import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data import fetch_player_logs
from src.features.build_matchup_dataset import (
    ENRICHED_FEATURE_STACK_VERSION,
    build_enriched_matchup_dataset,
    build_matchup_dataset,
)
from src.features.lineup_features import LINEUP_FEATURE_VERSION, build_lineup_features
from src.features.player_value_features import build_player_value_features
from src.features.projected_availability import (
    PROJECTED_AVAILABILITY_VERSION,
    build_projected_availability,
)
from src.models.calibrate import calibrate_predictions
from src.models.evaluate import compute_calibration_error
from src.models.tabular_model import (
    get_feature_columns,
    split_by_season,
    train_logistic_regression,
    train_xgboost,
)
from src.utils.logging import setup_logging
from src.utils.paths import DOCS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = DOCS_DIR / "experiments"
RESULTS_PATH = EXPERIMENTS_DIR / "m1_enriched_matchup_results.json"
SUMMARY_PATH = EXPERIMENTS_DIR / "m1_enriched_matchup_results.md"

MODEL_ORDER = [
    "logistic_regression",
    "xgboost",
    "xgboost_platt",
    "lightgbm",
    "lightgbm_platt",
    "catboost",
    "catboost_platt",
]

OPTIONAL_MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

LINEUP_TOKENS = [
    "expected_starter_continuity",
    "expected_top8_continuity",
    "projected_minutes_concentration",
    "bench_depth_quality",
    "rotation_stability",
    "lineup_familiarity",
    "available_top8_players",
]

AVAILABILITY_TOKENS = [
    "expected_missing_starter_value",
    "expected_missing_rotation_value",
    "projected_top8_availability_mean",
    "projected_top8_confidence_mean",
]

VALUE_TOKENS = [
    "projected_player_value_",
    "projected_top5_value_",
    "projected_top8_value_",
    "projected_available_",
]

SLICE_DESCRIPTIONS = {
    "all_test": "Entire held-out test set.",
    "playoffs": "Games with NBA playoff-style game ids (`004...`).",
    "regular_season": "Games with non-playoff ids.",
    "high_missing_value": "Top quartile of projected missing player value.",
    "low_missing_value": "Bottom quartile of projected missing player value.",
    "high_context_confidence": "Top quartile of projected context confidence.",
    "low_context_confidence": "Bottom quartile of projected context confidence.",
}


def _processed_path(*parts: str) -> pd.io.common.FilePath:
    return PROCESSED_DIR.joinpath(*parts)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_or_build_player_logs() -> pd.DataFrame:
    path = _processed_path("player_game_logs", "player_game_logs.parquet")
    if path.exists():
        logger.info("Loading processed player logs from %s", path)
        return pd.read_parquet(path)

    logger.info("Processed player logs missing; fetching seasons and building them now.")
    seasons = fetch_player_logs._load_seasons_config()
    season_types = fetch_player_logs._load_season_types_config()
    raw_frames = [
        fetch_player_logs.fetch_season_player_logs(season, season_type=season_type)
        for season in seasons
        for season_type in season_types
    ]
    raw_frames = [frame for frame in raw_frames if not frame.empty]
    if not raw_frames:
        msg = "Unable to build player logs; no raw season frames were fetched."
        raise RuntimeError(msg)

    player_logs = fetch_player_logs.build_player_game_logs(
        pd.concat(raw_frames, ignore_index=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    player_logs.to_parquet(path, index=False)
    return player_logs


def _load_if_exists(path: pd.io.common.FilePath) -> pd.DataFrame | None:
    if path.exists():
        logger.info("Loading cached artifact from %s", path)
        return pd.read_parquet(path)
    return None


def _game_id_set(frame: pd.DataFrame) -> set[str]:
    if "game_id" not in frame.columns:
        return set()
    return set(frame["game_id"].astype(str))


def _artifact_matches_game_universe(
    artifact: pd.DataFrame,
    games: pd.DataFrame,
    *,
    label: str,
    exact_game_ids: bool,
    required_column_values: dict[str, str] | None = None,
) -> bool:
    """Return whether a cached artifact matches the current game universe."""
    if artifact.empty or "game_id" not in artifact.columns:
        logger.info("Cached %s is stale because it has no game_id rows.", label)
        return False

    if required_column_values:
        for column, expected_value in required_column_values.items():
            if column not in artifact.columns:
                logger.info(
                    "Cached %s is stale because it is missing %s.",
                    label,
                    column,
                )
                return False
            actual_values = set(artifact[column].dropna().astype(str))
            if actual_values != {expected_value}:
                logger.info(
                    "Cached %s is stale because %s has values %s, expected %s.",
                    label,
                    column,
                    sorted(actual_values),
                    expected_value,
                )
                return False

    expected_ids = _game_id_set(games)
    actual_ids = _game_id_set(artifact)
    if exact_game_ids and actual_ids != expected_ids:
        logger.info(
            "Cached %s is stale: expected %d game ids, found %d.",
            label,
            len(expected_ids),
            len(actual_ids),
        )
        return False

    if "season_type" not in games.columns:
        return True

    game_types = games[["game_id", "season_type"]].copy()
    game_types["game_id"] = game_types["game_id"].astype(str)
    expected_types = set(game_types["season_type"].dropna().astype(str))
    actual_types = set(
        game_types[game_types["game_id"].isin(actual_ids)]["season_type"]
        .dropna()
        .astype(str)
    )
    missing_types = expected_types - actual_types
    if missing_types:
        logger.info(
            "Cached %s is stale because it has no rows for season types: %s",
            label,
            sorted(missing_types),
        )
        return False

    return True


def _load_current_artifact(
    path: pd.io.common.FilePath,
    games: pd.DataFrame,
    *,
    label: str,
    exact_game_ids: bool = False,
    required_column_values: dict[str, str] | None = None,
) -> pd.DataFrame | None:
    artifact = _load_if_exists(path)
    if artifact is None:
        return None
    if _artifact_matches_game_universe(
        artifact,
        games,
        label=label,
        exact_game_ids=exact_game_ids,
        required_column_values=required_column_values,
    ):
        return artifact
    return None


def ensure_experiment_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load or build the legacy and enriched matchup datasets used in experiments."""
    games = pd.read_parquet(_processed_path("games.parquet"))
    team_logs = pd.read_parquet(_processed_path("team_game_logs", "team_game_logs.parquet"))
    legacy_path = _processed_path("matchup_rows", "matchup_dataset.parquet")
    enriched_path = _processed_path("matchup_rows", "matchup_dataset_enriched.parquet")

    legacy_df = _load_current_artifact(
        legacy_path,
        games,
        label="legacy matchup dataset",
        exact_game_ids=True,
    )
    if legacy_df is None:
        logger.info("Legacy matchup dataset missing; rebuilding it.")
        legacy_df = build_matchup_dataset(games, team_logs)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_df.to_parquet(legacy_path, index=False)

    enriched_df = _load_current_artifact(
        enriched_path,
        games,
        label="enriched matchup dataset",
        exact_game_ids=True,
        required_column_values={
            "enriched_feature_stack_version": ENRICHED_FEATURE_STACK_VERSION,
        },
    )
    if enriched_df is not None:
        return legacy_df, enriched_df

    logger.info("Enriched matchup dataset missing; building full M1 feature stack.")
    player_logs = _load_or_build_player_logs()

    player_value_path = _processed_path("player_value_features", "player_value_features.parquet")
    player_value_path.parent.mkdir(parents=True, exist_ok=True)
    projected_path = _processed_path("projected_availability", "projected_availability.parquet")
    projected_path.parent.mkdir(parents=True, exist_ok=True)
    lineup_path = _processed_path("lineup_features", "lineup_features.parquet")
    lineup_path.parent.mkdir(parents=True, exist_ok=True)
    unresolved_path = _processed_path(
        "projected_availability",
        "unresolved_injury_entities.parquet",
    )

    stage_start = perf_counter()
    player_value_features = _load_current_artifact(
        player_value_path,
        games,
        label="player value features",
    )
    player_value_rebuilt = False
    if player_value_features is None:
        logger.info("Building player value features...")
        player_value_features = build_player_value_features(games, player_logs)
        player_value_features.to_parquet(player_value_path, index=False)
        player_value_rebuilt = True
        logger.info(
            "Saved player value features rows=%d in %.1fs",
            len(player_value_features),
            perf_counter() - stage_start,
        )
    else:
        logger.info(
            "Loaded cached player value features rows=%d in %.1fs",
            len(player_value_features),
            perf_counter() - stage_start,
        )

    stage_start = perf_counter()
    projected_availability = None
    if not player_value_rebuilt:
        projected_availability = _load_current_artifact(
            projected_path,
            games,
            label="projected availability",
            required_column_values={
                "availability_model_version": PROJECTED_AVAILABILITY_VERSION,
            },
        )
    unresolved = _load_if_exists(unresolved_path) if projected_availability is not None else None
    projected_rebuilt = False
    if projected_availability is None:
        logger.info("Building projected availability...")
        projected_availability, unresolved = build_projected_availability(
            games,
            player_logs,
            player_value_features=player_value_features,
        )
        projected_availability.to_parquet(projected_path, index=False)
        projected_rebuilt = True
        if unresolved is not None:
            unresolved.to_parquet(unresolved_path, index=False)
        logger.info(
            "Saved projected availability rows=%d unresolved=%d in %.1fs",
            len(projected_availability),
            0 if unresolved is None else len(unresolved),
            perf_counter() - stage_start,
        )
    else:
        logger.info(
            "Loaded cached projected availability rows=%d in %.1fs",
            len(projected_availability),
            perf_counter() - stage_start,
        )

    stage_start = perf_counter()
    lineup_features_df = None
    if not projected_rebuilt:
        lineup_features_df = _load_current_artifact(
            lineup_path,
            games,
            label="lineup features",
            required_column_values={
                "lineup_feature_stack_version": LINEUP_FEATURE_VERSION,
            },
        )
    if lineup_features_df is None:
        logger.info("Building lineup features...")
        lineup_features_df = build_lineup_features(games, player_logs, projected_availability)
        lineup_features_df.to_parquet(lineup_path, index=False)
        logger.info(
            "Saved lineup features rows=%d in %.1fs",
            len(lineup_features_df),
            perf_counter() - stage_start,
        )
    else:
        logger.info(
            "Loaded cached lineup features rows=%d in %.1fs",
            len(lineup_features_df),
            perf_counter() - stage_start,
        )

    stage_start = perf_counter()
    logger.info("Building enriched matchup dataset...")
    enriched_df = build_enriched_matchup_dataset(
        games,
        team_logs,
        player_logs,
        player_value_features=player_value_features,
        projected_availability=projected_availability,
        lineup_features_df=lineup_features_df,
    )
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(enriched_path, index=False)
    logger.info(
        "Saved enriched matchup dataset rows=%d cols=%d in %.1fs",
        len(enriched_df),
        len(enriched_df.columns),
        perf_counter() - stage_start,
    )
    return legacy_df, enriched_df


def _merge_feature_lists(base_features: list[str], extra_features: list[str]) -> list[str]:
    seen = set(base_features)
    merged = list(base_features)
    for feature in extra_features:
        if feature not in seen:
            merged.append(feature)
            seen.add(feature)
    return merged


def _select_token_features(feature_cols: list[str], tokens: list[str]) -> list[str]:
    return [col for col in feature_cols if any(token in col for token in tokens)]


def get_experiment_feature_sets(
    legacy_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Return the round-two feature-set variants used in enriched experiments."""
    legacy_features = get_feature_columns(legacy_df)
    enriched_features = get_feature_columns(enriched_df)
    legacy_feature_set = set(legacy_features)
    m1_only_features = [col for col in enriched_features if col not in legacy_feature_set]
    lineup_features = _select_token_features(m1_only_features, LINEUP_TOKENS)
    availability_features = _select_token_features(m1_only_features, AVAILABILITY_TOKENS)
    value_features = _select_token_features(m1_only_features, VALUE_TOKENS)
    confidence_features = [
        col for col in m1_only_features if col.endswith("projected_top8_confidence_mean")
    ]
    enriched_no_confidence = [
        col for col in enriched_features if col not in set(confidence_features)
    ]

    return {
        "legacy": (legacy_df, legacy_features),
        "enriched_all": (enriched_df, enriched_features),
        "enriched_no_confidence": (enriched_df, enriched_no_confidence),
        "enriched_value_only": (
            enriched_df,
            _merge_feature_lists(legacy_features, value_features),
        ),
        "enriched_availability_only": (
            enriched_df,
            _merge_feature_lists(legacy_features, availability_features),
        ),
        "enriched_lineup_only": (
            enriched_df,
            _merge_feature_lists(legacy_features, lineup_features),
        ),
        "m1_only": (enriched_df, m1_only_features),
    }


def build_test_slice_masks(enriched_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Build boolean game-id masks for the round-two slice evaluation."""
    _, _, test_df = split_by_season(enriched_df)
    game_ids = test_df["game_id"].astype(str).reset_index(drop=True)
    slice_masks: dict[str, pd.Series] = {
        "all_test": pd.Series(True, index=game_ids),
    }

    if "season_type" in test_df.columns:
        is_playoffs = test_df["season_type"].astype(str).str.lower().eq("playoffs")
        is_playoffs = is_playoffs.reset_index(drop=True)
    else:
        is_playoffs = game_ids.str.startswith("004")
    if 0 < int(is_playoffs.sum()) < len(game_ids):
        slice_masks["playoffs"] = pd.Series(is_playoffs.to_numpy(), index=game_ids)
    if 0 < int((~is_playoffs).sum()) < len(game_ids):
        slice_masks["regular_season"] = pd.Series((~is_playoffs).to_numpy(), index=game_ids)

    total_missing_value = (
        test_df[
            [
                "home_projected_player_value_missing",
                "away_projected_player_value_missing",
            ]
        ]
        .fillna(0.0)
        .sum(axis=1)
    )
    confidence_mean = (
        test_df[
            [
                "home_projected_top8_confidence_mean",
                "away_projected_top8_confidence_mean",
            ]
        ]
        .fillna(0.0)
        .mean(axis=1)
    )

    high_missing_threshold = float(total_missing_value.quantile(0.75))
    low_missing_threshold = float(total_missing_value.quantile(0.25))
    high_confidence_threshold = float(confidence_mean.quantile(0.75))
    low_confidence_threshold = float(confidence_mean.quantile(0.25))

    if high_missing_threshold > 0:
        high_missing_mask = total_missing_value >= high_missing_threshold
    else:
        high_missing_mask = total_missing_value > 0
    low_missing_mask = total_missing_value <= low_missing_threshold
    high_confidence_mask = confidence_mean >= high_confidence_threshold
    low_confidence_mask = confidence_mean <= low_confidence_threshold

    for slice_name, mask in {
        "high_missing_value": high_missing_mask,
        "low_missing_value": low_missing_mask,
        "high_context_confidence": high_confidence_mask,
        "low_context_confidence": low_confidence_mask,
    }.items():
        selected_count = int(mask.sum())
        if 0 < selected_count < len(test_df):
            slice_masks[slice_name] = pd.Series(mask.to_numpy(), index=game_ids)

    return slice_masks


def _evaluate_model_probabilities(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted_class = (probabilities >= 0.5).astype(int)
    calibration_error, _ = compute_calibration_error(y_true, probabilities)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predicted_class)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "calibration_error": float(calibration_error),
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
        if not bool(aligned_mask.any()):
            continue

        metrics = _evaluate_model_probabilities(y_true[aligned_mask], probabilities[aligned_mask])
        slice_results[slice_name] = {
            "game_count": int(aligned_mask.sum()),
            "positive_rate": float(y_true[aligned_mask].mean()),
            **metrics,
        }

    return slice_results


def _top_linear_coefficients(
    feature_cols: list[str],
    coefficients: np.ndarray,
    *,
    limit: int = 15,
) -> list[dict[str, float]]:
    ranked = sorted(
        zip(feature_cols, coefficients, strict=False),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]
    return [
        {"feature": feature, "coefficient": float(weight)}
        for feature, weight in ranked
    ]


def _top_feature_importances(
    feature_cols: list[str],
    importances: np.ndarray,
    *,
    limit: int = 15,
) -> list[dict[str, float]]:
    ranked = sorted(
        zip(feature_cols, importances, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return [
        {"feature": feature, "importance": float(weight)}
        for feature, weight in ranked
    ]


def _trained_model_result(
    *,
    validation_probs: np.ndarray,
    validation_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_game_ids: pd.Series,
    slice_masks: dict[str, pd.Series],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "trained",
        "validation": _evaluate_model_probabilities(validation_labels, validation_probs),
        "test": _evaluate_model_probabilities(test_labels, test_probs),
        "test_slices": _evaluate_slice_metrics(
            test_labels,
            test_probs,
            test_game_ids,
            slice_masks,
        ),
    }
    if extra:
        result.update(extra)
    return result


def _calibrated_variant_result(
    *,
    source_model: str,
    validation_probs: np.ndarray,
    validation_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_game_ids: pd.Series,
    slice_masks: dict[str, pd.Series],
    method: str = "platt",
) -> dict[str, Any]:
    calibrator = calibrate_predictions(validation_probs, validation_labels, method=method)
    validation_calibrated = calibrator.predict_proba(validation_probs)
    test_calibrated = calibrator.predict_proba(test_probs)
    return _trained_model_result(
        validation_probs=validation_calibrated,
        validation_labels=validation_labels,
        test_probs=test_calibrated,
        test_labels=test_labels,
        test_game_ids=test_game_ids,
        slice_masks=slice_masks,
        extra={
            "source_model": source_model,
            "calibration_method": method,
        },
    )


def _unavailable_model_result(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    """Train LightGBM with early stopping on validation data."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    logger.info(
        "LightGBM trained: %d trees (best iteration: %d)",
        model.n_estimators,
        int(model.best_iteration_),
    )
    return model


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    """Train CatBoost with validation-set early stopping."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    logger.info(
        "CatBoost trained: %d iterations (best iteration: %d)",
        model.get_param("iterations"),
        int(model.get_best_iteration()),
    )
    return model


def run_feature_set_experiment(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    slice_masks: dict[str, pd.Series],
) -> dict[str, Any]:
    """Train and evaluate round-two models on a single feature-set variant."""
    if not feature_cols:
        msg = f"Feature set '{name}' is empty and cannot be trained."
        raise ValueError(msg)

    train_df, val_df, test_df = split_by_season(df)
    X_train_frame = train_df[feature_cols].fillna(0)
    X_val_frame = val_df[feature_cols].fillna(0)
    X_test_frame = test_df[feature_cols].fillna(0)

    X_train = X_train_frame.to_numpy()
    y_train = train_df["target_home_win"].to_numpy()
    X_val = X_val_frame.to_numpy()
    y_val = val_df["target_home_win"].to_numpy()
    X_test = X_test_frame.to_numpy()
    y_test = test_df["target_home_win"].to_numpy()
    test_game_ids = test_df["game_id"].astype(str).reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(X_train)

    results: dict[str, Any] = {
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }

    logistic = train_logistic_regression(X_train, y_train, scaler)
    val_lr = logistic.predict_proba(scaler.transform(X_val))[:, 1]
    test_lr = logistic.predict_proba(scaler.transform(X_test))[:, 1]
    results["logistic_regression"] = _trained_model_result(
        validation_probs=val_lr,
        validation_labels=y_val,
        test_probs=test_lr,
        test_labels=y_test,
        test_game_ids=test_game_ids,
        slice_masks=slice_masks,
        extra={
            "top_coefficients": _top_linear_coefficients(
                feature_cols,
                logistic.coef_[0],
            )
        },
    )

    xgboost = train_xgboost(X_train, y_train, X_val, y_val)
    val_xgb = xgboost.predict_proba(X_val)[:, 1]
    test_xgb = xgboost.predict_proba(X_test)[:, 1]
    results["xgboost"] = _trained_model_result(
        validation_probs=val_xgb,
        validation_labels=y_val,
        test_probs=test_xgb,
        test_labels=y_test,
        test_game_ids=test_game_ids,
        slice_masks=slice_masks,
        extra={
            "best_iteration": int(xgboost.best_iteration),
            "top_importances": _top_feature_importances(
                feature_cols,
                xgboost.feature_importances_,
            ),
        },
    )
    results["xgboost_platt"] = _calibrated_variant_result(
        source_model="xgboost",
        validation_probs=val_xgb,
        validation_labels=y_val,
        test_probs=test_xgb,
        test_labels=y_test,
        test_game_ids=test_game_ids,
        slice_masks=slice_masks,
    )

    if _module_available("lightgbm"):
        lightgbm = train_lightgbm(X_train_frame, y_train, X_val_frame, y_val)
        val_lgbm = lightgbm.predict_proba(X_val_frame)[:, 1]
        test_lgbm = lightgbm.predict_proba(X_test_frame)[:, 1]
        results["lightgbm"] = _trained_model_result(
            validation_probs=val_lgbm,
            validation_labels=y_val,
            test_probs=test_lgbm,
            test_labels=y_test,
            test_game_ids=test_game_ids,
            slice_masks=slice_masks,
            extra={
                "best_iteration": int(lightgbm.best_iteration_),
                "top_importances": _top_feature_importances(
                    feature_cols,
                    lightgbm.feature_importances_,
                ),
            },
        )
        results["lightgbm_platt"] = _calibrated_variant_result(
            source_model="lightgbm",
            validation_probs=val_lgbm,
            validation_labels=y_val,
            test_probs=test_lgbm,
            test_labels=y_test,
            test_game_ids=test_game_ids,
            slice_masks=slice_masks,
        )
    else:
        results["lightgbm"] = _unavailable_model_result("lightgbm is not installed")

    if _module_available("catboost"):
        catboost = train_catboost(X_train, y_train, X_val, y_val)
        val_cat = catboost.predict_proba(X_val)[:, 1]
        test_cat = catboost.predict_proba(X_test)[:, 1]
        results["catboost"] = _trained_model_result(
            validation_probs=val_cat,
            validation_labels=y_val,
            test_probs=test_cat,
            test_labels=y_test,
            test_game_ids=test_game_ids,
            slice_masks=slice_masks,
            extra={
                "best_iteration": int(catboost.get_best_iteration()),
                "top_importances": _top_feature_importances(
                    feature_cols,
                    np.asarray(catboost.get_feature_importance()),
                ),
            },
        )
        results["catboost_platt"] = _calibrated_variant_result(
            source_model="catboost",
            validation_probs=val_cat,
            validation_labels=y_val,
            test_probs=test_cat,
            test_labels=y_test,
            test_game_ids=test_game_ids,
            slice_masks=slice_masks,
        )
    else:
        results["catboost"] = _unavailable_model_result("catboost is not installed")

    return results


def build_leaderboard_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten experiment results into sortable leaderboard rows."""
    rows: list[dict[str, Any]] = []
    for feature_set_name, feature_results in results["feature_sets"].items():
        for model_name in MODEL_ORDER:
            model_result = feature_results.get(model_name)
            if not model_result or model_result.get("status") != "trained":
                continue
            metrics = model_result["test"]
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    "feature_count": feature_results["feature_count"],
                    "metrics": metrics,
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["metrics"]["log_loss"], -row["metrics"]["accuracy"]),
    )


def build_summary_markdown(results: dict[str, Any]) -> str:
    """Render the experiment JSON into a tracked markdown summary."""
    leaderboard = results["leaderboard"]
    unavailable_rows: list[str] = []

    lines = [
        "# M1 Enriched Matchup Experiments",
        "",
        f"Generated at `{results['generated_at_utc']}`.",
        "",
        "## Overview",
        "",
    ]

    for feature_set_name, feature_results in results["feature_sets"].items():
        lines.append(
            f"- `{feature_set_name}`: `{feature_results['feature_count']}` features"
        )

    lines.extend(
        [
            "",
            "## Test Leaderboard",
            "",
            "| Rank | Feature Set | Model | Accuracy | Log Loss | Brier | ROC-AUC | ECE |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for rank, row in enumerate(leaderboard, start=1):
        metrics = row["metrics"]
        roc_auc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
        lines.append(
            "| "
            f"{rank} | {row['feature_set']} | {row['model_name']} | "
            f"{metrics['accuracy']:.4f} | {metrics['log_loss']:.4f} | "
            f"{metrics['brier_score']:.4f} | {roc_auc} | "
            f"{metrics['calibration_error']:.4f} |"
        )

    for feature_set_name, feature_results in results["feature_sets"].items():
        for model_name in ["lightgbm", "catboost"]:
            model_result = feature_results.get(model_name)
            if model_result and model_result.get("status") == "unavailable":
                unavailable_rows.append(
                    f"- `{feature_set_name}` / `{model_name}`: {model_result['reason']}"
                )

    if unavailable_rows:
        lines.extend(["", "## Unavailable Models", "", *unavailable_rows])

    best_model = results["best_test_model"]
    lines.extend(
        [
            "",
            "## Best Test Model",
            "",
            f"- Feature set: `{best_model['feature_set']}`",
            f"- Model: `{best_model['model_name']}`",
            f"- Accuracy: `{best_model['metrics']['accuracy']:.4f}`",
            f"- Log loss: `{best_model['metrics']['log_loss']:.4f}`",
            f"- ECE: `{best_model['metrics']['calibration_error']:.4f}`",
        ]
    )

    best_slice_results = best_model.get("test_slices", {})
    if best_slice_results:
        lines.extend(
            [
                "",
                "## Best Model Slices",
                "",
                "| Slice | Games | Accuracy | Log Loss | Brier | ROC-AUC | ECE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for slice_name, metrics in best_slice_results.items():
            roc_auc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
            lines.append(
                "| "
                f"{slice_name} | {metrics['game_count']} | {metrics['accuracy']:.4f} | "
                f"{metrics['log_loss']:.4f} | {metrics['brier_score']:.4f} | "
                f"{roc_auc} | {metrics['calibration_error']:.4f} |"
            )
            description = SLICE_DESCRIPTIONS.get(slice_name)
            if description:
                lines.append(f"Slice note: {slice_name} — {description}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `legacy` keeps the original production matchup representation.",
            "- `enriched_all` adds the new player-value, projected availability, and "
            "lineup-aware columns.",
            "- The `enriched_*_only` variants test whether each M1 feature family adds "
            "incremental value on top of the legacy backbone.",
            "- `m1_only` isolates only the new M1 columns to measure standalone signal.",
            "- `*_platt` rows apply Platt scaling using validation predictions before "
            "scoring the test set.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_experiments() -> dict[str, Any]:
    """Run the round-two comparison suite over legacy and enriched feature sets."""
    legacy_df, enriched_df = ensure_experiment_datasets()
    feature_sets = get_experiment_feature_sets(legacy_df, enriched_df)
    slice_masks = build_test_slice_masks(enriched_df)

    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "slice_descriptions": SLICE_DESCRIPTIONS,
        "feature_sets": {},
    }

    for feature_set_name, (frame, feature_cols) in feature_sets.items():
        logger.info(
            "Running experiment '%s' with %d features.",
            feature_set_name,
            len(feature_cols),
        )
        results["feature_sets"][feature_set_name] = run_feature_set_experiment(
            feature_set_name,
            frame,
            feature_cols,
            slice_masks,
        )

    leaderboard = build_leaderboard_rows(results)
    if not leaderboard:
        msg = "No experiment models were successfully trained."
        raise RuntimeError(msg)

    best_row = leaderboard[0]
    best_feature_result = results["feature_sets"][best_row["feature_set"]]
    best_model_result = best_feature_result[best_row["model_name"]]
    results["leaderboard"] = leaderboard
    results["best_test_model"] = {
        "feature_set": best_row["feature_set"],
        "model_name": best_row["model_name"],
        "metrics": best_row["metrics"],
        "test_slices": best_model_result.get("test_slices", {}),
    }
    return results


def main() -> None:
    """Run and persist the enriched matchup round-two experiment suite."""
    setup_logging()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_experiments()
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary_markdown(results), encoding="utf-8")

    best_model = results["best_test_model"]
    logger.info(
        "Best test result: %s / %s with log_loss=%.4f accuracy=%.4f",
        best_model["feature_set"],
        best_model["model_name"],
        best_model["metrics"]["log_loss"],
        best_model["metrics"]["accuracy"],
    )
    logger.info("Saved experiment report to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
