"""Tests for the live publishing layer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.app import publish_today


def _prediction_payload(date_str: str) -> dict:
    return {
        "date": date_str,
        "slate_type": "week",
        "window_start": date_str,
        "window_end": date_str,
        "generated_at": "2026-05-16T12:00:00Z",
        "model_version": "ensemble_v1",
        "dates_with_games": [{"date": date_str, "games_count": 1}],
        "context_summary": {
            "injury_coverage_rate": 0.5,
            "news_coverage_rate": 1.0,
        },
        "predictions": [
            {
                "game_id": "game-1",
                "game_date": date_str,
                "home_team_idx": 0,
                "away_team_idx": 1,
                "home_win_probability": 0.61,
                "away_win_probability": 0.39,
                "predicted_winner": "home",
                "confidence_bucket": "medium",
                "context_details": {"injury_mode": "partial", "news_mode": "live"},
                "top_model_factors": ["recent net rating"],
                "component_outputs": {"ensemble_probability": 0.61},
            }
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_publish_success_writes_dated_latest_and_manifest(tmp_path):
    """A successful publish writes both forecast files and a manifest."""

    def fake_generator(date_str: str, output_dir: Path, **_: object):
        payload = _prediction_payload(date_str)
        out_path = output_dir / f"{date_str}.json"
        _write_json(out_path, payload)
        return payload, out_path

    manifest, paths = publish_today.publish_predictions_for_date(
        "2026-01-15",
        published_root=tmp_path / "published",
        prediction_generator=fake_generator,
    )

    dated_path = tmp_path / "published" / "daily" / "2026-01-15.json"
    latest_path = tmp_path / "published" / "daily" / "latest.json"
    manifest_path = tmp_path / "published" / "manifest.json"

    assert dated_path.exists()
    assert latest_path.exists()
    assert manifest_path.exists()
    assert manifest["status"] == "published"
    assert manifest["published_file"] == "published/daily/2026-01-15.json"
    assert manifest["games_count"] == 1
    assert manifest["window_start"] == "2026-01-15"
    assert manifest["window_end"] == "2026-01-15"
    assert manifest["slate_type"] == "week"
    assert manifest["context_summary"]["injury_coverage_rate"] == 0.5
    assert set(paths) == {dated_path, latest_path, manifest_path}


def test_publish_no_games_preserves_latest_and_writes_manifest(tmp_path):
    """No-games days update only the manifest and keep the previous latest file."""
    publish_root = tmp_path / "published"
    latest_payload = _prediction_payload("2026-01-14")
    _write_json(publish_root / "daily" / "latest.json", latest_payload)
    _write_json(publish_root / "daily" / "2026-01-14.json", latest_payload)

    def no_games_generator(date_str: str, output_dir: Path, **_: object):
        assert date_str == "2026-01-15"
        assert output_dir.exists()
        return None

    manifest, paths = publish_today.publish_predictions_for_date(
        "2026-01-15",
        published_root=publish_root,
        prediction_generator=no_games_generator,
    )

    latest_path = publish_root / "daily" / "latest.json"
    assert json.loads(latest_path.read_text(encoding="utf-8"))["date"] == "2026-01-14"
    assert manifest["status"] == "no_games"
    assert manifest["latest_available_date"] == "2026-01-14"
    assert manifest["published_file"] == "published/daily/2026-01-14.json"
    assert manifest["window_start"] == "2026-01-15"
    assert manifest["window_end"] == "2026-01-15"
    assert paths == [publish_root / "manifest.json"]


def test_resolve_target_date_accepts_explicit_date():
    """Explicit dates bypass timezone-based resolution."""
    assert publish_today.resolve_target_date("2026-02-03") == "2026-02-03"


def test_resolve_target_date_defaults_to_eastern():
    """Implicit publish dates use America/New_York by default."""
    now = datetime(2026, 1, 2, 4, 30, tzinfo=timezone.utc)
    assert publish_today.resolve_target_date(None, now=now) == "2026-01-01"


def test_publish_failure_leaves_existing_files_untouched(tmp_path):
    """Failures should not overwrite the currently published slate."""
    publish_root = tmp_path / "published"
    latest_payload = _prediction_payload("2026-01-14")
    dated_path = publish_root / "daily" / "2026-01-14.json"
    latest_path = publish_root / "daily" / "latest.json"
    _write_json(dated_path, latest_payload)
    _write_json(latest_path, latest_payload)

    def failing_generator(date_str: str, output_dir: Path, **_: object):
        raise RuntimeError(f"boom for {date_str} in {output_dir}")

    with pytest.raises(RuntimeError):
        publish_today.publish_predictions_for_date(
            "2026-01-15",
            published_root=publish_root,
            prediction_generator=failing_generator,
        )

    assert json.loads(dated_path.read_text(encoding="utf-8"))["date"] == "2026-01-14"
    assert json.loads(latest_path.read_text(encoding="utf-8"))["date"] == "2026-01-14"
    assert not (publish_root / "manifest.json").exists()
