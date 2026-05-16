"""Publish a live weekly forecast snapshot for deployment consumers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from src.app.predict_today import (
    DEFAULT_FORECAST_WINDOW_DAYS,
    generate_predictions_for_window,
)
from src.utils.logging import setup_logging
from src.utils.paths import PUBLISHED_DIR

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "America/New_York"

PredictionGenerator = Callable[..., tuple[dict, Path] | None]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=target.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    shutil.copyfile(source, tmp_path)
    os.replace(tmp_path, target)


def resolve_target_date(
    date_str: str | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    now: datetime | None = None,
) -> str:
    """Resolve the target publish date in the requested timezone."""
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")

    current = now or datetime.now(ZoneInfo(timezone_name))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo(timezone_name))
    else:
        current = current.astimezone(ZoneInfo(timezone_name))
    return current.date().isoformat()


def _published_file_for_date(published_root: Path, date_str: str) -> Path:
    return published_root / "daily" / f"{date_str}.json"


def _repo_relative_path(path: Path, root: Path) -> str | None:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return None


def publish_predictions_for_date(
    date_str: str,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
    published_root: Path = PUBLISHED_DIR,
    prediction_generator: PredictionGenerator = generate_predictions_for_window,
) -> tuple[dict, list[Path]]:
    """Publish a forecast snapshot for the requested date."""
    publish_root = Path(published_root)
    repo_root = publish_root.parent
    manifest_path = publish_root / "manifest.json"
    latest_path = publish_root / "daily" / "latest.json"
    previous_latest = _read_json(latest_path)

    logger.info("Publishing forecast for %s (%s)", date_str, timezone_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_root = Path(tmp_dir)
        result = prediction_generator(date_str, output_dir=temp_root)

        if result is None:
            previous_date = previous_latest.get("date") if previous_latest else None
            previous_file = (
                _repo_relative_path(
                    _published_file_for_date(publish_root, previous_date),
                    repo_root,
                )
                if previous_date
                else None
            )
            manifest = {
                "status": "no_games",
                "target_date": date_str,
                "attempted_at": _utc_now_iso(),
                "latest_available_date": previous_date,
                "published_file": previous_file,
                "games_count": 0,
                "model_version": None,
                "slate_type": "week",
                "window_start": date_str,
                "window_end": date_str,
                "context_summary": {
                    "injury_coverage_rate": 0.0,
                    "news_coverage_rate": 0.0,
                },
            }
            _write_json_atomic(manifest_path, manifest)
            return manifest, [manifest_path]

        payload, temp_output_path = result
        dated_path = _published_file_for_date(publish_root, date_str)

        _copy_file_atomic(temp_output_path, dated_path)
        _copy_file_atomic(temp_output_path, latest_path)

    manifest = {
        "status": "published",
        "target_date": date_str,
        "attempted_at": _utc_now_iso(),
        "latest_available_date": date_str,
        "published_file": _repo_relative_path(dated_path, repo_root),
        "games_count": len(payload.get("predictions", [])),
        "model_version": payload.get("model_version"),
        "slate_type": payload.get("slate_type", "week"),
        "window_start": payload.get("window_start", date_str),
        "window_end": payload.get("window_end", date_str),
        "days_with_games": len(payload.get("dates_with_games", [])),
        "context_summary": payload.get("context_summary", {}),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest, [dated_path, latest_path, manifest_path]


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Publish this week's NBA forecast.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to publish for in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default=DEFAULT_TIMEZONE,
        help="Timezone used when --date is omitted.",
    )
    args = parser.parse_args(argv)

    try:
        target_date = resolve_target_date(args.date, args.timezone)
        manifest, _ = publish_predictions_for_date(
            target_date,
            timezone_name=args.timezone,
        )
        logger.info(
            "Publish finished with status=%s target_date=%s latest_available=%s days=%s",
            manifest["status"],
            manifest["target_date"],
            manifest["latest_available_date"],
            DEFAULT_FORECAST_WINDOW_DAYS,
        )
        return 0
    except Exception:
        logger.exception("Publish failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
