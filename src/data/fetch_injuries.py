"""Fetch and parse official NBA injury report PDFs for live forecasting."""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import fitz
import pandas as pd
import requests

from src.data.team_metadata import load_team_metadata_by_idx, load_team_name_lookup
from src.utils.logging import setup_logging
from src.utils.paths import RAW_DIR

logger = logging.getLogger(__name__)

DEFAULT_TIMEZONE = "America/New_York"
LATEST_REPORT_HOUR = 16
LATEST_REPORT_MINUTE = 30
REPORT_INTERVAL_MINUTES = 15
REPORT_URL_TEMPLATE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-predict/1.0)"}
_MATCHUP_PATTERN = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
_DATE_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_TIME_PATTERN = re.compile(r"^\d{2}:\d{2} \(ET\)$")
_PLAYER_PATTERN = re.compile(r"^[A-Za-z'. -]+,\s*[A-Za-z'. -]+$")
_TEAM_NAME_LOOKUP = load_team_name_lookup()


def _report_time_slug(moment: datetime) -> str:
    return moment.strftime("%I_%M%p")


def _candidate_report_times(
    report_date: str,
    *,
    as_of: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> list[datetime]:
    tz = ZoneInfo(timezone_name)
    report_day = datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=tz)
    current = as_of or datetime.now(tz)
    current = current.astimezone(tz)

    latest_possible = report_day.replace(
        hour=LATEST_REPORT_HOUR,
        minute=LATEST_REPORT_MINUTE,
        second=0,
        microsecond=0,
    )
    if current.date() == report_day.date():
        latest_possible = min(current.replace(second=0, microsecond=0), latest_possible)

    minutes = (latest_possible.minute // REPORT_INTERVAL_MINUTES) * REPORT_INTERVAL_MINUTES
    cursor = latest_possible.replace(minute=minutes, second=0, microsecond=0)
    if cursor < report_day:
        return []

    candidates: list[datetime] = []
    while cursor >= report_day:
        candidates.append(cursor)
        cursor -= timedelta(minutes=REPORT_INTERVAL_MINUTES)
    return candidates


def _report_url(report_time: datetime) -> str:
    return REPORT_URL_TEMPLATE.format(
        date=report_time.strftime("%Y-%m-%d"),
        time=_report_time_slug(report_time),
    )


def fetch_latest_injury_report_pdf(
    report_date: str,
    *,
    as_of: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    session: requests.Session | None = None,
) -> tuple[bytes, str, str] | None:
    """Return the latest available official report PDF for the given date."""
    client = session or requests.Session()
    for report_time in _candidate_report_times(
        report_date, as_of=as_of, timezone_name=timezone_name
    ):
        url = _report_url(report_time)
        try:
            response = client.get(url, headers=REQUEST_HEADERS, timeout=12)
            if response.status_code == 200 and response.headers.get("content-type", "").startswith(
                "application/pdf"
            ):
                generated_at = report_time.strftime("%Y-%m-%dT%H:%M:%S%z")
                return response.content, url, generated_at
        except Exception as exc:
            logger.debug("Skipping injury report candidate %s: %s", url, exc)
    return None


def _normalize_lines(pdf_bytes: bytes) -> list[str]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in document)
    lines = [line.strip() for line in text.splitlines()]
    filtered = []
    for line in lines:
        if not line:
            continue
        if line.startswith("Injury Report:") or line.startswith("Page "):
            continue
        if line in {
            "Game Date",
            "Game Time",
            "Matchup",
            "Team",
            "Player Name",
            "Current Status",
            "Reason",
        }:
            continue
        filtered.append(line)
    return filtered


def _target_game_lookup(games: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    team_metadata = load_team_metadata_by_idx()
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in games.iterrows():
        away_abbr = str(team_metadata[int(row["away_team_idx"])]["abbreviation"])
        home_abbr = str(team_metadata[int(row["home_team_idx"])]["abbreviation"])
        lookup[(pd.Timestamp(row["date"]).strftime("%m/%d/%Y"), f"{away_abbr}@{home_abbr}")] = {
            "game_id": row["game_id"],
            "home_team_idx": int(row["home_team_idx"]),
            "away_team_idx": int(row["away_team_idx"]),
            "target_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        }
    return lookup


def _is_boundary(line: str) -> bool:
    return (
        _DATE_PATTERN.match(line) is not None
        or _TIME_PATTERN.match(line) is not None
        or _MATCHUP_PATTERN.match(line) is not None
        or line == "NOT YET SUBMITTED"
        or line.upper() in _TEAM_NAME_LOOKUP
    )


def parse_injury_report_pdf(
    pdf_bytes: bytes,
    *,
    games: pd.DataFrame,
    source_url: str,
    report_generated_at: str,
) -> pd.DataFrame:
    """Parse one official injury report PDF into normalized team/player rows."""
    lines = _normalize_lines(pdf_bytes)
    targets = _target_game_lookup(games)

    rows: list[dict[str, object]] = []
    current_date: str | None = None
    current_time: str | None = None
    current_matchup: str | None = None
    current_target: dict[str, object] | None = None
    current_team_name: str | None = None
    current_team_idx: int | None = None
    current_team_has_rows = False

    def flush_empty_team() -> None:
        nonlocal current_team_has_rows
        if current_target is None or current_team_idx is None or current_team_has_rows:
            return
        rows.append(
            {
                "game_id": current_target["game_id"],
                "report_game_date": current_date,
                "team_idx": current_team_idx,
                "team_name": current_team_name,
                "player_name": None,
                "status": "CLEAR",
                "reason": None,
                "report_submitted": True,
                "report_time_et": current_time,
                "report_generated_at": report_generated_at,
                "source_url": source_url,
            }
        )
        current_team_has_rows = True

    i = 0
    while i < len(lines):
        line = lines[i]
        upper_line = line.upper()

        if _DATE_PATTERN.match(line):
            flush_empty_team()
            current_date = line
            current_time = None
            current_matchup = None
            current_target = None
            current_team_name = None
            current_team_idx = None
            current_team_has_rows = False
            i += 1
            continue

        if _TIME_PATTERN.match(line):
            flush_empty_team()
            current_time = line
            current_matchup = None
            current_target = None
            current_team_name = None
            current_team_idx = None
            current_team_has_rows = False
            i += 1
            continue

        if _MATCHUP_PATTERN.match(line):
            flush_empty_team()
            current_matchup = line
            current_target = targets.get((current_date or "", current_matchup))
            current_team_name = None
            current_team_idx = None
            current_team_has_rows = False
            i += 1
            continue

        if upper_line in _TEAM_NAME_LOOKUP:
            flush_empty_team()
            current_team_name = line
            current_team_idx = _TEAM_NAME_LOOKUP[upper_line]
            current_team_has_rows = False
            i += 1
            if i < len(lines) and lines[i] == "NOT YET SUBMITTED" and current_target is not None:
                rows.append(
                    {
                        "game_id": current_target["game_id"],
                        "report_game_date": current_date,
                        "team_idx": current_team_idx,
                        "team_name": current_team_name,
                        "player_name": None,
                        "status": "NOT YET SUBMITTED",
                        "reason": None,
                        "report_submitted": False,
                        "report_time_et": current_time,
                        "report_generated_at": report_generated_at,
                        "source_url": source_url,
                    }
                )
                current_team_has_rows = True
                i += 1
            continue

        if current_target is None or current_team_idx is None or not _PLAYER_PATTERN.match(line):
            i += 1
            continue

        player_name = line
        if i + 1 >= len(lines):
            break
        status = lines[i + 1]
        i += 2

        reason_parts: list[str] = []
        while i < len(lines):
            next_line = lines[i]
            next_upper = next_line.upper()
            if _is_boundary(next_line):
                break
            if _PLAYER_PATTERN.match(next_line) and i + 1 < len(lines):
                break
            if next_upper in _TEAM_NAME_LOOKUP:
                break
            reason_parts.append(next_line)
            i += 1

        rows.append(
            {
                "game_id": current_target["game_id"],
                "report_game_date": current_date,
                "team_idx": current_team_idx,
                "team_name": current_team_name,
                "player_name": player_name,
                "status": status,
                "reason": " ".join(reason_parts) or None,
                "report_submitted": True,
                "report_time_et": current_time,
                "report_generated_at": report_generated_at,
                "source_url": source_url,
            }
        )
        current_team_has_rows = True

    flush_empty_team()

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame[frame["game_id"].isin(games["game_id"])].reset_index(drop=True)


def fetch_injury_reports_for_games(
    games: pd.DataFrame,
    *,
    report_date: str | None = None,
    as_of: datetime | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch and cache the latest injury report snapshot relevant to `games`."""
    if games.empty:
        return pd.DataFrame()

    storage_dir = cache_dir or (RAW_DIR / "injuries")
    storage_dir.mkdir(parents=True, exist_ok=True)

    effective_report_date = (
        report_date or datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date().isoformat()
    )
    cache_path = storage_dir / f"official_injury_report_{effective_report_date}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        return cached[cached["game_id"].isin(games["game_id"])].reset_index(drop=True)

    fetched = fetch_latest_injury_report_pdf(
        effective_report_date,
        as_of=as_of,
        timezone_name=DEFAULT_TIMEZONE,
    )
    if fetched is None:
        return pd.DataFrame()

    pdf_bytes, source_url, report_generated_at = fetched
    frame = parse_injury_report_pdf(
        pdf_bytes,
        games=games,
        source_url=source_url,
        report_generated_at=report_generated_at,
    )
    if frame.empty:
        return frame

    frame.to_parquet(cache_path, index=False)
    return frame.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Fetch live official NBA injury reports.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Report date to fetch in YYYY-MM-DD format. Defaults to today's ET date.",
    )
    args = parser.parse_args(argv)

    report_date = args.date or datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).date().isoformat()
    fetched = fetch_latest_injury_report_pdf(report_date)
    if fetched is None:
        logger.warning("No injury report PDF found for %s.", report_date)
        return 1

    _, source_url, generated_at = fetched
    logger.info("Found injury report %s (%s).", source_url, generated_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
