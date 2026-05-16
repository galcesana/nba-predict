"""Streamlit dashboard for NBA predictions."""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Iterable

# Streamlit Cloud may run this file as the main module; repo root must be importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_root_str = str(_PROJECT_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.app import dashboard_data as data  # noqa: E402

PAGES = [
    "This Week's Games",
    "Game Detail",
    "Archive",
    "Performance",
    "Calibration",
    "Team Form",
    "Injury Impact",
    "News Sentiment",
]


def _set_page_config() -> None:
    st.set_page_config(
        page_title="NBA Predict Dashboard",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_dashboard_styles() -> None:
    """Apply the dashboard's visual system."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #07131d;
            --panel: rgba(13, 31, 44, 0.88);
            --panel-strong: rgba(17, 40, 56, 0.96);
            --border: rgba(104, 154, 176, 0.24);
            --text: #f5efe2;
            --muted: #91aab6;
            --home: #39c4c6;
            --away: #ff8159;
            --gold: #f0c15d;
            --success: #8bd17c;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(57, 196, 198, 0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(255, 129, 89, 0.12), transparent 26%),
                linear-gradient(180deg, #07131d 0%, #081a28 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: rgba(6, 17, 27, 0.96);
            border-right: 1px solid var(--border);
        }

        .hero-card,
        .metric-card,
        .section-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.18);
        }

        .hero-card {
            padding: 1.5rem 1.5rem 1rem 1.5rem;
            margin-bottom: 1rem;
        }

        .hero-eyebrow {
            color: var(--gold);
            font-size: 0.82rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0;
            color: var(--text);
        }

        .hero-copy {
            color: var(--muted);
            max-width: 60rem;
            margin-top: 0.6rem;
            margin-bottom: 0;
        }

        .metric-card {
            padding: 1rem 1rem 0.85rem 1rem;
            min-height: 124px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.95rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .metric-copy {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        .game-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.1rem 1.15rem;
            margin-bottom: 0.9rem;
        }

        .matchup-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.65rem;
        }

        .matchup-title {
            color: var(--text);
            font-size: 1.18rem;
            font-weight: 700;
            margin: 0;
        }

        .matchup-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.18rem;
        }

        .confidence-badge {
            display: inline-block;
            padding: 0.22rem 0.62rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #07131d;
            background: var(--gold);
        }

        .confidence-badge.low {
            background: #93a8b2;
        }

        .confidence-badge.medium {
            background: #52b7d0;
        }

        .confidence-badge.high {
            background: var(--success);
        }

        .prob-header,
        .prob-footer {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }

        .prob-footer {
            margin-top: 0.32rem;
            margin-bottom: 0;
        }

        .prob-bar {
            display: flex;
            width: 100%;
            height: 14px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .prob-away {
            background: linear-gradient(90deg, rgba(255, 129, 89, 0.88), rgba(255, 164, 116, 0.88));
        }

        .prob-home {
            background: linear-gradient(90deg, rgba(72, 185, 187, 0.9), rgba(57, 196, 198, 0.95));
        }

        .factor-chip {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid rgba(240, 193, 93, 0.24);
            background: rgba(240, 193, 93, 0.08);
            color: var(--text);
            padding: 0.26rem 0.65rem;
            margin: 0.18rem 0.28rem 0 0;
            font-size: 0.84rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="hero-eyebrow">Forecasts and Diagnostics</div>
          <h1 class="hero-title">NBA Predict</h1>
          <p class="hero-copy">
            Weekly win probabilities, model performance, calibration tracking,
            and matchup context in one place.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_row(items: Iterable[tuple[str, str, str]]) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for col, (label, value, copy) in zip(cols, items):
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _probability_bar(home_prob: float, home_team: str, away_team: str) -> str:
    away_prob = max(0.0, 1.0 - home_prob)
    return (
        '<div class="prob-header">'
        f"<span>{html.escape(away_team)}</span>"
        f"<span>{html.escape(home_team)}</span>"
        "</div>"
        '<div class="prob-bar">'
        f'<div class="prob-away" style="width:{away_prob * 100:.1f}%"></div>'
        f'<div class="prob-home" style="width:{home_prob * 100:.1f}%"></div>'
        "</div>"
        '<div class="prob-footer">'
        f"<span>{away_prob * 100:.1f}%</span>"
        f"<span>{home_prob * 100:.1f}%</span>"
        "</div>"
    )


def _forecast_status(payload: dict | None, manifest: dict | None) -> dict[str, str]:
    return data.describe_forecast_status(payload, manifest)


def _slate_caption(payload: dict | None, manifest: dict | None) -> str:
    status = _forecast_status(payload, manifest)
    if payload and payload.get("date"):
        if payload.get("window_start") and payload.get("window_end"):
            return (
                f"{status['message']} "
                f"Forecast window: {payload['window_start']} to {payload['window_end']}"
            )
        return f"{status['message']} Slate date: {payload['date']}"
    return status["message"]


def _slate_source_notice(payload: dict | None, manifest: dict | None) -> None:
    status = _forecast_status(payload, manifest)
    if status["state"] == "local":
        return
    if status["state"] == "published_this_week":
        st.success(status["message"])
    elif status["state"] in {"no_games", "bundled"}:
        st.info(status["message"])
    elif status["state"] == "stale":
        st.warning(status["message"])
    elif status["state"] == "published":
        st.success(status["message"])
    else:
        st.info(status["message"])


def _latest_daily_payload() -> dict | None:
    # Published forecast files are updated outside the Streamlit process,
    # so this loader must read from disk on every rerun.
    return data.load_latest_daily_predictions()


def _publish_manifest() -> dict | None:
    # Keep the publish status banner in sync with the latest tracked manifest.
    return data.load_publish_manifest()


def _archive_frame() -> pd.DataFrame:
    # The archive includes published daily snapshots that can change after deploy.
    return data.build_archive_dataframe()


@st.cache_data(show_spinner=False)
def _model_comparison() -> pd.DataFrame:
    return data.build_model_performance_table()


@st.cache_data(show_spinner=False)
def _rolling_validation() -> pd.DataFrame:
    return data.build_rolling_validation_frame()


@st.cache_data(show_spinner=False)
def _ensemble_weights() -> pd.DataFrame:
    return data.build_ensemble_weights_frame()


@st.cache_data(show_spinner=False)
def _calibration_frame() -> pd.DataFrame:
    return data.build_calibration_frame()


@st.cache_data(show_spinner=False)
def _team_logs() -> pd.DataFrame:
    return data.load_team_logs()


@st.cache_data(show_spinner=False)
def _injury_summary() -> pd.DataFrame:
    return data.build_injury_summary()


@st.cache_data(show_spinner=False)
def _news_summary() -> pd.DataFrame:
    return data.build_news_summary()


def render_today_page() -> None:
    payload = _latest_daily_payload()
    manifest = _publish_manifest()
    st.subheader("This Week's Forecasts")
    if not payload:
        _slate_source_notice(payload, manifest)
        return
    _slate_source_notice(payload, manifest)

    predictions = payload.get("predictions", [])
    dates_with_games = payload.get("dates_with_games", [])
    avg_edge = 0.0
    if predictions:
        avg_edge = sum(
            abs(float(pred["home_win_probability"]) - 0.5) for pred in predictions
        ) / len(predictions)

    _metric_row(
        [
            (
                "Forecast Window",
                f"{payload.get('window_start', payload.get('date', 'Unknown'))} → "
                f"{payload.get('window_end', payload.get('date', 'Unknown'))}",
                "Upcoming live schedule covered by this slate",
            ),
            ("Games This Week", str(len(predictions)), "Matchups in the current forecast window"),
            (
                "Days With Games",
                str(len(dates_with_games)),
                "Scheduled dates included in the slate",
            ),
            ("Average Edge", f"{avg_edge * 100:.1f}%", "Mean distance from a coin flip"),
        ]
    )

    st.caption(
        "Generated at "
        f"{payload.get('generated_at', 'unknown')} | "
        f"Model {payload.get('model_version', 'unknown')}"
    )

    grouped_predictions: dict[str, list[dict]] = {}
    for prediction in predictions:
        grouped_predictions.setdefault(
            prediction.get("game_date", payload.get("date", "Unknown")),
            [],
        ).append(prediction)

    for game_date, day_predictions in grouped_predictions.items():
        pretty_date = pd.to_datetime(game_date).strftime("%A, %b %d")
        st.markdown(f"### {pretty_date}")
        for prediction in day_predictions:
            home_team = data.team_abbr(prediction["home_team_idx"])
            away_team = data.team_abbr(prediction["away_team_idx"])
            badge_class = prediction.get("confidence_bucket", "low")
            factors_html = "".join(
                f'<span class="factor-chip">{html.escape(str(factor))}</span>'
                for factor in prediction.get("top_model_factors", [])
            )
            metadata_bits = [f"Game ID {prediction.get('game_id')}"]
            if prediction.get("game_label"):
                metadata_bits.append(str(prediction["game_label"]))
            if prediction.get("game_sub_label"):
                metadata_bits.append(str(prediction["game_sub_label"]))
            if prediction.get("series_text"):
                metadata_bits.append(str(prediction["series_text"]))
            metadata_line = " | ".join(metadata_bits)
            matchup_title = f"{html.escape(away_team)} at {html.escape(home_team)}"
            probability_html = _probability_bar(
                float(prediction["home_win_probability"]),
                home_team,
                away_team,
            )

            card_html = (
                '<div class="game-card">'
                '<div class="matchup-row">'
                "<div>"
                f'<div class="matchup-title">{matchup_title}</div>'
                f'<div class="matchup-subtitle">{html.escape(metadata_line)}</div>'
                "</div>"
                f'<span class="confidence-badge {badge_class}">'
                f"{html.escape(str(prediction.get('confidence_bucket', 'low')))}"
                "</span>"
                "</div>"
                f"{probability_html}"
                f'<div style="margin-top:0.65rem;">{factors_html}</div>'
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)


def render_game_detail_page() -> None:
    payload = _latest_daily_payload()
    manifest = _publish_manifest()
    st.subheader("Game Detail")
    if not payload or not payload.get("predictions"):
        _slate_source_notice(payload, manifest)
        return
    _slate_source_notice(payload, manifest)

    options = {
        (
            f"{pred.get('game_date', payload.get('date', 'unknown'))} | "
            f"{data.matchup_label(pred['home_team_idx'], pred['away_team_idx'])} "
            f"| {pred['game_id']}"
        ): pred["game_id"]
        for pred in payload["predictions"]
    }
    selected_label = st.selectbox("Select a game", list(options.keys()))
    prediction = data.get_prediction_detail(payload, options[selected_label])
    if not prediction:
        st.warning("Selected game details could not be loaded.")
        return

    home_team = data.team_abbr(prediction["home_team_idx"])
    away_team = data.team_abbr(prediction["away_team_idx"])
    st.markdown(
        _probability_bar(float(prediction["home_win_probability"]), home_team, away_team),
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Predicted Winner", prediction.get("predicted_winner", "unknown").title())
    metric_cols[1].metric("Confidence", prediction.get("confidence_bucket", "unknown").title())
    metric_cols[2].metric("Model Version", payload.get("model_version", "unknown"))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Component Outputs")
        st.dataframe(
            data.build_component_output_frame(prediction),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Top Model Factors")
        for factor in prediction.get("top_model_factors", []):
            st.markdown(f"- {factor}")

    with right:
        st.markdown("#### Recent Team Form")
        tabs = st.tabs([home_team, away_team])
        with tabs[0]:
            home_form = data.build_team_form_frame(int(prediction["home_team_idx"]))
            if home_form.empty:
                st.info("No form data available.")
            else:
                st.line_chart(
                    home_form.set_index("date")[["net_rating", "point_diff"]],
                    width="stretch",
                )
                st.dataframe(
                    home_form[
                        [
                            "date",
                            "opponent",
                            "location",
                            "result",
                            "points_for",
                            "points_against",
                            "net_rating",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
        with tabs[1]:
            away_form = data.build_team_form_frame(int(prediction["away_team_idx"]))
            if away_form.empty:
                st.info("No form data available.")
            else:
                st.line_chart(
                    away_form.set_index("date")[["net_rating", "point_diff"]],
                    width="stretch",
                )
                st.dataframe(
                    away_form[
                        [
                            "date",
                            "opponent",
                            "location",
                            "result",
                            "points_for",
                            "points_against",
                            "net_rating",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )


def render_archive_page() -> None:
    archive = _archive_frame()
    st.subheader("Historical Archive")
    if archive.empty:
        st.info("Historical forecasts and backtests will appear here once they are available.")
        return

    source_options = ["all"] + sorted(archive["source"].dropna().unique().tolist())
    team_options = ["all"] + sorted(
        set(archive["home_team"].dropna()).union(set(archive["away_team"].dropna()))
    )

    col1, col2 = st.columns(2)
    source_filter = col1.selectbox("Source", source_options)
    team_filter = col2.selectbox("Team", team_options)

    filtered = archive.copy()
    if source_filter != "all":
        filtered = filtered[filtered["source"] == source_filter]
    if team_filter != "all":
        filtered = filtered[
            (filtered["home_team"] == team_filter) | (filtered["away_team"] == team_filter)
        ]

    st.dataframe(
        filtered[
            [
                "date",
                "source",
                "matchup",
                "home_win_probability",
                "away_win_probability",
                "predicted_winner",
                "actual_winner",
                "confidence_bucket",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_performance_page() -> None:
    comparison = _model_comparison()
    rolling = _rolling_validation()
    weights = _ensemble_weights()

    st.subheader("Model Performance")
    if comparison.empty:
        st.info("Model performance metrics are not available in this deployment yet.")
        return

    best_row = comparison.dropna(subset=["log_loss"]).sort_values("log_loss").iloc[0]
    _metric_row(
        [
            ("Best Log Loss", f"{best_row['log_loss']:.4f}", best_row["model"]),
            (
                "Best Accuracy",
                f"{comparison['accuracy'].max():.3f}",
                "Across loaded model families",
            ),
            ("Models Compared", str(len(comparison)), "Baselines, ablations, fusion, ensemble"),
        ]
    )

    st.markdown("#### Test-Set Comparison")
    st.dataframe(comparison, width="stretch", hide_index=True)

    if not rolling.empty:
        st.markdown("#### Rolling Validation Trend")
        rolling_chart = rolling.set_index("fold")[["accuracy", "log_loss"]]
        st.line_chart(rolling_chart, width="stretch")

    if not weights.empty:
        st.markdown("#### Ensemble Weights")
        st.bar_chart(weights.set_index("model"), width="stretch")


def render_calibration_page() -> None:
    calibration = _calibration_frame()
    st.subheader("Calibration")
    if calibration.empty:
        st.info("Calibration metrics are not available in this deployment yet.")
        return

    ece = data.compute_expected_calibration_error(calibration)
    _metric_row(
        [
            ("ECE", f"{ece:.4f}", "Expected calibration error"),
            ("Bins", str(len(calibration)), "Uniform probability buckets"),
            ("Max Gap", f"{calibration['abs_gap'].max():.4f}", "Largest observed reliability gap"),
        ]
    )

    chart = calibration.set_index("bin_mid")[["avg_pred", "actual_rate", "ideal"]]
    st.line_chart(chart, width="stretch")
    st.dataframe(calibration, width="stretch", hide_index=True)


def render_team_form_page() -> None:
    logs = _team_logs()
    st.subheader("Team Form")
    if logs.empty:
        st.info("Recent team form is not available in this deployment yet.")
        return

    team_ids = sorted(int(team_idx) for team_idx in logs["team_idx"].dropna().unique())
    labels = {f"{data.team_abbr(team_idx)} | {team_idx}": team_idx for team_idx in team_ids}
    selected = st.selectbox("Select a team", list(labels.keys()))
    team_idx = labels[selected]
    form = data.build_team_form_frame(team_idx)

    if form.empty:
        st.info("No recent games are available for the selected team.")
        return

    _metric_row(
        [
            (
                "Recent Record",
                f"{int(form['won'].sum())}-{len(form) - int(form['won'].sum())}",
                "Last 10 games",
            ),
            ("Avg Net Rating", f"{form['net_rating'].mean():.2f}", "Trailing 10 games"),
            ("Avg Point Diff", f"{form['point_diff'].mean():.2f}", "Trailing 10 games"),
        ]
    )

    st.line_chart(
        form.set_index("date")[["net_rating", "point_diff"]],
        width="stretch",
    )
    st.dataframe(
        form[
            [
                "date",
                "opponent",
                "location",
                "result",
                "points_for",
                "points_against",
                "net_rating",
                "pace",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def render_injury_page() -> None:
    summary = _injury_summary()
    st.subheader("Injury Impact")
    if summary.empty:
        st.info("Injury context is not available in this deployment yet.")
        return

    availability_rate = summary["data_available_rate"].mean()
    if availability_rate == 0:
        st.warning(
            "Current injury inputs are still proxy-based. This view is useful for relative "
            "team instability, but not yet for live player-level status."
        )

    st.bar_chart(
        summary.set_index("team")[["avg_estimated_value_missing", "avg_players_out"]],
        width="stretch",
    )
    st.dataframe(summary, width="stretch", hide_index=True)


def render_news_page() -> None:
    summary = _news_summary()
    st.subheader("News Sentiment")
    if summary.empty:
        st.info("News context is not available in this deployment yet.")
        return

    coverage_rate = summary["coverage_rate"].mean()
    if coverage_rate == 0:
        st.info(
            "Live pregame news coverage is not populated in the current artifact set, "
            "so this view is reflecting the fallback news features."
        )

    st.bar_chart(
        summary.set_index("team")[["avg_sentiment_72h", "avg_article_volume"]],
        width="stretch",
    )
    st.dataframe(summary, width="stretch", hide_index=True)


def render_dashboard() -> None:
    _set_page_config()
    apply_dashboard_styles()
    _hero()

    with st.sidebar:
        st.title("NBA Predict")
        latest_daily = _latest_daily_payload()
        manifest = _publish_manifest()
        st.caption(_slate_caption(latest_daily, manifest))
        selected_page = st.radio("Pages", PAGES)

    if selected_page == "This Week's Games":
        render_today_page()
    elif selected_page == "Game Detail":
        render_game_detail_page()
    elif selected_page == "Archive":
        render_archive_page()
    elif selected_page == "Performance":
        render_performance_page()
    elif selected_page == "Calibration":
        render_calibration_page()
    elif selected_page == "Team Form":
        render_team_form_page()
    elif selected_page == "Injury Impact":
        render_injury_page()
    elif selected_page == "News Sentiment":
        render_news_page()


def main() -> None:
    render_dashboard()


if __name__ == "__main__":
    main()
