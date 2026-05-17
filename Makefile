.PHONY: setup fetch-data build-features build-m1-features train-baseline train-model evaluate predict-today serve-api test lint clean

# ---- Setup ----
setup:
	pip install -r requirements.txt

# ---- Data Pipeline ----
fetch-data:
	python -m src.data.fetch_games
	python -m src.data.fetch_boxscores

build-features:
	python -m src.features.build_team_game_logs
	python -m src.features.rolling_features
	python -m src.features.schedule_features
	python -m src.features.build_matchup_dataset

build-m1-features:
	python -m src.data.fetch_player_logs
	python -m src.features.player_value_features
	python -m src.features.projected_availability
	python -m src.features.lineup_features
	python -m src.features.build_matchup_dataset

# ---- Training ----
train-baseline:
	python -m src.models.elo
	python -m src.models.tabular_model

train-model:
	python -m src.models.train

# ---- Evaluation ----
evaluate:
	python -m src.models.evaluate

# ---- Prediction ----
predict-today:
	python -m src.app.predict_today

serve-api:
	uvicorn src.app.api:app --reload

# ---- Quality ----
test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# ---- Housekeeping ----
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
