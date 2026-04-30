# Arenix: Gaming Demand Intelligence

Arenix is a Flask dashboard that forecasts next-month player demand for Steam games using historical player activity and a validation-selected demand model. The app combines model predictions with live Steam player counts, game artwork, search, a demand leaderboard, and game-to-game comparison views.

## What It Does

- Forecasts the next monthly player count for supported games.
- Reports whether each forecast came from XGBoost, the selected lag baseline, or the runtime fallback baseline.
- Shows the latest observed historical player count and live Steam player count when available.
- Classifies demand as low, medium, or high.
- Compares multiple games using real historical series plus the next forecast point.
- Uses Steam artwork and live-player APIs for product-style dashboard polish.

## Dataset

The included dataset is `model_data.csv`.

- Historical range: `2016-08-01` to `2021-09-01`
- Granularity: monthly
- Target: `players`
- Model features: `year`, `month`, `event_flag`, `game_encoded`, `lag_1`, `lag_7`, `rolling_7`

Because the bundled data ends in September 2021, forecasts are next-period forecasts from the latest available historical row, not claims about the current real-world month. Live Steam player counts are displayed separately as product context.

When `data/model_data_updated.csv` exists, the Flask app uses it automatically so inference rows stay aligned with the latest retrained model. You can override this with `ARENIX_DATA_PATH=/path/to/data.csv`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost may also require the OpenMP runtime:

```bash
brew install libomp
```

If XGBoost cannot load because the native OpenMP runtime is missing, the Flask app remains usable by falling back to a transparent lag-and-rolling-average baseline. API responses include `model_source` and `model_warning` so this is visible instead of hidden.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5002`.

## Test

```bash
pytest
```

## Train

```bash
python train_model.py
```

The training script uses a time-based validation split so later months are evaluated as future data instead of randomly mixed with older records. It evaluates XGBoost against a lag-1 baseline and saves the candidate with the lowest validation MAE. It also writes `model/training_report.json` with selected-model metadata, candidate metrics, feature schema, data hash, date range, and target-source metadata.

## Daily Steam Retraining Pipeline

Steam's player-count API gives the number of players online at the moment it is called. The project now stores those daily snapshots first, then turns them into model-ready rows before retraining.

To avoid treating a single live count as a full monthly target, `build_live_training_data.py` only creates a game-month row after at least 7 snapshots are available for that game and month. Generated rows include `target_source=steam_current_players_monthly_mean` and `snapshot_count` so the live-data target definition is explicit.

Run the full pipeline:

```bash
python run_daily_pipeline.py
```

What it does:

```text
Steam API gives a fresh player-count snapshot
        ↓
Save it in data/steam_player_snapshots.csv
        ↓
Build data/model_data_updated.csv
        ↓
Retrain model/final_demand_model.pkl
        ↓
Deploy platform can pick up the new committed model
```

Useful commands:

```bash
python collect_steam_snapshots.py
python build_live_training_data.py
python train_model.py --data data/model_data_updated.csv
python train_model.py --data data/model_data_updated.csv --metrics model/training_report.json
python run_daily_pipeline.py --skip-collect
```

The GitHub Actions workflow in `.github/workflows/daily-retrain.yml` can run this every day after you push the project to GitHub. If your deployment service is connected to the GitHub repo, the new committed model can trigger a redeploy automatically.

## Project Structure

```text
app.py                         Flask API and web routes
train_model.py                 Model training, comparison, and time-based evaluation
demand_model.py                Serializable baseline model used when it wins validation
run_daily_pipeline.py          Daily collect-build-train workflow
collect_steam_snapshots.py     Steam live-player snapshot collector
build_live_training_data.py    Converts snapshots into model-ready data
game_catalog.py                Shared game names and Steam app ids
data/                          Generated Steam snapshots and updated training data
model_data.csv                 Historical game demand dataset
model/final_demand_model.pkl   Selected trained demand model artifact
model/training_report.json     Saved training metrics and model/data metadata
templates/                     Dashboard pages
static/style.css               App styling
tests/                         Regression tests
```

## Current Limitations

- Unsupported games return a clear error instead of fabricated predictions.
- Steam API calls can fail or be rate-limited, so the app falls back to the latest historical player count when live data is unavailable.
