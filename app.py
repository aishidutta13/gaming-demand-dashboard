from flask import Flask, jsonify, render_template
import os
import pickle
import pandas as pd
import requests
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from game_catalog import GAME_APPIDS, game_category, normalize_game_name, steam_header_url, title_game

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "final_demand_model.pkl"
BASE_DATA_PATH = BASE_DIR / "model_data.csv"
UPDATED_DATA_PATH = BASE_DIR / "data" / "model_data_updated.csv"


def resolve_data_path():
    configured_path = os.getenv("ARENIX_DATA_PATH")

    if configured_path:
        return Path(configured_path).expanduser()

    if UPDATED_DATA_PATH.exists():
        return UPDATED_DATA_PATH

    return BASE_DATA_PATH


DATA_PATH = resolve_data_path()

_model = None
_model_load_error = None
_model_warning_logged = False

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

feature_cols = [
    "year", "month", "event_flag",
    "game_encoded", "lag_1", "lag_7", "rolling_7"
]


class ModelRuntimeUnavailable(RuntimeError):
    pass


class PredictionModelError(RuntimeError):
    pass


def is_xgboost_runtime_error(exc):
    message = str(exc).lower()
    runtime_markers = [
        "xgboost library",
        "libxgboost",
        "libomp",
        "openmp",
        "library not loaded",
    ]
    return any(marker in message for marker in runtime_markers)


def get_model():
    global _model, _model_load_error

    if _model_load_error is not None:
        raise _model_load_error

    if _model is None:
        try:
            with MODEL_PATH.open("rb") as f:
                _model = pickle.load(f)
        except Exception as exc:
            if is_xgboost_runtime_error(exc):
                _model_load_error = ModelRuntimeUnavailable(
                    "XGBoost model could not be loaded because the native runtime "
                    "dependency is unavailable."
                )
                raise _model_load_error from exc

            raise PredictionModelError("Unable to load prediction model.") from exc

    return _model


def model_source_label(model):
    return getattr(model, "model_source", model.__class__.__name__.lower())


def predict_with_model(X_input):
    try:
        model = get_model()
        prediction = model.predict(X_input[feature_cols])[0]
    except ModelRuntimeUnavailable:
        raise
    except Exception as exc:
        raise PredictionModelError(
            "Model prediction failed. Check feature schema and model artifact."
        ) from exc

    return max(0, int(round(prediction))), model_source_label(model)


def relative_path(path):
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def get_all_dataset_games():
    return (
        df["game"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def get_game_rows(game):
    normalized = normalize_game_name(game)
    rows = df[df["game"].str.lower().str.strip() == normalized].copy()
    return rows.sort_values("date")


def find_best_game_match(game):
    game = normalize_game_name(game)
    all_games = get_all_dataset_games()

    if game in all_games:
        return game

    matches = get_close_matches(game, all_games, n=1, cutoff=0.70)
    return matches[0] if matches else None


@lru_cache(maxsize=512)
def lookup_steam_store(game):
    try:
        normalized = normalize_game_name(game)

        response = requests.get(
            "https://store.steampowered.com/api/storesearch/",
            params={"term": normalized, "l": "en", "cc": "US"},
            timeout=4
        )
        data = response.json()
        items = data.get("items", [])

        if not items:
            return {"appid": None, "image_url": None}

        exact_match = None

        for item in items:
            item_name = normalize_game_name(item.get("name", ""))
            if item_name == normalized:
                exact_match = item
                break

        best = exact_match or items[0]
        appid = best.get("id")

        if appid:
            return {
                "appid": int(appid),
                "image_url": steam_header_url(appid)
            }

        return {
            "appid": None,
            "image_url": best.get("tiny_image")
        }

    except Exception as e:
        print("Steam store image lookup error:", e)
        return {"appid": None, "image_url": None}


def get_game_media(game):
    normalized = normalize_game_name(game)
    appid = GAME_APPIDS.get(normalized)

    if appid:
        return {
            "appid": appid,
            "image_url": steam_header_url(appid)
        }

    return lookup_steam_store(normalized)


def get_local_game_media(game):
    normalized = normalize_game_name(game)
    appid = GAME_APPIDS.get(normalized)

    if not appid:
        return {"appid": None, "image_url": None}

    return {
        "appid": appid,
        "image_url": steam_header_url(appid)
    }


@lru_cache(maxsize=512)
def get_live_players(appid):
    try:
        if not appid:
            return None

        url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}"
        response = requests.get(url, timeout=5)
        data = response.json()
        player_count = data.get("response", {}).get("player_count")

        if player_count is None:
            return None

        return int(player_count)

    except Exception as e:
        print("Live player API error:", e)
        return None


def demand_level(players):
    if players >= 400000:
        return "High"
    elif players >= 100000:
        return "Medium"
    return "Low"


def recommendation(level):
    if level == "High":
        return "Increase server capacity"
    elif level == "Medium":
        return "Maintain normal resources"
    return "Schedule promotional event"


def calculate_gain_percent(current_players, predicted_players):
    if current_players is None or current_players == 0:
        return 0.0

    gain = ((predicted_players - current_players) / current_players) * 100
    return round(gain, 1)


def trend_label(gain_percent):
    if gain_percent >= 5:
        return "Rising"
    if gain_percent <= -5:
        return "Falling"
    return "Stable"


def forecast_confidence(game_rows):
    recent = game_rows["players"].tail(7)

    if len(recent) < 4 or recent.mean() == 0:
        return "Medium"

    volatility = recent.std() / recent.mean()

    if volatility <= 0.18:
        return "High"
    if volatility <= 0.35:
        return "Medium"
    return "Low"


def server_risk_score(predicted_players, gain_percent):
    load_score = min(80, (predicted_players / 500000) * 80)
    growth_score = min(20, max(0, gain_percent) / 50 * 20)
    return int(round(min(100, load_score + growth_score)))


def opportunity_score(current_players, predicted_players, gain_percent):
    if current_players is None or current_players <= 0:
        return 0

    growth_component = min(55, max(0, gain_percent) * 1.6)
    room_component = max(0, 35 - min(35, current_players / 150000 * 35))
    forecast_component = min(10, predicted_players / 250000 * 10)
    return int(round(min(100, growth_component + room_component + forecast_component)))


def anomaly_signal(game_rows):
    recent = game_rows["players"].tail(7)

    if len(recent) < 4:
        return "Normal"

    baseline = recent.iloc[:-1].mean()
    latest = recent.iloc[-1]

    if baseline == 0:
        return "Normal"

    change = ((latest - baseline) / baseline) * 100

    if change >= 35:
        return "Unusual Spike"
    if change <= -35:
        return "Sudden Drop"
    return "Normal"


def business_action(level, trend, server_risk, opportunity):
    if server_risk >= 75:
        return "Scale servers before the next demand window"
    if opportunity >= 65:
        return "Prioritize promotion while growth potential is high"
    if trend == "Falling":
        return "Monitor drop and test re-engagement campaign"
    if level == "Medium":
        return "Maintain capacity and watch demand movement"
    return recommendation(level)


def explain_prediction(trend, confidence, gain_percent, anomaly):
    movement = f"{trend.lower()} because forecast change is {gain_percent:+.1f}%."
    confidence_note = f"Confidence is {confidence.lower()} based on recent demand stability."
    anomaly_note = "" if anomaly == "Normal" else f" Recent history shows {anomaly.lower()}."
    return movement + " " + confidence_note + anomaly_note


def build_next_period_features(game_rows):
    latest_row = game_rows.iloc[-1]
    recent_players = game_rows["players"].tail(7)
    forecast_date = latest_row["date"] + pd.DateOffset(months=1)

    return pd.DataFrame([{
        "year": forecast_date.year,
        "month": forecast_date.month,
        "event_flag": 0,
        "game_encoded": latest_row["game_encoded"],
        "lag_1": latest_row["players"],
        "lag_7": recent_players.iloc[0] if len(recent_players) >= 7 else latest_row["players"],
        "rolling_7": recent_players.mean()
    }]), forecast_date


def build_history(game_rows, periods=7):
    history_rows = game_rows.tail(periods)

    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "players": int(round(row["players"])),
            "viewer_count": int(round(row["viewer_count"])) if pd.notna(row["viewer_count"]) else None
        }
        for _, row in history_rows.iterrows()
    ]


def baseline_predict(X_input):
    row = X_input.iloc[0]
    recent_trend = (row["lag_1"] - row["lag_7"]) / 6 if row["lag_7"] else 0
    prediction = row["rolling_7"] + recent_trend
    return max(0, int(round(prediction)))


def build_intelligence(game_rows, current_players, predicted_players):
    gain_percent = calculate_gain_percent(current_players, predicted_players)
    trend = trend_label(gain_percent)
    confidence = forecast_confidence(game_rows)
    risk = server_risk_score(predicted_players, gain_percent)
    opportunity = opportunity_score(current_players, predicted_players, gain_percent)
    anomaly = anomaly_signal(game_rows)
    level = demand_level(predicted_players)

    return {
        "gain_percent": gain_percent,
        "trend_label": trend,
        "forecast_confidence": confidence,
        "server_risk_score": risk,
        "opportunity_score": opportunity,
        "anomaly_signal": anomaly,
        "business_action": business_action(level, trend, risk, opportunity),
        "prediction_explanation": explain_prediction(trend, confidence, gain_percent, anomaly)
    }


def get_prediction_data(game, use_live_players=True, use_store_lookup=True):
    searched_game = str(game).lower().strip()
    matched_game = find_best_game_match(searched_game)

    if matched_game is None:
        return {
            "error": "Unsupported game. Please choose a game from the available dashboard list.",
            "searched_game": searched_game
        }

    media = (
        get_game_media(matched_game)
        if use_store_lookup
        else get_local_game_media(matched_game)
    )

    appid = media.get("appid")
    live_players = get_live_players(appid) if use_live_players else None

    game_rows = get_game_rows(matched_game)

    if game_rows.empty:
        return {
            "error": "Unsupported game. No historical data is available for this title.",
            "searched_game": searched_game
        }

    X_input, forecast_date = build_next_period_features(game_rows)

    model_source = "xgboost"
    model_warning = None

    try:
        predicted_players, model_source = predict_with_model(X_input)
    except ModelRuntimeUnavailable:
        global _model_warning_logged

        if not _model_warning_logged:
            print(
                "XGBoost model is unavailable because the native OpenMP runtime "
                "is missing. Using the transparent historical baseline fallback. "
                "On macOS, install it with: brew install libomp"
            )
            _model_warning_logged = True

        predicted_players = baseline_predict(X_input)
        model_source = "historical_baseline_fallback"
        model_warning = "XGBoost could not be loaded, so a lag-and-rolling-average baseline was used."

    latest_row = game_rows.iloc[-1]
    latest_historical_players = int(round(latest_row["players"]))

    current_players = live_players if live_players is not None else latest_historical_players
    player_count_source = "steam_live" if live_players is not None else "latest_historical"
    level = demand_level(predicted_players)
    intelligence = build_intelligence(game_rows, current_players, predicted_players)

    display_name = title_game(matched_game)

    return {
        "searched_game": searched_game,
        "matched_game": display_name,
        "game": display_name,
        "display_name": display_name,
        "steam_appid": appid,
        "image_url": media.get("image_url"),

        "live_players": live_players,
        "latest_historical_players": latest_historical_players,
        "current_players": current_players,
        "player_count_source": player_count_source,
        "predicted_players": predicted_players,
        "gain_percent": intelligence["gain_percent"],
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "forecast_horizon": "next_month",
        "last_observed_date": latest_row["date"].strftime("%Y-%m-%d"),
        "history": build_history(game_rows),
        "model_source": model_source,
        "model_warning": model_warning,
        "data_source": relative_path(DATA_PATH),
        "category": game_category(matched_game),
        **intelligence,

        "demand_level": level,
        "recommendation": recommendation(level)
    }


def get_dashboard_card_data(game):
    data = get_prediction_data(game, use_live_players=False, use_store_lookup=False)
    return None if data.get("error") else data


def get_game_listing(game):
    matched_game = find_best_game_match(game)
    media = get_local_game_media(matched_game)
    display_name = title_game(matched_game)

    return {
        "key": matched_game,
        "name": display_name,
        "matched_game": display_name,
        "display_name": display_name,
        "steam_appid": media.get("appid"),
        "image_url": media.get("image_url")
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/compare")
def compare():
    return render_template("compare.html")


@app.route("/predict/<path:game>")
def predict(game):
    data = get_prediction_data(game)
    status_code = 404 if data.get("error") else 200
    return jsonify(data), status_code


@app.route("/dashboard-data")
def dashboard_data():
    all_games = get_all_dataset_games()
    results = [get_dashboard_card_data(game) for game in all_games]
    results = [result for result in results if result is not None]
    results.sort(key=lambda item: item.get("predicted_players") or 0, reverse=True)
    return jsonify(results)


@app.route("/compare-games/<path:games>")
def compare_games(games):
    game_list = [g.strip() for g in games.split(",") if g.strip()]
    results = [get_prediction_data(game) for game in game_list]
    return jsonify(results)


@app.route("/recommended-games")
def recommended_games():
    all_games = get_all_dataset_games()
    return jsonify([get_game_listing(game) for game in all_games])


if __name__ == "__main__":
    app.run(debug=True, port=5002)
