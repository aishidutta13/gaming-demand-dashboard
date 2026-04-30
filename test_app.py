import pytest

import app as arenix_app


class FakeModel:
    model_source = "xgboost"

    def predict(self, X):
        return [123456]


class BrokenModel:
    def predict(self, X):
        raise ValueError("feature schema mismatch")


def test_unknown_game_returns_error():
    data = arenix_app.get_prediction_data("a game that is not in the dataset")

    assert "error" in data
    assert data["searched_game"] == "a game that is not in the dataset"


def test_known_game_prediction_has_forecast_metadata(monkeypatch):
    monkeypatch.setattr(arenix_app, "_model", FakeModel())
    monkeypatch.setattr(arenix_app, "_model_load_error", None)
    monkeypatch.setattr(arenix_app, "get_live_players", lambda appid: None)

    data = arenix_app.get_prediction_data("dota 2")

    assert "error" not in data
    assert data["forecast_horizon"] == "next_month"
    assert data["forecast_date"] > data["last_observed_date"]
    assert isinstance(data["predicted_players"], int)
    assert len(data["history"]) > 0
    assert data["trend_label"] in {"Rising", "Stable", "Falling"}
    assert data["forecast_confidence"] in {"High", "Medium", "Low"}
    assert 0 <= data["server_risk_score"] <= 100
    assert 0 <= data["opportunity_score"] <= 100
    assert data["category"]
    assert data["prediction_explanation"]
    assert data["model_source"] == "xgboost"
    assert data["live_players"] is None
    assert data["player_count_source"] == "latest_historical"
    assert data["current_players"] == data["latest_historical_players"]
    assert data["data_source"]


def test_dashboard_card_uses_model_prediction(monkeypatch):
    monkeypatch.setattr(arenix_app, "_model", FakeModel())
    monkeypatch.setattr(arenix_app, "_model_load_error", None)

    data = arenix_app.get_dashboard_card_data("dota 2")

    assert data["predicted_players"] == 123456
    assert data["model_source"] == "xgboost"


def test_prediction_model_errors_are_not_hidden(monkeypatch):
    monkeypatch.setattr(arenix_app, "_model", BrokenModel())
    monkeypatch.setattr(arenix_app, "_model_load_error", None)
    monkeypatch.setattr(arenix_app, "get_live_players", lambda appid: None)

    with pytest.raises(arenix_app.PredictionModelError):
        arenix_app.get_prediction_data("dota 2")


def test_predict_route_returns_404_for_unsupported_game():
    client = arenix_app.app.test_client()

    response = client.get("/predict/not-a-real-game-title")

    assert response.status_code == 404
    assert response.get_json()["error"]
