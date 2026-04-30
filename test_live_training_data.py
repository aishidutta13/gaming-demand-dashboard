import pandas as pd

from build_live_training_data import build_updated_training_data


def test_build_updated_training_data_appends_future_snapshot_rows(tmp_path):
    base_path = tmp_path / "model_data.csv"
    snapshot_path = tmp_path / "steam_player_snapshots.csv"
    output_path = tmp_path / "model_data_updated.csv"

    base_rows = []
    for month in range(1, 9):
        base_rows.append({
            "players": 1000 + month * 100,
            "peak_players": 1500 + month * 100,
            "date": f"2021-{month:02d}-01",
            "game": "dota 2",
            "viewer_count": 100,
            "event_flag": 0,
            "year": 2021,
            "month": month,
            "game_encoded": 1,
            "lag_1": 900 + month * 100,
            "lag_7": 300 + month * 100,
            "rolling_7": 1100,
        })

    pd.DataFrame(base_rows).to_csv(base_path, index=False)
    snapshot_rows = []
    for day in range(1, 8):
        snapshot_rows.append({
            "collected_at": f"2021-09-{day:02d}T00:00:00+00:00",
            "date": f"2021-09-{day:02d}",
            "game": "dota 2",
            "steam_appid": 570,
            "current_players": 2500,
            "source": "steam_current_players",
        })

    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False)

    build_updated_training_data(base_path, snapshot_path, output_path)
    updated = pd.read_csv(output_path)
    latest = updated.sort_values("date").iloc[-1]

    assert len(updated) == 9
    assert latest["date"] == "2021-09-01"
    assert latest["players"] == 2500
    assert latest["lag_1"] == 1800
    assert latest["lag_7"] == 1200
    assert latest["target_source"] == "steam_current_players_monthly_mean"
    assert latest["snapshot_count"] == 7


def test_build_updated_training_data_waits_for_enough_snapshots(tmp_path):
    base_path = tmp_path / "model_data.csv"
    snapshot_path = tmp_path / "steam_player_snapshots.csv"
    output_path = tmp_path / "model_data_updated.csv"

    pd.DataFrame([{
        "players": 1000,
        "peak_players": 1500,
        "date": "2021-08-01",
        "game": "dota 2",
        "viewer_count": 100,
        "event_flag": 0,
        "year": 2021,
        "month": 8,
        "game_encoded": 1,
        "lag_1": 900,
        "lag_7": 300,
        "rolling_7": 800,
    }]).to_csv(base_path, index=False)

    pd.DataFrame([{
        "collected_at": "2021-09-01T00:00:00+00:00",
        "date": "2021-09-01",
        "game": "dota 2",
        "steam_appid": 570,
        "current_players": 2500,
        "source": "steam_current_players",
    }]).to_csv(snapshot_path, index=False)

    build_updated_training_data(base_path, snapshot_path, output_path)
    updated = pd.read_csv(output_path)

    assert len(updated) == 1
    assert updated.iloc[0]["target_source"] == "historical_monthly_activity"
