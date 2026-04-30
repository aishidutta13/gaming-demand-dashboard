import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BASE_DATA_PATH = BASE_DIR / "model_data.csv"
SNAPSHOT_PATH = BASE_DIR / "data" / "steam_player_snapshots.csv"
OUTPUT_PATH = BASE_DIR / "data" / "model_data_updated.csv"
MIN_SNAPSHOTS_PER_MONTH = 7

FEATURE_COLS = [
    "year",
    "month",
    "event_flag",
    "game_encoded",
    "lag_1",
    "lag_7",
    "rolling_7",
]

MODEL_COLUMNS = [
    "players",
    "peak_players",
    "date",
    "game",
    "viewer_count",
    "event_flag",
    "year",
    "month",
    "game_encoded",
    "lag_1",
    "lag_7",
    "rolling_7",
]
METADATA_COLUMNS = ["target_source", "snapshot_count"]
SCAFFOLD_COLUMNS = [
    "game",
    "date",
    "players",
    "peak_players",
    "viewer_count",
    "event_flag",
    "game_encoded",
    *METADATA_COLUMNS,
]


def load_base_data(path=BASE_DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df["game"] = df["game"].str.lower().str.strip()

    if "target_source" not in df.columns:
        df["target_source"] = "historical_monthly_activity"

    if "snapshot_count" not in df.columns:
        df["snapshot_count"] = pd.NA

    return df.sort_values(["game", "date"]).reset_index(drop=True)


def load_snapshots(path=SNAPSHOT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No snapshot file found at {path}")

    snapshots = pd.read_csv(path, parse_dates=["date", "collected_at"])
    snapshots["game"] = snapshots["game"].str.lower().str.strip()
    snapshots["current_players"] = pd.to_numeric(snapshots["current_players"], errors="coerce")
    return snapshots.dropna(subset=["current_players"])


def build_monthly_rows(base_df, snapshots, min_snapshots_per_month=MIN_SNAPSHOTS_PER_MONTH):
    game_codes = (
        base_df[["game", "game_encoded"]]
        .drop_duplicates("game")
        .set_index("game")["game_encoded"]
    )

    monthly = (
        snapshots
        .assign(date=snapshots["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["game", "date"], as_index=False)
        .agg(
            players=("current_players", "mean"),
            peak_players=("current_players", "max"),
            snapshot_count=("current_players", "count")
        )
    )
    monthly = monthly[monthly["snapshot_count"] >= min_snapshots_per_month]
    monthly["players"] = monthly["players"].round(2)
    monthly["viewer_count"] = pd.NA
    monthly["event_flag"] = 0
    monthly["target_source"] = "steam_current_players_monthly_mean"
    monthly["game_encoded"] = monthly["game"].map(game_codes)
    monthly = monthly.dropna(subset=["game_encoded"])
    monthly["game_encoded"] = monthly["game_encoded"].astype(int)

    latest_existing = base_df.groupby("game")["date"].max()
    monthly["latest_existing"] = monthly["game"].map(latest_existing)
    monthly = monthly[monthly["date"] > monthly["latest_existing"]].drop(columns=["latest_existing"])

    return monthly


def add_time_features(combined):
    combined = combined.sort_values(["game", "date"]).copy()
    combined["year"] = combined["date"].dt.year
    combined["month"] = combined["date"].dt.month

    grouped = combined.groupby("game")["players"]
    combined["lag_1"] = grouped.shift(1)
    combined["lag_7"] = grouped.shift(7)
    combined["rolling_7"] = (
        grouped
        .shift(1)
        .groupby(combined["game"])
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return combined


def build_updated_training_data(
    base_path=BASE_DATA_PATH,
    snapshot_path=SNAPSHOT_PATH,
    output_path=OUTPUT_PATH,
    min_snapshots_per_month=MIN_SNAPSHOTS_PER_MONTH,
):
    base_df = load_base_data(base_path)
    snapshots = load_snapshots(snapshot_path)
    live_rows = build_monthly_rows(base_df, snapshots, min_snapshots_per_month)

    if live_rows.empty:
        print(
            "No new model-ready Steam rows were available. "
            f"Need at least {min_snapshots_per_month} snapshots per game-month."
        )
        output_df = base_df
    else:
        scaffold = pd.concat([
            base_df[SCAFFOLD_COLUMNS],
            live_rows[SCAFFOLD_COLUMNS],
        ], ignore_index=True)
        featured = add_time_features(scaffold)

        live_keys = set(zip(live_rows["game"], live_rows["date"]))
        live_featured = featured[
            featured.apply(lambda row: (row["game"], row["date"]) in live_keys, axis=1)
        ]
        output_columns = MODEL_COLUMNS + METADATA_COLUMNS
        output_df = pd.concat([base_df[output_columns], live_featured[output_columns]], ignore_index=True)
        output_df = output_df.sort_values(["game", "date"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved updated training data to {output_path}")
    print(f"Rows: {len(output_df)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build model-ready training data from Steam snapshots.")
    parser.add_argument("--base-data", type=Path, default=BASE_DATA_PATH)
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--min-snapshots-per-month", type=int, default=MIN_SNAPSHOTS_PER_MONTH)
    args = parser.parse_args()

    build_updated_training_data(
        args.base_data,
        args.snapshots,
        args.output,
        args.min_snapshots_per_month,
    )


if __name__ == "__main__":
    main()
