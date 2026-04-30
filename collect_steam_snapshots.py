import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from game_catalog import canonical_game_appids


BASE_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = BASE_DIR / "data" / "steam_player_snapshots.csv"
STEAM_PLAYERS_URL = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"


def fetch_current_players(appid, timeout=8):
    response = requests.get(STEAM_PLAYERS_URL, params={"appid": appid}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return int(data["response"]["player_count"])


def collect_snapshots(output_path=SNAPSHOT_PATH):
    collected_at = datetime.now(timezone.utc).replace(microsecond=0)
    rows = []

    for game, appid in canonical_game_appids().items():
        try:
            players = fetch_current_players(appid)
            rows.append({
                "collected_at": collected_at.isoformat(),
                "date": collected_at.date().isoformat(),
                "game": game,
                "steam_appid": appid,
                "current_players": players,
                "source": "steam_current_players"
            })
            print(f"{game}: {players}")
        except Exception as exc:
            print(f"{game}: skipped ({exc})")

    if not rows:
        raise RuntimeError("No Steam snapshots were collected.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)

    if output_path.exists():
        old_df = pd.read_csv(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = (
        combined
        .sort_values(["date", "game", "collected_at"])
        .drop_duplicates(["date", "game", "steam_appid"], keep="last")
    )
    combined.to_csv(output_path, index=False)

    print(f"Saved {len(new_df)} new snapshots to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Collect daily Steam player-count snapshots.")
    parser.add_argument("--output", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()

    collect_snapshots(args.output)


if __name__ == "__main__":
    main()
