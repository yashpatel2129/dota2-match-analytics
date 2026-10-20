"""Stage 2 - TRANSFORM + LOAD: clean raw JSON, derive features, load into SQLite.

Usage:
    python -m pipeline.transform

Output: data/dota.db with tables
    matches  one row per game, raw + derived columns (see `build_matches`)
    heroes   hero_id -> name
    peers    players I've shared games with (names are NOT published)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "dota.db"

GAME_MODES = {
    0: "Unknown", 1: "All Pick", 2: "Captains Mode", 3: "Random Draft", 4: "Single Draft",
    5: "All Random", 13: "Limited Heroes", 16: "Captains Draft", 18: "Ability Draft",
    19: "Event", 21: "1v1 Mid", 22: "Ranked/All Draft", 23: "Turbo",
}
LOBBY_TYPES = {
    0: "Normal", 1: "Practice", 2: "Tournament", 4: "Co-op Bots", 5: "Team Match",
    6: "Solo Queue", 7: "Ranked", 8: "Solo Mid", 9: "Battle Cup",
}
SESSION_GAP_S = 2 * 3600  # a gap of >2h between games starts a new session


def load_raw(name: str):
    return json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))


def build_matches(raw: list[dict], hero_names: dict[int, str]) -> pd.DataFrame:
    """Raw API rows -> analysis-ready DataFrame (pure function, unit-tested)."""
    df = pd.DataFrame(raw).drop_duplicates("match_id")
    df = df.dropna(subset=["radiant_win"])  # a few games have no result (abandoned/unscored)
    df = df.sort_values("start_time").reset_index(drop=True)

    # --- target variable ---------------------------------------------------
    on_radiant = df["player_slot"] < 128
    df["win"] = (on_radiant == df["radiant_win"].astype(bool)).astype(int)
    df["side"] = np.where(on_radiant, "Radiant", "Dire")

    # --- time features (US Eastern - where I play from) ----
    ts = pd.to_datetime(df["start_time"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df["played_at"] = ts.dt.strftime("%Y-%m-%d %H:%M")
    df["date"] = ts.dt.strftime("%Y-%m-%d")
    df["year"] = ts.dt.year
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.dayofweek  # Monday=0

    # --- labels ------------------------------------------------------------
    df["hero"] = df["hero_id"].map(hero_names).fillna(df["hero_id"].map(lambda x: f"Hero {x}"))
    df["mode"] = df["game_mode"].map(GAME_MODES).fillna(df["game_mode"].map(lambda x: f"Mode {x}"))
    df["lobby"] = df["lobby_type"].map(LOBBY_TYPES).fillna(df["lobby_type"].map(lambda x: f"Lobby {x}"))
    df["is_turbo"] = (df["game_mode"] == 23).astype(int)

    # --- missing data handling --------------------------------------------
    # party_size is NULL for ~25% of games (older/some lobbies). Treat as "unknown"
    # for party analyses rather than assuming solo.
    df["party_known"] = df["party_size"].notna().astype(int)
    df["party_size"] = df["party_size"].fillna(0).astype(int)  # 0 = unknown
    for col in ("hero_damage", "tower_damage", "hero_healing"):
        df[col] = df[col].fillna(0)

    # --- performance features -----------------------------------------------
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)
    df["duration_min"] = df["duration"] / 60

    # --- sessions & momentum -------------------------------------------------
    end = df["start_time"] + df["duration"]
    gap = df["start_time"] - end.shift(1)
    df["gap_since_prev_s"] = gap
    new_session = gap.isna() | (gap > SESSION_GAP_S)
    df["session_id"] = new_session.cumsum()
    df["game_in_session"] = df.groupby("session_id").cumcount() + 1
    df["game_in_day"] = df.groupby("date").cumcount() + 1
    df["games_that_day"] = df.groupby("date")["match_id"].transform("count")

    prev_win = df.groupby("session_id")["win"].shift(1)  # NaN for first game of a session
    df["prev_win"] = prev_win
    df["streak"] = _signed_streak(df)  # streak going INTO this game, within session
    return df


def _signed_streak(df: pd.DataFrame) -> pd.Series:
    """+k = k consecutive wins before this game, -k = k consecutive losses (same session)."""
    out = np.zeros(len(df), dtype=int)
    sess = df["session_id"].to_numpy()
    win = df["win"].to_numpy()
    for i in range(1, len(df)):
        if sess[i] != sess[i - 1]:
            continue  # new session -> no carry-over
        prev = out[i - 1]
        if win[i - 1] == 1:
            out[i] = prev + 1 if prev > 0 else 1
        else:
            out[i] = prev - 1 if prev < 0 else -1
    return pd.Series(out, index=df.index)


def main() -> None:
    heroes = {h["id"]: h["localized_name"] for h in load_raw("heroes")}
    matches = build_matches(load_raw("matches"), heroes)
    peers = pd.DataFrame(load_raw("peers"))

    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:
        matches.to_sql("matches", con, if_exists="replace", index=False)
        pd.DataFrame({"hero_id": list(heroes), "name": list(heroes.values())}).to_sql(
            "heroes", con, if_exists="replace", index=False)
        peers.to_sql("peers", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_matches_date ON matches(date)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_matches_hero ON matches(hero_id)")
    print(f"loaded {len(matches):,} matches, {len(peers)} peers -> {DB}")


if __name__ == "__main__":
    main()
