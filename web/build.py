"""Stage 4 - PUBLISH: render the static site in docs/ (served by GitHub Pages).

Usage:
    python web/build.py

Reads data/dota.db + docs/findings.json and writes docs/index.html and docs/matches.html.
Privacy: friends' names and account IDs are replaced by "Friend 1..N" before publishing.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "dota.db"
DOCS = ROOT / "docs"
RAW = ROOT / "data" / "raw"

# Columns the front-end needs (raw API names + played_at, the Eastern wall-clock time)
MATCH_COLS = [
    "match_id", "played_at", "start_time", "duration", "hero_id", "player_slot", "radiant_win", "game_mode",
    "lobby_type", "kills", "deaths", "assists", "party_size", "average_rank", "gold_per_min", "xp_per_min",
    "last_hits", "denies", "hero_damage", "tower_damage", "hero_healing", "level", "leaver_status",
    "item_0", "item_1", "item_2", "item_3", "item_4", "item_5", "item_neutral",
]


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def export() -> dict:
    with sqlite3.connect(DB) as con:
        m = pd.read_sql(f"SELECT {', '.join(MATCH_COLS)} FROM matches ORDER BY start_time", con)
        peers = pd.read_sql("SELECT * FROM peers ORDER BY with_games DESC", con)
        heroes = pd.read_sql("SELECT hero_id AS id, name AS localized_name FROM heroes", con)
    # party_size 0 means "unknown" in the DB; the front-end expects null
    m["party_size"] = m["party_size"].where(m["party_size"] > 0)
    # SQLite stores booleans as 0/1; the front-end compares against real JS booleans
    m["radiant_win"] = m["radiant_win"].astype(bool)
    m = m.astype(object).where(m.notna(), None)

    peers = peers.copy()
    peers["personaname"] = [f"Friend {i + 1}" for i in range(len(peers))]
    peers["account_id"] = None
    keep = ["personaname", "account_id", "with_games", "with_win", "against_games", "against_win", "games", "last_played"]
    return {
        "MATCHES": m.to_dict("records"),
        "PEERS": peers[keep].astype(object).where(peers[keep].notna(), None).to_dict("records"),
        "HEROES": heroes.to_dict("records"),
        "ITEMS": json.loads((RAW / "item_ids.json").read_text(encoding="utf-8")),
        "FINDINGS": json.loads((DOCS / "findings.json").read_text(encoding="utf-8")),
    }


def render(template: Path, out: Path, data: dict) -> None:
    html = template.read_text(encoding="utf-8")
    for key, value in data.items():
        html = html.replace(f"/*{key}*/", dumps(value))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    data = export()
    render(ROOT / "web" / "dashboard_template.html", DOCS / "index.html", data)
    render(ROOT / "web" / "tables_template.html", DOCS / "matches.html", data)


if __name__ == "__main__":
    main()
