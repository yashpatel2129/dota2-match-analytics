"""Stage 1 - EXTRACT: download raw data from the OpenDota API into data/raw/.

Usage:
    python -m pipeline.extract --account-id 123456789

Design notes
- Raw responses are stored untouched (one JSON file per endpoint) so every later
  stage can be re-run offline and results are reproducible.
- `significant=0` is REQUIRED. By default OpenDota hides "non-significant" modes
  (e.g. Turbo), which silently dropped ~50% of my games in my first attempt.
- Requests retry with exponential backoff because the free API is rate limited.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://api.opendota.com/api"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Per-match columns requested from /players/{id}/matches
MATCH_FIELDS = [
    "start_time", "duration", "hero_id", "player_slot", "radiant_win", "game_mode",
    "lobby_type", "kills", "deaths", "assists", "party_size", "lane_role",
    "hero_variant", "average_rank", "gold_per_min", "xp_per_min", "last_hits",
    "denies", "hero_damage", "tower_damage", "hero_healing", "level",
    "leaver_status", "item_0", "item_1", "item_2", "item_3", "item_4", "item_5",
    "item_neutral", "lane", "is_roaming", "skill", "cluster",
]


def get_json(url: str, params=None, retries: int = 5):
    """GET with exponential backoff on rate limits / transient errors."""
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"  HTTP {resp.status_code}, retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Giving up on {url}")


def save(name: str, data) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    n = len(data) if hasattr(data, "__len__") else "-"
    print(f"saved {path.name}: {n} records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--account-id", default=os.environ.get("DOTA_ACCOUNT_ID"),
                    help="Dota/Steam32 account id (or set DOTA_ACCOUNT_ID)")
    args = ap.parse_args()
    if not args.account_id:
        ap.error("account id required")

    base = f"{API}/players/{args.account_id}"
    params = [("significant", 0)] + [("project", f) for f in ["match_id", *MATCH_FIELDS]]
    save("matches", get_json(f"{base}/matches", params=params))
    save("peers", get_json(f"{base}/peers"))
    save("heroes", get_json(f"{API}/heroes"))
    save("item_ids", get_json(f"{API}/constants/item_ids"))


if __name__ == "__main__":
    main()
