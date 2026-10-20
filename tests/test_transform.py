"""Unit tests for the cleaning / feature logic - the part where silent bugs would corrupt every result."""
import pandas as pd
import pytest

from pipeline.transform import SESSION_GAP_S, build_matches

HEROES = {1: "Anti-Mage", 2: "Axe"}


def row(match_id, start, slot, radiant_win, **kw):
    base = dict(match_id=match_id, start_time=start, duration=1800, hero_id=1, player_slot=slot,
                radiant_win=radiant_win, game_mode=22, lobby_type=7, kills=5, deaths=0, assists=5,
                party_size=1, hero_damage=None, tower_damage=None, hero_healing=None)
    base.update(kw)
    return base


def test_win_flag_radiant_and_dire():
    df = build_matches([row(1, 1000, 0, True), row(2, 9000, 0, False),
                        row(3, 20000, 128, True), row(4, 30000, 128, False)], HEROES)
    assert df["win"].tolist() == [1, 0, 0, 1]
    assert df["side"].tolist() == ["Radiant", "Radiant", "Dire", "Dire"]


def test_unscored_games_are_dropped():
    df = build_matches([row(1, 1000, 0, True), row(2, 9000, 0, None)], HEROES)
    assert len(df) == 1


def test_duplicates_removed_and_sorted_chronologically():
    df = build_matches([row(2, 9000, 0, True), row(1, 1000, 0, True), row(1, 1000, 0, True)], HEROES)
    assert df["match_id"].tolist() == [1, 2]


def test_unknown_party_size_is_not_assumed_solo():
    df = build_matches([row(1, 1000, 0, True, party_size=None), row(2, 9000, 0, True, party_size=1)], HEROES)
    assert df["party_known"].tolist() == [0, 1]
    assert df["party_size"].tolist() == [0, 1]


def test_kda_never_divides_by_zero():
    df = build_matches([row(1, 1000, 0, True, kills=4, assists=6, deaths=0)], HEROES)
    assert df["kda"].iloc[0] == 10


def test_sessions_split_on_long_gap():
    t0 = 1_000_000
    rows = [row(1, t0, 0, True), row(2, t0 + 1800 + 600, 0, True),                       # 10 min after game 1 ended
            row(3, t0 + 2 * 1800 + 600 + SESSION_GAP_S + 5, 0, True)]                     # > 2h after game 2 ended
    df = build_matches(rows, HEROES)
    assert df["session_id"].nunique() == 2
    assert df["game_in_session"].tolist() == [1, 2, 1]


def test_signed_streak_resets_between_sessions_and_on_flip():
    t = 1_000_000
    step = 1800 + 600
    # W W L L L  |(new session)|  W
    wins = [True, True, False, False, False]
    rows = [row(i, t + i * step, 0, w) for i, w in enumerate(wins)]
    rows.append(row(99, t + 5 * step + SESSION_GAP_S + 10, 0, True))
    df = build_matches(rows, HEROES)
    # streak is the streak going INTO each game
    assert df["streak"].tolist() == [0, 1, 2, -1, -2, 0]
    assert pd.isna(df["prev_win"].iloc[0]) and pd.isna(df["prev_win"].iloc[-1])


def test_game_in_day_counts_per_calendar_day():
    day = 86_400
    base = 1_700_000_000
    df = build_matches([row(1, base, 0, True), row(2, base + 4000, 0, True), row(3, base + day, 0, True)], HEROES)
    assert df["game_in_day"].tolist() == [1, 2, 1]
    assert df["games_that_day"].tolist() == [2, 2, 1]


@pytest.mark.parametrize("wins,n,expected_inside", [(50, 100, 50.0), (500, 1000, 50.0)])
def test_wilson_interval_contains_point_estimate(wins, n, expected_inside):
    from analysis.stats import wilson
    lo, hi = wilson(wins, n)
    assert lo < expected_inside < hi
    assert hi - lo < (25 if n == 100 else 8)  # more data -> tighter interval
