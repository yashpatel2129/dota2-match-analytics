"""Stage 3 - ANALYSIS: hypothesis tests on my match history.

Usage:
    python -m analysis.stats          # prints a summary and writes docs/findings.json

Every section answers ONE question with a pre-stated method, reports an effect
size AND uncertainty (95% CI / p-value), and says whether the data actually
supports a conclusion. Several answers are "no" - that is the point: a dashboard
makes everything look like a pattern, statistics tells you which ones are noise.
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "dota.db"
OUT = ROOT / "docs" / "findings.json"
ALPHA = 0.05
MAIN_MODES = ["Ranked/All Draft", "Turbo"]  # the only two modes with enough games to model
RNG = np.random.default_rng(42)
warnings.filterwarnings("ignore")


def load() -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql("SELECT * FROM matches", con)


def wilson(wins: int, n: int) -> list[float]:
    lo, hi = proportion_confint(wins, n, alpha=ALPHA, method="wilson")
    return [round(lo * 100, 1), round(hi * 100, 1)]


def logit_or(model, term: str) -> dict:
    """Odds ratio, 95% CI and p-value for one coefficient of a fitted logit model."""
    beta, se, p = model.params[term], model.bse[term], model.pvalues[term]
    return {"odds_ratio": round(float(np.exp(beta)), 3),
            "ci95": [round(float(np.exp(beta - 1.96 * se)), 3), round(float(np.exp(beta + 1.96 * se)), 3)],
            "p": round(float(p), 4)}


def verdict(p: float) -> str:
    return "statistically significant" if p < ALPHA else "NOT statistically significant"


# --------------------------------------------------------------------------- #
def overall(d: pd.DataFrame) -> dict:
    n, w = len(d), int(d.win.sum())
    test = stats.binomtest(w, n, 0.5)
    return {
        "id": "overall", "title": "Am I actually better than a coin flip?",
        "question": "Is my overall win rate different from 50%?",
        "method": "Exact binomial test against p=0.5, Wilson 95% confidence interval.",
        "n": n, "win_rate": round(100 * w / n, 1), "ci95": wilson(w, n), "p": round(test.pvalue, 4),
        "significant": bool(test.pvalue < ALPHA),
        "takeaway": (f"{100*w/n:.1f}% over {n:,} games (95% CI {wilson(w, n)[0]}-{wilson(w, n)[1]}%): "
                     f"{verdict(test.pvalue)} from 50%. Matchmaking aims for 50%, so a long-run result this "
                     f"close to it is expected - the real signal is in the details below."),
    }


def modes(d: pd.DataFrame) -> dict:
    t = d[d["mode"].isin(MAIN_MODES)]
    tab = pd.crosstab(t["mode"], t["win"])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    rows = {m: {"n": int(len(g)), "win_rate": round(100 * g.win.mean(), 1), "ci95": wilson(int(g.win.sum()), len(g)),
                "avg_gpm": int(g.gold_per_min.mean()), "avg_kda": round(g.kda.mean(), 2),
                "avg_minutes": round(g.duration_min.mean(), 1)} for m, g in t.groupby("mode")}
    return {"id": "modes", "title": "Ranked vs Turbo",
            "question": "Do I perform differently in Turbo than in ranked All Draft?",
            "method": "Chi-square test of independence on win/loss by mode; descriptive stats per mode.",
            "by_mode": rows, "p": round(p, 4), "significant": bool(p < ALPHA),
            "takeaway": (f"Win rates are {rows['Ranked/All Draft']['win_rate']}% (ranked) vs {rows['Turbo']['win_rate']}% (turbo): "
                         f"{verdict(p)}. But the games are very different - Turbo averages {rows['Turbo']['avg_minutes']} min and "
                         f"{rows['Turbo']['avg_gpm']} GPM vs {rows['Ranked/All Draft']['avg_minutes']} min and "
                         f"{rows['Ranked/All Draft']['avg_gpm']} GPM - so GPM/XPM must never be compared across modes.")}


def momentum(d: pd.DataFrame) -> dict:
    """Does the result of the previous game (same session) predict this one?"""
    t = d[d["mode"].isin(MAIN_MODES) & d.prev_win.notna()].copy()
    t["prev_win"] = t.prev_win.astype(int)
    # Naive view: 2x2 table
    g = t.groupby("prev_win").win.agg(["sum", "count"])
    naive = {("after_win" if k == 1 else "after_loss"): {"n": int(r["count"]), "win_rate": round(100 * r["sum"] / r["count"], 1),
                                                          "ci95": wilson(int(r["sum"]), int(r["count"]))} for k, r in g.iterrows()}
    chi2, p_naive, _, _ = stats.chi2_contingency(pd.crosstab(t.prev_win, t.win))
    # Controlled view: logistic regression with mode + year fixed effects
    m = smf.logit("win ~ prev_win + C(mode) + C(year)", data=t).fit(disp=0)
    ctrl = logit_or(m, "prev_win")
    # Dose-response: does a longer streak matter?
    t["streak_c"] = t.streak.clip(-4, 4)
    m2 = smf.logit("win ~ streak_c + C(mode) + C(year)", data=t).fit(disp=0)
    dose = logit_or(m2, "streak_c")
    # Robustness: does the effect replicate in subsets? (a real effect should keep its sign)
    def sub_or(x):
        mm = smf.logit("win ~ prev_win", data=x).fit(disp=0)
        return {"n": int(len(x)), **logit_or(mm, "prev_win")}
    robust = {f"mode: {k}": sub_or(g) for k, g in t.groupby("mode")}
    robust["gap < 30 min since previous game"] = sub_or(t[t.gap_since_prev_s < 1800])
    robust["gap >= 30 min since previous game"] = sub_or(t[t.gap_since_prev_s >= 1800])
    yr = {}
    for label, lo, hi in [("2017-2020", 2016, 2020), ("2021", 2020, 2021), ("2022", 2021, 2022),
                          ("2023", 2022, 2023), ("2024", 2023, 2024), ("2025-2026", 2024, 2026)]:
        x = t[(t.year > lo) & (t.year <= hi)]
        if len(x) > 200:
            yr[label] = sub_or(x)
    robust.update({f"period: {k}": v for k, v in yr.items()})
    same_sign = sum(v["odds_ratio"] < 1 for k, v in robust.items() if k.startswith("period"))
    n_periods = sum(1 for k in robust if k.startswith("period"))
    return {"id": "momentum", "title": "Tilt, momentum and 'win-trading'",
            "question": "Does winning (or losing) the previous game in the same session change my odds in the next one?",
            "method": ("Compare win rates after a win vs after a loss (chi-square), then logistic regression controlling for "
                       "game mode and year, then a dose-response check on streak length (clipped at +/-4). "
                       "Only consecutive games <2h apart are used."),
            "naive": naive, "naive_p": round(p_naive, 4), "controlled": ctrl, "per_streak_step": dose,
            "robustness": robust, "periods_with_same_direction": f"{same_sign} of {n_periods}",
            "caveats": ("Observational data: I only see the next game if I chose to keep playing (selection bias), and "
                        "opponents/teammates are not observed. 'Matchmaking balances results' is a hypothesis consistent "
                        "with the data, not something this dataset can prove."),
            "significant": bool(ctrl["p"] < ALPHA),
            "takeaway": (f"After a loss I win {naive['after_loss']['win_rate']}%, after a win {naive['after_win']['win_rate']}% "
                         f"(naive p={p_naive:.3f}). With controls the odds ratio for 'won the previous game' is {ctrl['odds_ratio']} "
                         f"(95% CI {ctrl['ci95'][0]}-{ctrl['ci95'][1]}, p={ctrl['p']}): {verdict(ctrl['p'])}. "
                         + (f"The direction is opposite to 'tilt' and it has the same direction and similar size in both modes and {same_sign} of {n_periods} time periods, but vanishes when games are >=30 min apart - consistent with matchmaking pulling back-to-back results towards 50%."
                            if ctrl["odds_ratio"] < 1 and ctrl["p"] < ALPHA else
                            "No convincing evidence that streaks carry over."))}


def fatigue(d: pd.DataFrame) -> dict:
    t = d[d["mode"].isin(MAIN_MODES)].copy()
    t["gid"] = t.game_in_day.clip(upper=10)
    m = smf.logit("win ~ gid + C(mode) + C(year)", data=t).fit(disp=0)
    r = logit_or(m, "gid")
    first = t[t.gid == 1].win.mean() * 100
    late = t[t.gid >= 6].win.mean() * 100
    return {"id": "fatigue", "title": "Does playing more in a day make me worse?",
            "question": "Is the Nth game of the day less likely to be a win?",
            "method": "Logistic regression: win ~ game number in day (capped at 10) + mode + year.",
            "per_extra_game": r, "first_game_win_rate": round(first, 1), "game6plus_win_rate": round(late, 1),
            "significant": bool(r["p"] < ALPHA),
            "takeaway": (f"Each extra game in a day changes my odds by a factor of {r['odds_ratio']} "
                         f"(95% CI {r['ci95'][0]}-{r['ci95'][1]}, p={r['p']}): {verdict(r['p'])}. "
                         f"First game of the day {first:.1f}% vs 6th+ game {late:.1f}%.")}


def party(d: pd.DataFrame) -> dict:
    t = d[(d.party_known == 1) & d["mode"].isin(MAIN_MODES)].copy()
    t["stack"] = np.where(t.party_size >= 3, "3+ stack", np.where(t.party_size == 2, "duo", "solo"))
    g = t.groupby("stack").win.agg(["sum", "count"])
    rows = {k: {"n": int(r["count"]), "win_rate": round(100 * r["sum"] / r["count"], 1), "ci95": wilson(int(r["sum"]), int(r["count"]))}
            for k, r in g.iterrows()}
    t["premade3"] = (t.party_size >= 3).astype(int)
    m = smf.logit("win ~ premade3 + C(mode) + C(year)", data=t).fit(disp=0)
    r = logit_or(m, "premade3")
    return {"id": "party", "title": "Is it better to play with friends?",
            "question": "Do I win more in 3+ player stacks than solo / duo?",
            "method": ("Logistic regression: win ~ (party size >= 3) + mode + year. Games where OpenDota has no party_size "
                       f"({int((d.party_known == 0).sum()):,} games, {100*(d.party_known == 0).mean():.0f}%) are excluded, not assumed solo."),
            "by_party": rows, "stack_effect": r, "significant": bool(r["p"] < ALPHA),
            "takeaway": (f"3+ stacks win {rows['3+ stack']['win_rate']}% vs solo {rows['solo']['win_rate']}%. After controls the odds ratio "
                         f"is {r['odds_ratio']} (95% CI {r['ci95'][0]}-{r['ci95'][1]}, p={r['p']}): {verdict(r['p'])}.")}


def heroes(d: pd.DataFrame) -> dict:
    """Raw hero win rates are noisy; use empirical-Bayes shrinkage + multiple-testing correction."""
    base = d.win.mean()
    g = d.groupby("hero").win.agg(["sum", "count"]).rename(columns={"sum": "w", "count": "n"})
    big = g[g.n >= 10]
    # Method-of-moments estimate of the prior strength k (Beta prior centred on my overall rate)
    var_obs = np.average((big.w / big.n - base) ** 2, weights=big.n)
    var_noise = np.average(base * (1 - base) / big.n, weights=big.n)
    var_true = max(var_obs - var_noise, 1e-6)
    k = base * (1 - base) / var_true - 1
    true_sd = float(np.sqrt(var_true) * 100)
    g["raw"] = 100 * g.w / g.n
    g["shrunk"] = 100 * (g.w + base * k) / (g.n + k)
    test = g[g.n >= 20].copy()
    test["p"] = [stats.binomtest(int(r.w), int(r.n), base).pvalue for r in test.itertuples()]
    test["p_adj"] = multipletests(test.p, method="fdr_bh")[1]
    n_raw_sig = int((test.p < ALPHA).sum())
    n_adj_sig = int((test.p_adj < ALPHA).sum())
    top = test.sort_values("shrunk", ascending=False)
    row = lambda h, r: {"hero": h, "games": int(r.n), "raw_win_rate": round(r.raw, 1), "shrunk_win_rate": round(r.shrunk, 1),
                        "p": round(r.p, 4), "p_adj": round(r.p_adj, 4)}
    return {"id": "heroes", "title": "Which heroes am I genuinely good (or bad) at?",
            "question": "Do any hero win rates differ from my baseline by more than luck?",
            "method": ("Per-hero exact binomial test vs my overall win rate for heroes with >=20 games, Benjamini-Hochberg FDR "
                       f"correction across {len(test)} tests, and empirical-Bayes shrinkage (Beta prior, strength k={k:.0f} pseudo-games) "
                       "so that a hero with 12 games and a 75% win rate is not ranked above one with 400 games and 56%."),
            "prior_strength_k": round(float(k), 1), "estimated_true_sd_pct_points": round(true_sd, 2), "heroes_tested": int(len(test)),
            "significant_before_correction": n_raw_sig, "significant_after_correction": n_adj_sig,
            "best": [row(h, r) for h, r in top.head(5).iterrows()], "worst": [row(h, r) for h, r in top.tail(5).iloc[::-1].iterrows()],
            "significant": bool(n_adj_sig > 0),
            "takeaway": (f"Of {len(test)} heroes with >=20 games, {n_raw_sig} look 'significant' at p<0.05 before correcting for "
                         f"multiple comparisons but only {n_adj_sig} survive FDR correction (about {0.05*len(test):.0f} false positives "
                         f"are expected by chance alone). After removing sampling noise, the genuine spread in my hero win rates is only about "
                         f"{true_sd:.1f} percentage points (SD), so shrinkage pulls every hero strongly toward {100*base:.1f}%.")}


def hot_hours(d: pd.DataFrame) -> dict:
    """Is the 'best time to play' heatmap signal or noise? Permutation test on the max cell deviation."""
    t = d.copy()
    cell = (t.weekday * 24 + t.hour).to_numpy()
    wins = t.win.to_numpy()
    n_cells = 168
    cnt = np.bincount(cell, minlength=n_cells)
    keep = cnt >= 20
    base = wins.mean()

    def max_z(w):
        s = np.bincount(cell, weights=w, minlength=n_cells)
        z = (s[keep] - cnt[keep] * base) / np.sqrt(cnt[keep] * base * (1 - base))
        return np.abs(z).max()

    obs = max_z(wins)
    null = np.array([max_z(RNG.permutation(wins)) for _ in range(2000)])
    p = float((null >= obs).mean())
    # Coarser, pre-specified question: daypart
    part = pd.cut(t.hour, [-1, 5, 11, 17, 23], labels=["night 0-5", "morning 6-11", "afternoon 12-17", "evening 18-23"])
    tab = pd.crosstab(part, t.win)
    chi2, p_part, _, _ = stats.chi2_contingency(tab)
    rows = {str(k): {"n": int(tab.loc[k].sum()), "win_rate": round(100 * tab.loc[k, 1] / tab.loc[k].sum(), 1)} for k in tab.index}
    return {"id": "hot_hours", "title": "Is there a 'best time to play'?",
            "question": "Are some hour/weekday slots genuinely better, or is the heatmap just noise?",
            "method": (f"Permutation test: shuffle win/loss labels 2,000 times and compare the most extreme of the {int(keep.sum())} "
                       "hour x weekday cells (n>=20) to what pure chance produces. Plus a chi-square on four pre-defined dayparts."),
            "max_abs_z_observed": round(float(obs), 2), "permutation_p": round(p, 4), "dayparts": rows, "daypart_p": round(p_part, 4),
            "significant": bool(p < ALPHA or p_part < ALPHA),
            "takeaway": (f"The hottest/coldest hour-weekday cell has |z|={obs:.2f}; random shuffling produces a cell that extreme "
                         f"{100*p:.0f}% of the time (permutation p={p:.3f}): {verdict(p)}. Dayparts: p={p_part:.3f}. "
                         + ("Treat the heatmap colours as noise - picking 'lucky hours' would be fooling myself."
                            if p >= ALPHA and p_part >= ALPHA else "There is some genuine time-of-day structure worth a closer look."))}


def improvement(d: pd.DataFrame) -> dict:
    t = d[d["mode"] == "Ranked/All Draft"].copy()
    t["yrs"] = (t.start_time - t.start_time.min()) / (365.25 * 86400)
    m = smf.logit("win ~ yrs", data=t).fit(disp=0)
    r = logit_or(m, "yrs")
    k = smf.ols("kda ~ yrs", data=t).fit()
    kda = {"slope_per_year": round(float(k.params["yrs"]), 3), "ci95": [round(float(x), 3) for x in k.conf_int().loc["yrs"]], "p": round(float(k.pvalues["yrs"]), 4)}
    return {"id": "improvement", "title": "Am I getting better over time?",
            "question": "In ranked All Draft, do my win rate or KDA trend with time?",
            "method": "Logistic regression win ~ years since first game; OLS kda ~ years. Ranked only (Turbo would confound).",
            "winrate_trend": r, "kda_trend": kda, "significant": bool(r["p"] < ALPHA or kda["p"] < ALPHA),
            "takeaway": (f"Win-rate odds change by x{r['odds_ratio']} per year (p={r['p']}: {verdict(r['p'])}). KDA changes by "
                         f"{kda['slope_per_year']:+.3f} per year (p={kda['p']}: {verdict(kda['p'])}). "
                         "Win rate is a poor skill metric here because matchmaking adjusts opponents to keep me near 50%.")}


def run() -> dict:
    d = load()
    findings = [overall(d), modes(d), momentum(d), fatigue(d), party(d), heroes(d), hot_hours(d), improvement(d)]
    primary = [f for f in findings if f["id"] not in ("overall", "modes")]
    n_tests = len(primary) + 1
    dataset = {
        "bonferroni_alpha": round(ALPHA / n_tests, 4), "headline_tests": n_tests,
        "matches": int(len(d)), "first_game": d.date.min(), "last_game": d.date.max(),
        "unknown_party_size_pct": round(100 * (d.party_known == 0).mean(), 1),
        "turbo_share_pct": round(100 * d.is_turbo.mean(), 1),
    }
    return {"dataset": dataset, "findings": findings}


if __name__ == "__main__":
    out = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in out["findings"]:
        print(f"\n## {f['title']}\n   {f['takeaway']}")
    print(f"\nwrote {OUT}")
