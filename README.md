# Dota 2 Match Analytics — what 6,853 games say about how I (and my friends) actually play

An end-to-end data project on my own Dota 2 history (Nov 2017 – Oct 2026): an **API → SQLite → statistical analysis → static dashboard** pipeline, built to answer a question every long-time player argues about:

> *Are the patterns I "feel" while playing — tilt, lucky hours, comfort heroes, playing with friends — real, or am I seeing faces in noise?*

I started by building a normal dashboard. The charts were full of convincing patterns, so I added the hypothesis-testing layer to find out which of them survive scrutiny. **Most don't — and learning to tell the difference was the actual point of the project.**

**Live dashboard:** *(enable GitHub Pages on the `docs/` folder and paste the link here)*

---

## Key findings

All tests are on all games unless noted. Effect sizes come with 95% CIs; I ran 7 headline tests, so the Bonferroni threshold is p < 0.007.

| Question | Method | Result |
|---|---|---|
| Am I better than a coin flip? | Exact binomial test, Wilson CI | **50.4%** over 6,853 games (CI 49.2–51.6%) — not different from 50% (matchmaking at work) |
| Does winning/losing the last game change my next game? | χ² → logistic regression with mode + year controls → dose-response | **Yes.** After a loss I win **53.2%**, after a win **48.3%** (OR for "won last game" = **0.82**, CI 0.72–0.92, **p = 0.0009**, survives Bonferroni). Opposite of "tilt" |
| Is that real or a fluke? | Robustness splits | Same direction in both modes and in **5 of 6 time periods**; **disappears when games are ≥30 min apart** (OR 1.04, p = 0.80) — what you'd expect if matchmaking rebalances back-to-back results |
| Do I get worse the more I play in a day? | Logistic regression on game-number-in-day | **No evidence** (OR 0.987 per extra game, p = 0.22) |
| Do I win more with a 3+ friend stack? | Logistic regression, party size ≥3 vs rest | 52.2% vs 49.3% solo, but **not significant** after controls (OR 1.07, p = 0.35) |
| Which heroes am I really good at? | Per-hero binomial tests, Benjamini–Hochberg FDR, empirical-Bayes shrinkage | 4 of 70 heroes look "significant" naively; **0 survive** multiple-testing correction. True spread between my heroes is only ≈2.2 pts (SD). My "best" hero, Viper (64.1% raw), shrinks to 52.9% |
| Is there a best time to play? | Permutation test (2,000 shuffles) on the hour × weekday heatmap | Hottest cell \|z\|=2.96 appears by chance **31%** of the time (p = 0.31). **The heatmap is noise** |
| Am I improving over time? | Logistic / OLS on years since first ranked game | No significant trend in win rate (p = 0.74) or KDA (+0.10/yr, p = 0.07) — win rate is a weak skill metric when matchmaking keeps adjusting opponents |

Full output (methods, CIs, caveats): [`docs/findings.json`](docs/findings.json), also rendered at the top of the dashboard.

---

## What I learned

**Statistics**
- **Dashboards manufacture patterns.** A 7×24 heatmap has 168 cells; some will always look "hot". A permutation test showed the most extreme cell is exactly what chance produces. Same story for heroes: 4 of 70 looked significant, which is about what 5% false positives predicts.
- **Multiple comparisons matter.** Without FDR correction I would have written "I'm great on Viper and terrible on Magnus." With it, neither claim is supportable.
- **Small samples lie.** Empirical-Bayes shrinkage (Beta prior estimated from the data itself) ranks heroes sensibly — 12 games at 75% no longer outranks 400 games at 56%.
- **Report effect size + uncertainty, not just p-values**, and write down the caveat: the momentum result is observational and has selection bias (I only see game N+1 if I chose to keep playing), so "matchmaking balances results" is a hypothesis consistent with the data, not a proven mechanism.

**Data engineering / quality** (each of these was a real bug I hit and fixed)
- **Silent API default dropped half my data.** OpenDota hides "non-significant" modes unless `significant=0` is passed — my first pull returned 3,429 games instead of 6,854 and every Turbo game was missing. I found it because the win/loss endpoint reported 3,429 games while the match list had 6,854 — the two totals disagreed.
- **Type fidelity across layers.** SQLite stores booleans as 0/1; the JavaScript front-end compared against real booleans and the dashboard briefly showed a **0% win rate**. Caught by actually loading the rendered page and checking the numbers, not just that it ran.
- **Missing data is not "solo".** 25.5% of games have no party-size field. Treating them as solo would have contaminated the friends analysis, so they are excluded from it explicitly and the exclusion is documented.
- **Metrics aren't comparable across contexts.** Turbo games average 26.5 min and 849 GPM vs 43.1 min and 457 GPM in ranked — pooling them would make "my GPM doubled" a meaningless statement. Mode is a control in every model.
- **Timezones.** All time features are computed once in the pipeline (US Eastern) and shipped to the browser as wall-clock strings, so charts don't change depending on where the viewer opens them.
- **Privacy.** Friends' names/IDs are replaced with `Friend 1..N` before publishing; the raw data never leaves `data/` (git-ignored).

---

## Architecture

```
OpenDota API ──► pipeline/extract.py ──► data/raw/*.json        (immutable raw layer, retry + backoff)
                                              │
                 pipeline/transform.py ◄──────┘   clean, dedupe, feature engineering (sessions, streaks, labels)
                        │
                        ▼
                  data/dota.db  (SQLite)  ◄── sql/queries.sql   (window functions, CTEs, aggregates)
                        │
                 analysis/stats.py   hypothesis tests ──► docs/findings.json
                        │
                 web/build.py        anonymise + render ──► docs/index.html, docs/matches.html  (GitHub Pages)
```

| Layer | Tech |
|---|---|
| Extraction | Python, `requests`, exponential-backoff retries |
| Storage / modelling | SQLite, pandas |
| Statistics | `scipy.stats`, `statsmodels` (logit / OLS), permutation tests, Benjamini–Hochberg |
| Visualisation | Static HTML + Chart.js (no server needed) |
| Quality | `pytest` (10 unit tests on the transformation logic: win flag, dedupe, sessions, streaks, missing data) |

## Run it yourself

```bash
pip install -r requirements.txt
# your account id = the number in your OpenDota/Dotabuff profile URL; enable "Expose Public Match Data" in Dota 2 settings
python -m pipeline.extract --account-id <YOUR_ID>
python -m pipeline.transform
python -m analysis.stats
python web/build.py
python -m pytest
# then open docs/index.html
```

## Repo layout

```
pipeline/   extract.py, transform.py      ETL
analysis/   stats.py                       statistical tests -> docs/findings.json
sql/        queries.sql                    example analytical SQL
web/        templates + build.py           static site generator
docs/       built site (GitHub Pages)
tests/      test_transform.py
```

## Limitations & next steps

- One player's history (plus teammates seen in it); results describe *me*, not Dota in general.
- Observational data — I can show that a pattern exists, not why. Replay-parsed data (lane, item timings, ward maps) exists for only a minority of old matches.
- The friends analysis only sees teammates that OpenDota recorded as "peers" (only 10 teammates recorded). Pulling each friend's own history would allow per-friend and synergy analysis.
- Ideas: hierarchical (mixed-effects) model for hero effects; modelling matchmaking explicitly; scheduling the pipeline (cron/Airflow) with incremental loads instead of full pulls; a Streamlit version.

*Data: [OpenDota API](https://docs.opendota.com/). Not affiliated with Valve.*
