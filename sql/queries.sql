-- Example analytical queries against data/dota.db (SQLite).
-- Run:  sqlite3 data/dota.db < sql/queries.sql      (or open the DB in DBeaver / DB Browser)

-- 1) Win rate and games by mode ------------------------------------------------
SELECT mode,
       COUNT(*)                          AS games,
       ROUND(100.0 * AVG(win), 1)        AS win_pct,
       ROUND(AVG(duration_min), 1)       AS avg_minutes,
       ROUND(AVG(gold_per_min))          AS avg_gpm
FROM matches
GROUP BY mode
ORDER BY games DESC;

-- 2) Momentum: win rate right after a win vs after a loss (same session) -----------
SELECT CASE prev_win WHEN 1 THEN 'after win' WHEN 0 THEN 'after loss' END AS context,
       COUNT(*)                   AS games,
       ROUND(100.0 * AVG(win), 1) AS win_pct
FROM matches
WHERE prev_win IS NOT NULL
GROUP BY prev_win;

-- 3) Window function: rolling 100-game win rate, sampled every 250 games ---------------
WITH r AS (
  SELECT date,
         ROW_NUMBER() OVER (ORDER BY start_time) AS n,
         AVG(win) OVER (ORDER BY start_time ROWS BETWEEN 99 PRECEDING AND CURRENT ROW) AS roll100
  FROM matches
)
SELECT n, date, ROUND(100 * roll100, 1) AS rolling_win_pct
FROM r
WHERE n >= 100 AND n % 250 = 0;

-- 4) Hero pool: min 20 games, with a Wilson-style lower bound (more honest than raw %) -----
WITH h AS (
  SELECT hero, COUNT(*) AS n, SUM(win) AS w FROM matches GROUP BY hero HAVING COUNT(*) >= 20
)
SELECT hero, n,
       ROUND(100.0 * w / n, 1) AS raw_win_pct,
       ROUND(100.0 * ((1.0*w/n + 1.96*1.96/(2*n) - 1.96*SQRT((1.0*w/n*(1-1.0*w/n) + 1.96*1.96/(4*n))/n)) / (1 + 1.96*1.96/n)), 1) AS wilson_lower_95
FROM h
ORDER BY wilson_lower_95 DESC
LIMIT 10;

-- 5) Heaviest play days and how they ended -------------------------------------------------
SELECT date, COUNT(*) AS games, SUM(win) AS wins, COUNT(*) - SUM(win) AS losses
FROM matches
GROUP BY date
ORDER BY games DESC
LIMIT 10;

-- 6) Sessions: how long did I play in a sitting and how did it end? ------------------------
SELECT session_id,
       MIN(played_at)                         AS started,
       COUNT(*)                               AS games,
       ROUND(SUM(duration) / 3600.0, 1)       AS hours,
       SUM(win)                               AS wins
FROM matches
GROUP BY session_id
ORDER BY games DESC
LIMIT 10;
