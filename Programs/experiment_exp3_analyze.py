
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 3 Analyzer — Dynamic p responsiveness
------------------------------------------------
Metrics:
- Elasticity: slope beta in OLS y_t = alpha + beta * p_t
  * For PD, y_t = 1(JJ); for BoS, y_t = 1(JJ or FF)
- Lag: argmax cross-correlation lag between p_t and y_t over lags 0..LAG_MAX
- Overshoot: max residual |y_t - yhat_t| in a window around the path midpoint (for up/down)
- Re-equilibration time: earliest t after midpoint with |y_t - yhat_t| <= eps for M consecutive rounds
- BoS switching cost & success: detection of switch from single-point focus (JJ-only or FF-only) to rotation JJ<->FF;
  * cost = # of mismatched (0-payoff) rounds during transition
  * success = 1 if rotation holds for >= R rounds after switch else 0
Outputs:
- match_level_metrics_exp3.csv
- summary_metrics_exp3.csv
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

IN_CSV = "experiment_exp3.csv"
OUT_MATCH = "match_level_metrics_exp3.csv"
OUT_SUMMARY = "summary_metrics_exp3.csv"

LAG_MAX = 10
EPS = 0.1
M_STABLE = 5
R_ROTATE = 6   # need at least 6 rounds (JJFFJJ...) to count as success

def is_rotation_JJ_FF(seq: List[str]) -> bool:
    if len(seq) < 2: return False
    a, b = seq[0], seq[1]
    if not ({a,b} == {"JJ","FF"}): return False
    for i, s in enumerate(seq):
        exp = a if (i % 2 == 0) else b
        if s != exp: return False
    return True

def analyze_one_match(dfm: pd.DataFrame):
    game = dfm["game_type"].iloc[0]
    path = dfm["path"].iloc[0]

    p = dfm["p_now"].to_numpy(dtype=float)
    pairs = [f"{a}{b}" for a,b in zip(dfm["answer1"], dfm["answer2"])]
    y = np.array([1.0 if ((pair=="JJ") if game=="PD" else (pair in ("JJ","FF"))) else 0.0 for pair in pairs], dtype=float)

    # Elasticity via OLS slope
    if len(p) >= 2 and len(np.unique(p)) >= 2:
        beta, alpha = np.polyfit(p, y, 1)  # y ≈ beta*p + alpha
    else:
        beta, alpha = np.nan, np.nan

    # Lag via cross-correlation (non-negative lags)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(0, min(LAG_MAX, len(y)-1) + 1):
        yp = y[lag:]
        pp = p[:len(yp)]
        if len(yp) > 1 and np.std(yp)>1e-8 and np.std(pp)>1e-8:
            corr = np.corrcoef(pp, yp)[0,1]
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
    lag_rounds = float(best_lag)

    # Predicted yhat for residual-based measures
    if not np.isnan(beta):
        yhat = beta * p + alpha
        yhat = np.clip(yhat, 0.0, 1.0)
    else:
        yhat = np.full_like(y, np.nan)

    # Midpoint index for up/down, else center for rw
    n = len(y)
    mid = n//2

    # Overshoot: max |y - yhat| near midpoint ±W
    W = max(3, n//5)
    lo = max(0, mid - W)
    hi = min(n, mid + W)
    overshoot = float(np.nanmax(np.abs(y[lo:hi] - yhat[lo:hi])) if n>0 else np.nan)

    # Re-equilibration time: first t >= mid with |y - yhat| <= EPS for M consecutive
    reeq_time = np.nan
    for t in range(mid, n - M_STABLE + 1):
        window_ok = np.all(np.abs(y[t:t+M_STABLE] - yhat[t:t+M_STABLE]) <= EPS)
        if window_ok:
            reeq_time = float(t+1)  # 1-indexed
            break

    # BoS-specific switching
    bos_switch_cost = np.nan
    bos_switch_success = np.nan
    if game == "BoS":
        # detect earliest t where from t onwards sequence is JJ<->FF rotation
        # also detect if before t there was single-point focus (all JJ or all FF for >=3 rounds)
        found = False
        for t in range(0, n-1):
            if is_rotation_JJ_FF(pairs[t:]):
                # cost: count zero-payoff rounds from last single-focus block till t
                # identify last contiguous block of same NE before t
                k = t-1
                while k >= 0 and pairs[k] in ("JJ","FF"):
                    k -= 1
                pre_block = pairs[k+1:t]
                single_focus = len(pre_block) >= 3 and len(set(pre_block)) == 1
                cost = sum(1 for s in pairs[k+1:t] if s not in ("JJ","FF"))
                bos_switch_cost = float(cost) if single_focus else 0.0
                # success: rotation holds for >= R_ROTATE
                bos_switch_success = 1.0 if len(pairs[t:]) >= R_ROTATE else 0.0
                found = True
                break
        if not found:
            bos_switch_cost = 0.0
            bos_switch_success = 0.0

    welfare_total = float(dfm["points1"].sum() + dfm["points2"].sum())
    welfare_per_round = welfare_total / n if n>0 else np.nan

    return {
        "rounds": float(n),
        "elasticity_beta": float(beta),
        "lag_rounds": lag_rounds,
        "overshoot": overshoot,
        "reequilibration_time": reeq_time,
        "welfare_total": welfare_total,
        "welfare_per_round": welfare_per_round,
        "bos_switch_cost": bos_switch_cost,
        "bos_switch_success": bos_switch_success,
    }

def main():
    df = pd.read_csv(IN_CSV)
    keys = ["game_type","path","L_window","reasoning_mode","pairing_type","player1","player2","seed","match_id"]
    rows = []
    for k, g in df.groupby(keys):
        g = g.sort_values("round")
        metrics = analyze_one_match(g)
        rec = dict(zip(keys, k))
        rec.update(metrics)
        rows.append(rec)
    match_df = pd.DataFrame(rows)
    match_df.to_csv(OUT_MATCH, index=False)

    # Summaries by condition (collapse players to compare modes & paths & L windows)
    sum_keys = ["game_type","path","L_window","reasoning_mode","pairing_type"]
    agg = {
        "rounds":"mean",
        "elasticity_beta":"mean",
        "lag_rounds":"mean",
        "overshoot":"mean",
        "reequilibration_time":"mean",
        "welfare_total":"mean",
        "welfare_per_round":"mean",
        "bos_switch_cost":"mean",
        "bos_switch_success":"mean",
    }
    summary = match_df.groupby(sum_keys, dropna=False).agg(agg).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved match-level metrics to {OUT_MATCH}")
    print(f"Saved summary metrics to {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
