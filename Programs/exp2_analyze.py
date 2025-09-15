
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 2 Analyzer — Metrics & Comparisons
---------------------------------------------
Reads experiment_exp2.csv and produces:
- Last-round & last-k (k=3,5) cooperation (PD) / coordination (BoS)
- Time to stability (1-cycle or 2-cycle)
- Equilibrium hit rate:
    * PD: DD at end / stabilized to DD
    * BoS: NE (JJ or FF) at end / stabilized to NE / stabilized rotation (JJ↔FF)
- Welfare: total and per-round average
Outputs:
- match_level_metrics.csv
- summary_metrics.csv (grouped by game_type, horizon_type, param (T/p), reasoning_mode, pairing_type)
- (Optional) simple plots (saved as PNG)
"""

import math
import pandas as pd
import numpy as np
from typing import List, Tuple

IN_CSV = "experiment_exp2.csv"
OUT_MATCH = "match_level_metrics.csv"
OUT_SUMMARY = "summary_metrics.csv"

def pair_to_str(a1, a2):
    return f"{a1}{a2}"

def last_k_rate(pairs: List[str], target: str, k: int) -> float:
    if len(pairs) == 0:
        return float("nan")
    if k > len(pairs):
        k = len(pairs)
    tail = pairs[-k:]
    return np.mean([1.0 if p == target else 0.0 for p in tail])

def is_rotation_JJ_FF(subseq: List[str]) -> bool:
    if len(subseq) < 2: return False
    a, b = subseq[0], subseq[1]
    if not ({a, b} == {"JJ", "FF"}): return False
    for i, s in enumerate(subseq):
        expect = a if (i % 2 == 0) else b
        if s != expect:
            return False
    return True

def time_to_stability(pairs: List[str]) -> float:
    """
    Earliest t (1-indexed rounds) such that from t..end the sequence is stable:
    - 1-cycle (constant pair), or
    - 2-cycle JJ<->FF (for BoS fairness rotation), or
    - for generality, any 2-cycle XY<->XY repeats (we limit to JJ/FF as spec).
    Returns NaN if never stabilizes.
    """
    n = len(pairs)
    if n <= 1: return float("nan")
    # 1-cycle
    for t in range(n-1):
        if all(p == pairs[t] for p in pairs[t:]):
            return t+1  # 1-indexed
    # 2-cycle JJ<->FF
    for t in range(n-2):
        subseq = pairs[t:]
        if is_rotation_JJ_FF(subseq):
            return t+1
    return float("nan")

def analyze_match(df_match: pd.DataFrame):
    game = df_match["game_type"].iloc[0]
    pairs = [pair_to_str(a1,a2) for a1,a2 in zip(df_match["answer1"], df_match["answer2"])]
    welfare_total = float(df_match["points1"].sum() + df_match["points2"].sum())
    rounds = len(pairs)
    welfare_per_round = welfare_total / rounds if rounds>0 else float("nan")
    last_pair = pairs[-1] if rounds>0 else None

    # PD metrics: cooperation = JJ
    coop_last = float(last_pair == "JJ") if last_pair is not None else float("nan")
    coop_last3 = last_k_rate(pairs, "JJ", 3)
    coop_last5 = last_k_rate(pairs, "JJ", 5)

    # BoS metrics: coordination = JJ or FF
    coord_last = float(last_pair in ("JJ","FF")) if last_pair is not None else float("nan")
    coord_last3 = np.mean([1.0 if p in ("JJ","FF") else 0.0 for p in pairs[-min(3,rounds):]]) if rounds>0 else float("nan")
    coord_last5 = np.mean([1.0 if p in ("JJ","FF") else 0.0 for p in pairs[-min(5,rounds):]]) if rounds>0 else float("nan")

    # Equilibrium hits
    ended_in_DD = float(last_pair == "FF") if last_pair is not None else float("nan")  # PD
    ended_in_NE = float(last_pair in ("JJ","FF")) if last_pair is not None else float("nan")  # BoS

    # Stabilization
    t_stab = time_to_stability(pairs)
    stabilized_to_DD = float(not math.isnan(t_stab) and all(p=="FF" for p in pairs[int(t_stab-1):]))
    stabilized_to_NE = float(not math.isnan(t_stab) and all(p in ("JJ","FF") for p in pairs[int(t_stab-1):]))
    stabilized_rotation = 0.0
    if not math.isnan(t_stab):
        if is_rotation_JJ_FF(pairs[int(t_stab-1):]):
            stabilized_rotation = 1.0

    return {
        "rounds": rounds,
        "welfare_total": welfare_total,
        "welfare_per_round": welfare_per_round,
        "coop_last": coop_last,
        "coop_last3": coop_last3,
        "coop_last5": coop_last5,
        "coord_last": coord_last,
        "coord_last3": coord_last3,
        "coord_last5": coord_last5,
        "ended_in_DD": ended_in_DD,
        "ended_in_NE": ended_in_NE,
        "t_stability": t_stab,
        "stabilized_to_DD": stabilized_to_DD,
        "stabilized_to_NE": stabilized_to_NE,
        "stabilized_rotation": stabilized_rotation,
    }

def main():
    df = pd.read_csv(IN_CSV)
    group_keys = ["game_type","horizon_type","T","p","E_T","reasoning_mode","pairing_type","player1","player2","seed","match_id"]
    match_rows = []
    for keys, g in df.groupby(group_keys):
        metrics = analyze_match(g.sort_values("round"))
        rec = dict(zip(group_keys, keys))
        rec.update(metrics)
        match_rows.append(rec)
    match_df = pd.DataFrame(match_rows)
    match_df.to_csv(OUT_MATCH, index=False)

    # Summary (aggregate by condition; collapse over specific players to compare modes/horizons)
    summary_keys = ["game_type","horizon_type","T","p","E_T","reasoning_mode","pairing_type"]
    agg = {
        "rounds":"mean",
        "welfare_total":"mean",
        "welfare_per_round":"mean",
        "coop_last":"mean",
        "coop_last3":"mean",
        "coop_last5":"mean",
        "coord_last":"mean",
        "coord_last3":"mean",
        "coord_last5":"mean",
        "ended_in_DD":"mean",
        "ended_in_NE":"mean",
        "t_stability":"mean",
        "stabilized_to_DD":"mean",
        "stabilized_to_NE":"mean",
        "stabilized_rotation":"mean",
    }
    summary = match_df.groupby(summary_keys, dropna=False).agg(agg).reset_index()
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved match-level metrics to {OUT_MATCH}")
    print(f"Saved summary metrics to {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
