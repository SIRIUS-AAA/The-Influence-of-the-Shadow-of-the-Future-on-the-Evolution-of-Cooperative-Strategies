
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 4 — Modular SCoT (Social Chain-of-Thought) Switches
--------------------------------------------------------------
Adds modular SCoT controls and richer logging to a SQLite DB.

Parameters / Switches
- use_scot (master)
- scot_P : Prediction (produce predicted probability of opponent choosing Cooperation J)
- scot_J : Long-term plan / rotation plan (per-round brief plan text)
- scot_R : Reflection before action (per-round brief reflection text)
- cot_enabled : keep plain CoT as separate flag (non-social)

Schema
- games(game_id, timestamp, game_type, horizon_type, p_mode, p_trace, use_scot, scot_P, scot_J, scot_R, cot_enabled,
         player1, player2, reasoning_mode, T, p_fixed, E_T, seed, match_id)
- rounds(id, game_id, round, p_now, answer1, answer2, points1, points2, total1, total2,
         agent1_pred_prob_C, agent2_pred_prob_C,
         agent1_plan, agent2_plan,
         agent1_reflection, agent2_reflection)

This script contains:
- Minimal agents (always-J / always-F) and placeholders for LLM calls
- Runner for fixed horizon or continuation probability p (supports static p or a p_trace list/mode)
- Interactive CLI menu (optional) and programmatic run() API
"""

import os
import json
import time
import math
import random
import sqlite3
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Tuple, Dict, Any

from openai import OpenAI
import numpy as np
import pandas as pd



OPENAI_API_KEY = "sk-vWPlVDMJizXLti9l69ah2W5l0bdGKxwh1OnBBRuhwtPJL4ed"
DEEPSEEK_API_KEY = "sk-cb387c428d9343328cea734e6ae0f9f5"
ANTHROPIC_API_KEY = ''
MOONSHOT_API_KEY = 'sk-8dW8q949DUpVaWjKtATKUovMvabr0VlU0cE3WTFYsdu0Ugwz'
DOUBAO_API_KEY = '068a6b39-c13c-4df2-8253-7d5bfe1cca51'




def api_request_openai_chat(model, max_tokens, temperature, messages, max_retries=5, base_delay=1.0, backoff_factor=2.0):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.chatanywhere.tech/v1/chat/completions")
    retries = 0
    while retries <= max_retries:
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return resp
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(base_delay * (backoff_factor ** (retries - 1)))




def api_request_doubao(model, messages, max_tokens=8, temperature=0.0):
    client = OpenAI(api_key=DOUBAO_API_KEY, base_url="https://ark.cn-beijing.volces.com/api/v3",)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return resp


def api_request_deepseek(model, messages, max_tokens=8, temperature=0.0):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return resp

def api_request_kimi(model, messages, max_tokens=8, temperature=0.0):
    client = OpenAI(api_key=MOONSHOT_API_KEY, base_url="https://api.moonshot.ai/v1")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    return resp



def act_gpt4(prompt: str, _: int) -> str:
    try:
        messages=[{"role": "user", "content": prompt}]
        resp = api_request_openai_chat("gpt-4", max_tokens=4, temperature=0.0, messages=messages)
        return resp.choices[0].message.content
    except Exception:
        return "J"

def act_deepseek(prompt: str, _: int) -> str:
    try:
        messages=[{"role": "user", "content": prompt}]
        resp = api_request_deepseek("deepseek-chat", max_tokens=4, temperature=0.0, messages=messages)
        return resp.choices[0].message.content
    except Exception:
        return "J"
def act_kimi(prompt: str, _: int) -> str:
    try:
        messages=[{"role": "user", "content": prompt}]
        resp = api_request_kimi("kimi-k2-0711-preview", max_tokens=4, temperature=0.0, messages=messages)
        return resp.choices[0].message.content
    except Exception:
        return "J"
    
def act_doubao(prompt: str, _: int) -> str:
    try:
        messages=[{"role": "user", "content": prompt}]
        resp = api_request_doubao("doubao-seed-1-6-250615", max_tokens=4, temperature=0.0, messages=messages)
        return resp.choices[0].message.content
    except Exception:
        return "J"




# -------------------- Agents (stubs + heuristics) --------------------
def act_J(prompt: str, _: int) -> str: return "J"
def act_F(prompt: str, _: int) -> str: return "F"

# (You can plug your LLM functions here; must return raw string that contains J/F)
AGENTS = {
    "act_doubao,":act_doubao,
    "act_kimi,":act_kimi,
    "act_deepseek,":act_deepseek,
    "act_gpt4,":act_gpt4,
    "act_J": act_J,
    "act_F": act_F,
    # "act_gpt4": act_gpt4, ...
}

# -------------------- Payoffs --------------------
def payoff(game_type: str, a1: str, a2: str) -> Tuple[int,int]:
    if game_type == "PD":
        if a1 == "J" and a2 == "J": return 8, 8
        if a1 == "J" and a2 == "F": return 0, 10
        if a1 == "F" and a2 == "J": return 10, 0
        if a1 == "F" and a2 == "F": return 5, 5
    elif game_type == "BoS":
        if a1 == a2 == "J": return 10, 7
        if a1 == a2 == "F": return 7, 10
        if a1 != a2: return 0, 0
    return -9999, -9999

def parse_choice(text: str) -> str:
    if not isinstance(text, str): return "J"
    t = text.strip().upper()
    for k in ["OPTION J", " ANSWER J", " CHOOSE J", " A: OPTION J", " A: J"]:
        if k in t: return "J"
    for k in ["OPTION F", " ANSWER F", " CHOOSE F", " A: OPTION F", " A: F"]:
        if k in t: return "F"
    for ch in reversed(t):
        if ch in ("J","F"): return ch
    return "J"

# -------------------- SCoT Helpers --------------------
def scot_prediction_prob_C(history_summary: str, context: str) -> float:
    """
    Placeholder: predict probability opponent cooperates (J).
    In real use, query an LLM for a number in [0,1].
    Here we use a simple heuristic based on last token in summary.
    """
    if "Last pair=JJ" in history_summary: return 0.8
    if "Last pair=FF" in history_summary: return 0.2
    return 0.5

def scot_plan_text(game_type: str, round_i: int, history_summary: str) -> str:
    """
    Placeholder plan text. Replace with LLM-generated long-term/rotation plan.
    """
    if game_type == "BoS":
        return f"Round {round_i}: aim to coordinate; consider alternating JJ/FF if fairness required."
    else:
        return f"Round {round_i}: sustain cooperation (JJ) unless opponent defects repeatedly."

def scot_reflection_text(game_type: str, round_i: int, p_now: float, pred_prob_C: Optional[float]) -> str:
    """
    Placeholder reflection text. Replace with LLM reflection.
    """
    base = f"Round {round_i}: p={p_now:.2f}. "
    if pred_prob_C is not None:
        base += f"Predict opp coop={pred_prob_C:.2f}. "
    if game_type == "PD":
        return base + "Weigh long-term payoff of JJ vs temptation of one-shot F."
    return base + "Seek focal point; prefer my-payoff-favored NE but avoid miscoordination."

def build_history_summary(pairs: List[str], L: Optional[int]) -> str:
    if not pairs: return "Recent 0 rounds. Last pair=NA."
    window = pairs if (L is None or L < 0) else pairs[-min(L, len(pairs)):]
    cJJ = sum(1 for x in window if x=="JJ")
    cFF = sum(1 for x in window if x=="FF")
    cJF = sum(1 for x in window if x=="JF")
    cFJ = sum(1 for x in window if x=="FJ")
    last = window[-1] if window else "NA"
    return f"Recent {len(window)} rounds summary: JJ={cJJ}, FF={cFF}, JF={cJF}, FJ={cFJ}. Last pair={last}."

# -------------------- Config Dataclasses --------------------
@dataclass
class SCoTSwitches:
    use_scot: bool = False
    scot_P: bool = False
    scot_J: bool = False
    scot_R: bool = False
    cot_enabled: bool = False  # plain CoT (non-social), retained separately

@dataclass
class GameConfig:
    game_type: str = "PD"               # "PD" or "BoS"
    horizon_type: str = "fixed"         # "fixed" or "continuation"
    T: Optional[int] = 10               # used if fixed
    p_mode: str = "fixed"               # "fixed", "trace", or "dynamic"
    p_fixed: Optional[float] = None     # used if p_mode == "fixed"
    p_trace: Optional[List[float]] = None  # used if p_mode == "trace" or "dynamic"
    max_round: int = 100
    reasoning_mode: str = "baseline"    # "baseline" or "scot"
    L_window: int = -1                  # -1 for infinite history in summaries

@dataclass
class MatchMeta:
    player1: str
    player2: str
    seed: int = 0
    match_id: int = 0

# -------------------- DB Layer --------------------
DDL_GAMES = """
CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    game_type TEXT,
    horizon_type TEXT,
    p_mode TEXT,
    p_trace TEXT,
    use_scot INTEGER,
    scot_P INTEGER,
    scot_J INTEGER,
    scot_R INTEGER,
    cot_enabled INTEGER,
    player1 TEXT,
    player2 TEXT,
    reasoning_mode TEXT,
    T INTEGER,
    p_fixed REAL,
    E_T REAL,
    seed INTEGER,
    match_id INTEGER
);
"""

DDL_ROUNDS = """
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER,
    round INTEGER,
    p_now REAL,
    answer1 TEXT,
    answer2 TEXT,
    points1 INTEGER,
    points2 INTEGER,
    total1 INTEGER,
    total2 INTEGER,
    agent1_pred_prob_C REAL,
    agent2_pred_prob_C REAL,
    agent1_plan TEXT,
    agent2_plan TEXT,
    agent1_reflection TEXT,
    agent2_reflection TEXT,
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);
"""

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(DDL_GAMES)
    cur.execute(DDL_ROUNDS)
    conn.commit()
    return conn

# -------------------- Runner --------------------
def generate_round_indices(cfg: GameConfig) -> List[int]:
    if cfg.horizon_type == "fixed":
        return list(range(1, cfg.T + 1))
    # continuation horizon: controlled by p_trace stopping; still provide 1..max_round
    return list(range(1, cfg.max_round + 1))

def choose_action(agent_fn: Callable, prompt: str, round_idx: int) -> str:
    raw = agent_fn(prompt, round_idx-1)
    return parse_choice(raw)

def run_match(db_path: str, cfg: GameConfig, scot: SCoTSwitches, meta: MatchMeta):
    # Prepare DB
    conn = init_db(db_path)
    cur = conn.cursor()

    # Compute E[T] if continuation with fixed p
    E_T = None
    if cfg.horizon_type == "continuation" and cfg.p_mode == "fixed" and cfg.p_fixed is not None and 0 < cfg.p_fixed < 1:
        E_T = 1.0 / (1.0 - cfg.p_fixed)

    # Insert game row
    game_row = {
        "timestamp": int(time.time()),
        "game_type": cfg.game_type,
        "horizon_type": cfg.horizon_type,
        "p_mode": cfg.p_mode,
        "p_trace": json.dumps(cfg.p_trace) if cfg.p_trace is not None else None,
        "use_scot": int(scot.use_scot),
        "scot_P": int(scot.scot_P),
        "scot_J": int(scot.scot_J),
        "scot_R": int(scot.scot_R),
        "cot_enabled": int(scot.cot_enabled),
        "player1": meta.player1,
        "player2": meta.player2,
        "reasoning_mode": cfg.reasoning_mode,
        "T": cfg.T,
        "p_fixed": cfg.p_fixed,
        "E_T": E_T,
        "seed": meta.seed,
        "match_id": meta.match_id,
    }
    placeholders = ",".join(["?"]*len(game_row))
    cur.execute(f"INSERT INTO games ({','.join(game_row.keys())}) VALUES ({placeholders})", list(game_row.values()))
    game_id = cur.lastrowid
    conn.commit()

    random.seed(meta.seed); np.random.seed(meta.seed)

    # pick agents
    agent1 = AGENTS[meta.player1]
    agent2 = AGENTS[meta.player2]

    totals = [0,0]
    pairs: List[str] = []
    rounds_idx = generate_round_indices(cfg)

    # Build static rule text (simplified); you can expand with your prior prompt templates
    def rule_text_for(player: int) -> str:
        if cfg.game_type == "PD":
            return ("You are playing a repeated Prisoner's Dilemma.\n"
                    "Payoffs: JJ->(8,8), JF->(0,10), FJ->(10,0), FF->(5,5).\n")
        else:
            if player == 1:
                return ("You are playing a repeated Battle of the Sexes.\n"
                        "Payoffs (you, other): JJ->(10,7), FF->(7,10), mismatch->(0,0).\n")
            else:
                return ("You are playing a repeated Battle of the Sexes.\n"
                        "Payoffs (you, other): JJ->(7,10), FF->(10,7), mismatch->(0,0).\n")

    for i in rounds_idx:
        # Determine p_now and continuation stopping
        if cfg.horizon_type == "fixed":
            p_now = None
        else:
            # continuation: p from fixed value or from trace
            if cfg.p_mode == "fixed":
                p_now = float(cfg.p_fixed)
            else:
                # trace/dynamic path provided
                if cfg.p_trace is None or len(cfg.p_trace) < i:
                    p_now = float(cfg.p_trace[-1]) if cfg.p_trace else 0.5
                else:
                    p_now = float(cfg.p_trace[i-1])

        # Build history summary for SCoT
        history_summary = build_history_summary(pairs, cfg.L_window if cfg.reasoning_mode == "scot" or scot.use_scot else 0)

        # SCoT fields default
        a1_pred = a2_pred = None
        a1_plan = a2_plan = None
        a1_refl = a2_refl = None

        # SCoT prediction (P)
        if scot.use_scot and scot.scot_P:
            a1_pred = scot_prediction_prob_C(history_summary, "p_now" if p_now is not None else "fixed horizon")
            a2_pred = scot_prediction_prob_C(history_summary, "p_now" if p_now is not None else "fixed horizon")

        # SCoT plan (J)
        if scot.use_scot and scot.scot_J:
            a1_plan = scot_plan_text(cfg.game_type, i, history_summary)
            a2_plan = scot_plan_text(cfg.game_type, i, history_summary)

        # SCoT reflection (R)
        if scot.use_scot and scot.scot_R:
            a1_refl = scot_reflection_text(cfg.game_type, i, p_now if p_now is not None else float('nan'), a1_pred)
            a2_refl = scot_reflection_text(cfg.game_type, i, p_now if p_now is not None else float('nan'), a2_pred)

        # Build prompts
        ptxt = "" if p_now is None else f"Continuation probability this round p={p_now:.2f}.\n"
        prompt1 = rule_text_for(1) + ptxt + history_summary + "\nChoose your action (Option J or Option F). A: Option"
        prompt2 = rule_text_for(2) + ptxt + history_summary + "\nChoose your action (Option J or Option F). A: Option"

        # CoT-only (if enabled and not using SCoT) can be injected into prompt; here we omit for brevity
        a1 = choose_action(agent1, prompt1, i)
        a2 = choose_action(agent2, prompt2, i)
        pair = f"{a1}{a2}"
        pairs.append(pair)

        r1, r2 = payoff(cfg.game_type, a1, a2)
        totals[0] += r1; totals[1] += r2

        # Insert round row
        round_row = {
            "game_id": game_id,
            "round": i,
            "p_now": p_now,
            "answer1": a1,
            "answer2": a2,
            "points1": r1,
            "points2": r2,
            "total1": totals[0],
            "total2": totals[1],
            "agent1_pred_prob_C": a1_pred,
            "agent2_pred_prob_C": a2_pred,
            "agent1_plan": a1_plan,
            "agent2_plan": a2_plan,
            "agent1_reflection": a1_refl,
            "agent2_reflection": a2_refl,
        }
        placeholders_r = ",".join(["?"]*len(round_row))
        cur.execute(f"INSERT INTO rounds ({','.join(round_row.keys())}) VALUES ({placeholders_r})", list(round_row.values()))
        conn.commit()

        # stop for continuation with geometric stopping (use previous round p for stop test)
        if cfg.horizon_type == "continuation":
            # continue with p_now; after acting, decide to stop
            if random.random() > (p_now if p_now is not None else 0.0):
                break

    conn.close()
    return game_id

# -------------------- CLI (optional) --------------------
def interactive_menu():
    print("=== Experiment 4 — SCoT Modular Switches ===")
    db_path = input("DB path (default: game_results.db): ").strip() or "game_results.db"
    game_type = input("Game type [PD/BoS] (default: PD): ").strip().upper() or "PD"
    horizon_type = input("Horizon [fixed/continuation] (default: fixed): ").strip() or "fixed"
    reasoning_mode = input("Reasoning [baseline/scot] (default: baseline): ").strip() or "baseline"
    L_win = int(input("History L [1/5/-1(inf)] (default: -1): ").strip() or -1)

    use_scot = (input("use_scot [y/n] (default n): ").strip().lower() == "y")
    scot_P = (input("scot_P prediction [y/n] (default n): ").strip().lower() == "y")
    scot_J = (input("scot_J plan [y/n] (default n): ").strip().lower() == "y")
    scot_R = (input("scot_R reflection [y/n] (default n): ").strip().lower() == "y")
    cot_enabled = (input("cot_enabled plain CoT [y/n] (default n): ").strip().lower() == "y")

    if horizon_type == "fixed":
        T = int(input("T (default 10): ").strip() or 10)
        p_mode = "fixed"; p_fixed = None; p_trace = None
    else:
        p_mode = input("p_mode [fixed/trace] (default fixed): ").strip() or "fixed"
        T = None
        if p_mode == "fixed":
            p_fixed = float(input("p (0-1, default 0.8): ").strip() or 0.8)
            p_trace = None
        else:
            raw = input("p_trace as comma-separated (e.g., 0.2,0.3,0.5,0.6 ...): ").strip()
            p_trace = [float(x) for x in raw.split(",") if x]
            p_fixed = None

    player1 = input(f"Player1 agent name {list(AGENTS.keys())} (default act_J): ").strip() or "act_J"
    player2 = input(f"Player2 agent name {list(AGENTS.keys())} (default act_F): ").strip() or "act_F"

    seed = int(input("seed (default 0): ").strip() or 0)
    match_id = int(input("match_id (default 0): ").strip() or 0)

    cfg = GameConfig(
        game_type=game_type, horizon_type=horizon_type, T=T,
        p_mode=p_mode, p_fixed=p_fixed, p_trace=p_trace,
        max_round=100, reasoning_mode=reasoning_mode, L_window=L_win
    )
    scot = SCoTSwitches(use_scot=use_scot, scot_P=scot_P, scot_J=scot_J, scot_R=scot_R, cot_enabled=cot_enabled)
    meta = MatchMeta(player1=player1, player2=player2, seed=seed, match_id=match_id)

    gid = run_match(db_path, cfg, scot, meta)
    print(f"Done. game_id={gid} written to {db_path}")

if __name__ == "__main__":
    # For direct run without CLI, you can set a quick demo here:
    if os.environ.get("SCOT4_DEMO", "0") == "1":
        cfg = GameConfig(game_type="PD", horizon_type="continuation", T=None,
                         p_mode="fixed", p_fixed=0.8, p_trace=None,
                         max_round=50, reasoning_mode="scot", L_window=5)
        scot = SCoTSwitches(use_scot=True, scot_P=True, scot_J=True, scot_R=True, cot_enabled=False)
        meta = MatchMeta(player1="act_J", player2="act_F", seed=0, match_id=0)
        run_match("game_results.db", cfg, scot, meta)
    else:
        interactive_menu()
