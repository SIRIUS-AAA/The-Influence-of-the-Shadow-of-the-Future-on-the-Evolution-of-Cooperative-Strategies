
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 3 Runner — Dynamic continuation probability p_t
----------------------------------------------------------
Design:
- Paths:
  * up:    linear 0.2 -> 0.8
  * down:  linear 0.8 -> 0.2
  * random-walk: p_{t+1} = p_t + eta, clipped to [0,1]
- Games: PD, BoS
- Reasoning modes: baseline, scot (k-vote)
- Pairings: A vs A, A vs B
- History window L ∈ {1,5,inf}: provide a compact summary of last L rounds in prompt
- Termination: geometric each round with current p_t; hard cap max_round
- Output: experiment_exp3.csv
"""

import os
import time
import random
from typing import List, Tuple, Callable, Any

import numpy as np
import pandas as pd

from openai import OpenAI



CONFIG = {
    "games": ["PD", "BoS"],
    "paths": ["up", "down", "rw"],
    "rw_eta": 0.05,          # random-walk step std (zero-mean normal), clipped
    "start_p": {"up":0.2, "down":0.8, "rw":0.5},
    "end_p":   {"up":0.8, "down":0.2, "rw":None},
    "max_round": 100,
    "reasoning_modes": ["baseline", "scot"],
    "L_windows": [1, 5, -1],  # -1 for infinity
    "seeds": [0,1,2,3,4],
    "include_cross_model": True,
    "matches_per_seed": 1,
    "scot_k": 5,
    "out_csv": "experiment_exp3.csv",
}

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




def act_J(_: str, __: int) -> str: return "J"
def act_F(_: str, __: int) -> str: return "F"

# ---------------- Mechanics ----------------
def parse_choice(text: str) -> str:
    if not isinstance(text, str): return "J"
    t = text.strip().upper()
    for tk in ["OPTION J", " CHOOSE J", " ANSWER J", " A: J", " A: OPTION J"]:
        if tk in t: return "J"
    for tk in ["OPTION F", " CHOOSE F", " ANSWER F", " A: F", " A: OPTION F"]:
        if tk in t: return "F"
    for ch in reversed(t):
        if ch in ("J", "F"): return ch
    return "J"

def payoff(game_type: str, a1: str, a2: str) -> Tuple[int, int]:
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

def path_sequence(path: str, max_round: int, eta: float, start_p: float, end_p: float=None) -> List[float]:
    seq = []
    if path in ("up", "down"):
        for t in range(max_round):
            frac = t / max(1, max_round-1)
            p = start_p + (end_p - start_p) * frac
            seq.append(max(0.0, min(1.0, p)))
    else:
        p = start_p
        for _ in range(max_round):
            seq.append(max(0.0, min(1.0, p)))
            p = p + np.random.normal(0.0, eta)
            p = max(0.0, min(1.0, p))
    return seq

def build_history_summary(pairs: List[str], L: int) -> str:
    if L == 0 or len(pairs) == 0:
        return ""
    if L < 0:  # infinity
        window = pairs
    else:
        window = pairs[-min(L, len(pairs)):]

    cJJ = sum(1 for x in window if x=="JJ")
    cFF = sum(1 for x in window if x=="FF")
    cJF = sum(1 for x in window if x=="JF")
    cFJ = sum(1 for x in window if x=="FJ")
    last = window[-1] if window else "NA"
    return (f"Recent {len(window)} rounds summary: JJ={cJJ}, FF={cFF}, JF={cJF}, FJ={cFJ}. "
            f"Last pair={last}. ")

def build_prompt(game: str, path: str):
    if game == "PD":
        rules = (
            "If you choose Option J and the other player chooses Option J, then you both win 8 points.\n"
            "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
            "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
            "If both choose Option F, then you both win 5 points.\n\n"
        )
        base = "You are playing a repeated game (Prisoner's Dilemma variant). Options: J or F.\n" + rules
        return base, base
    else:
        rules1 = (
            "If both choose Option J, you win 10 points and the other player wins 7 points.\n"
            "If both choose Option F, you win 7 points and the other player wins 10 points.\n"
            "If the options differ, you both win 0 points.\n\n"
        )
        rules2 = (
            "If both choose Option J, you win 7 points and the other player wins 10 points.\n"
            "If both choose Option F, you win 10 points and the other player wins 7 points.\n"
            "If the options differ, you both win 0 points.\n\n"
        )
        base1 = "You are playing a repeated coordination game (Battle of the Sexes). Options: J or F.\n" + rules1
        base2 = "You are playing a repeated coordination game (Battle of the Sexes). Options: J or F.\n" + rules2
        return base1, base2

def get_action(player_func: Callable, prompt_base: str, summary: str, p_now: float, round_i: int, mode: str, scot_k: int) -> str:
    if mode == "baseline":
        prompt = (
            prompt_base +
            f"The continuation probability this round is p={p_now:.2f} (higher p means more likely to continue).\n" +
            summary +
            f"Round {round_i}. Which Option do you choose, Option J or Option F?\n" +
            "Answer with 'Option J' or 'Option F'.\nA: Option"
        )
        raw = player_func(prompt, round_i-1)
        return parse_choice(raw)
    # scot
    votes = []
    for _ in range(scot_k):
        prompt = (
            prompt_base +
            f"The continuation probability this round is p={p_now:.2f} (higher p means more likely to continue).\n" +
            summary +
            f"Round {round_i}. Briefly consider the other's incentives and recent history, then choose.\n" +
            "Final answer: 'Option J' or 'Option F'.\nA:"
        )
        raw = player_func(prompt, round_i-1)
        votes.append(parse_choice(raw))
    return "J" if votes.count("J") >= votes.count("F") else "F"

def run_match(game: str, path: str, L: int, player_1, player_2, mode: str, seed: int, match_id: int, scot_k: int, max_round: int, eta: float, start_p: float, end_p: float):
    random.seed(seed); np.random.seed(seed)
    p1_prompt, p2_prompt = build_prompt(game, path)
    totals = [0,0]
    rows = []
    pairs = []
    p_seq = path_sequence(path, max_round, eta, start_p, end_p)

    for i, p_now in enumerate(p_seq, start=1):
        # stop with probability 1-p_now
        if i > 1 and (random.random() > p_seq[i-2]):  # continue with previous p
            break

        summary = build_history_summary(pairs, L)
        a1 = get_action(player_1, p1_prompt, summary, p_now, i, mode, scot_k)
        a2 = get_action(player_2, p2_prompt, summary, p_now, i, mode, scot_k)
        pair = f"{a1}{a2}"
        pairs.append(pair)

        r1, r2 = payoff(game, a1, a2)
        totals[0] += r1; totals[1] += r2

        rows.append([
            game, path, L if L>=0 else np.inf, mode, seed, match_id,
            player_1.__name__, player_2.__name__,
            "self" if player_1.__name__ == player_2.__name__ else "cross",
            i, p_now, a1, a2, r1, r2, totals[0], totals[1]
        ])

    return rows

def main():
    agents = [
        # act_doubao,
        act_kimi,
        # act_deepseek,
        # act_gpt4,
        # act_J,
        # act_F,
        # act_defect_once,
        # Optional: anthropic-style two-arg function
        # wrap Claude2 into a two-arg adapter for consistency
    ]
    pairings = [(a,a) for a in agents]
    if CONFIG["include_cross_model"]:
        pairings += [(a,b) for a in agents for b in agents if a != b]

    all_rows = []
    for game in CONFIG["games"]:
        for path in CONFIG["paths"]:
            for L in CONFIG["L_windows"]:
                for mode in CONFIG["reasoning_modes"]:
                    for seed in CONFIG["seeds"]:
                        for match_id in range(CONFIG["matches_per_seed"]):
                            for p1, p2 in pairings:
                                rows = run_match(
                                    game=game, path=path, L=L,
                                    player_1=p1, player_2=p2,
                                    mode=mode, seed=seed, match_id=match_id,
                                    scot_k=CONFIG["scot_k"], max_round=CONFIG["max_round"],
                                    eta=CONFIG["rw_eta"],
                                    start_p=CONFIG["start_p"][path],
                                    end_p=CONFIG["end_p"][path]
                                )
                                print(rows)
                                all_rows += rows

    cols = ["game_type","path","L_window","reasoning_mode","seed","match_id","player1","player2","pairing_type","round","p_now","answer1","answer2","points1","points2","total1","total2"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(CONFIG["out_csv"], index=False)
    print(f"Saved {len(df)} rows to {CONFIG['out_csv']}")

if __name__ == "__main__":
    main()
