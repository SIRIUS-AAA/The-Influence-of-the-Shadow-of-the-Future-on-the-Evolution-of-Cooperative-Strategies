
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 2 Runner — Finite vs Unknown Horizon (Backward Induction vs Shadow of the Future)
-------------------------------------------------------------------------------------------
Design:
- Games: PD, BoS
- Horizons:
    * fixed T ∈ {10, 30, 50, 100}
    * continuation p ∈ {0.2, 0.5, 0.8} with geometric stopping (E[T] ≈ 1/(1-p))
- Reasoning modes: baseline, scot
- Pairing: A vs A, A vs B (at least two different LLMs; local fallbacks included)
- Seeds: configurable
- Output: experiment_exp2.csv with per-round records

Notes:
- Online models are optional. If API keys are missing, agents fallback to simple behaviors.
"""

import os
import time
import random
import itertools
from typing import List, Tuple, Callable, Any

import numpy as np
import pandas as pd
from openai import OpenAI
# Optional LLM backends




OPENAI_API_KEY = "sk-vWPlVDMJizXLti9l69ah2W5l0bdGKxwh1OnBBRuhwtPJL4ed"
DEEPSEEK_API_KEY = "sk-cb387c428d9343328cea734e6ae0f9f5"
ANTHROPIC_API_KEY = ''
MOONSHOT_API_KEY = 'sk-8dW8q949DUpVaWjKtATKUovMvabr0VlU0cE3WTFYsdu0Ugwz'
DOUBAO_API_KEY = '068a6b39-c13c-4df2-8253-7d5bfe1cca51'


# ------------------ Config ------------------
CONFIG = {
    "games": ["PD", "BoS"],
    "horizon_types": ["fixed", "continuation"],
    "T_list": [2, 2, 2, 2],
    "p_list": [0.2, 0.5, 0.8],
    "max_round": 100,
    "reasoning_modes": ["baseline", "scot"],
    "seeds": [0, 1, 2, 3, 4],
    "scot_k": 5,
    "include_cross_model": True,
    "matches_per_seed": 1,
    "out_csv": "experiment_exp2.csv",
}




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
def act_defect_once(_: str, i: int) -> str:
    answers = ["F", "J", "J", "J", "J", "J", "J", "J", "J", "J"]
    idx = max(0, min(i, len(answers)-1))
    return answers[idx]

# ------------------ Mechanics ------------------
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

def build_prompt(game_type: str, horizon_type: str, T=None, p=None):
    if game_type == "PD":
        rules = (
            "If you choose Option J and the other player chooses Option J, then you both win 8 points.\n"
            "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
            "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
            "If both choose Option F, then you both win 5 points.\n\n"
        )
        horizon = f"You will play exactly {T} rounds.\n" if horizon_type=="fixed" else f"After each round, with probability {p} the game continues; otherwise it ends.\n"
        base = "You are playing a repeated game with another player. Options: J or F.\n" + horizon + rules
        return base, base
    elif game_type == "BoS":
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
        horizon = f"You will play exactly {T} rounds.\n" if horizon_type=="fixed" else f"After each round, with probability {p} the game continues; otherwise it ends.\n"
        base1 = "You are playing a repeated game with another player. Options: J or F.\n" + horizon + rules1
        base2 = "You are playing a repeated game with another player. Options: J or F.\n" + horizon + rules2
        return base1, base2
    raise ValueError("Unknown game_type")

def generate_rounds(horizon_type: str, T=None, p=None, max_round=100) -> List[int]:
    if horizon_type == "fixed":
        return list(range(1, T+1))
    rounds, i = [], 0
    while True:
        i += 1
        rounds.append(i)
        if random.random() > p or i >= max_round:
            break
    return rounds

def get_action(player_func: Callable, prompt_base: str, round_i: int, mode: str, scot_k: int) -> str:
    if mode == "baseline":
        prompt = prompt_base + f"Round {round_i}. Which Option do you choose, Option J or Option F?\nAnswer with 'Option J' or 'Option F'.\nA: Option"
        raw = player_func(prompt, round_i-1)
        return parse_choice(raw)
    # scot
    votes = []
    for _ in range(scot_k):
        prompt = prompt_base + f"Round {round_i}. Briefly consider the other's incentives, then pick.\nFinal answer: 'Option J' or 'Option F'.\nA:"
        raw = player_func(prompt, round_i-1)
        votes.append(parse_choice(raw))
    return "J" if votes.count("J") >= votes.count("F") else "F"

def run_one_match(game_type: str, horizon_type: str, T, p, player_1, player_2, mode: str, seed: int, match_id: int, scot_k: int, max_round: int):
    random.seed(seed); np.random.seed(seed)
    p1_prompt, p2_prompt = build_prompt(game_type, horizon_type, T=T, p=p)
    history1 = ""; history2 = ""
    totals = [0,0]
    rows = []
    rounds = generate_rounds(horizon_type, T=T, p=p, max_round=max_round)
    for i in rounds:
        a1 = get_action(player_1, p1_prompt + history1, i, mode, scot_k)
        a2 = get_action(player_2, p2_prompt + history2, i, mode, scot_k)
        r1, r2 = payoff(game_type, a1, a2)
        totals[0] += r1; totals[1] += r2
        history1 += f"In round {i}, you chose Option {a1} and the other chose Option {a2}. You won {r1}, they won {r2}.\n"
        history2 += f"In round {i}, you chose Option {a2} and the other chose Option {a1}. You won {r2}, they won {r1}.\n"
        rows.append([
            game_type, horizon_type, T, p, (1.0/(1-p) if (p and p<1) else None),
            mode, seed, match_id,
            player_1.__name__, player_2.__name__,
            "self" if player_1.__name__ == player_2.__name__ else "cross",
            i, a1, a2, r1, r2, totals[0], totals[1]
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
        pairings += list(itertools.permutations(agents, 2))

    all_rows = []
    for game in CONFIG["games"]:
        for horizon in CONFIG["horizon_types"]:
            for mode in CONFIG["reasoning_modes"]:
                for seed in CONFIG["seeds"]:
                    if horizon == "fixed":
                        for T in CONFIG["T_list"]:
                            for match_id in range(CONFIG["matches_per_seed"]):
                                for p1, p2 in pairings:
                                    row = run_one_match(game, horizon, T, None, p1, p2, mode, seed, match_id, CONFIG["scot_k"], CONFIG["max_round"])
                                    print(row)
                                    all_rows += row
                    else:
                        for p in CONFIG["p_list"]:
                            for match_id in range(CONFIG["matches_per_seed"]):
                                for p1, p2 in pairings:
                                    row = run_one_match(game, horizon, None, p, p1, p2, mode, seed, match_id, CONFIG["scot_k"], CONFIG["max_round"])
                                    print(row)
                                    all_rows += row

    cols = ["game_type","horizon_type","T","p","E_T","reasoning_mode","seed","match_id","player1","player2","pairing_type","round","answer1","answer2","points1","points2","total1","total2"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(CONFIG["out_csv"], index=False)
    print(f"Saved {len(df)} rows to {CONFIG['out_csv']}")

if __name__ == "__main__":
    main()
