import os
import csv
import time
import math
import json
import random
import itertools
from typing import List, Tuple, Dict, Callable, Any
from openai import OpenAI
import numpy as np
import pandas as pd



try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except Exception:
    Anthropic = None
    HUMAN_PROMPT = ""
    AI_PROMPT = ""

try:
    import openai
except Exception:
    openai = None

# =============================
# Experiment configuration
# =============================
CONFIG = {
    "games": ["PD", "BoS"],
    "horizon_types": ["fixed", "continuation"],  # fixed T vs geometric continuation p
    "T_list": [4, 4, 4],
    "p_list": [0.2, 0.5, 0.8],
    "max_round": 100,  # safety cap
    "reasoning_modes": ["baseline", "cot", "scot"],
    # seeds for repeated matches; extend this for larger experiments
    "seeds": [0, 1, 2, 3, 4],
    # SCoT vote count
    "scot_k": 5,
    # Output file
    "out_csv": "experiment_exp1.csv",
    # Whether to include cross-model pairings A vs B
    "include_cross_model": True,
    # Number of matches per (condition, pairing, seed). If >1, vary an internal match_id.
    "matches_per_seed": 1,
}

# =============================
# LLM/Agent registry
# =============================
# NOTE: Replace API_KEYs to enable online models. Otherwise, local baselines run.

def safe_get_env(k, default=None):
    try:
        return os.environ.get(k, default)
    except Exception:
        return default
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


def api_request_anthropic(model, instruct, ask, max_tokens=8, temperature=0.0):
    client = Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=5, timeout=20.0)
    resp = client.completions.create(
        model=model,
        temperature=temperature,
        max_tokens_to_sample=max_tokens,
        prompt=f"{HUMAN_PROMPT} {instruct} {AI_PROMPT} {ask}"
    )
    return resp


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

def act_claude2(instruct: str, ask: str) -> str:
    try:
        resp = api_request_anthropic("claude-2", instruct, ask, max_tokens=8, temperature=0.0)
        return getattr(resp, "completion", "").strip()
    except Exception:
        return "J"

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



# Heuristic / baseline agents
def act_J(_: str, __: int) -> str: return "J"
def act_F(_: str, __: int) -> str: return "F"
def act_defect_once(_: str, i: int) -> str:
    answers = ["F", "J", "J", "J", "J", "J", "J", "J", "J", "J"]
    idx = max(0, min(i, len(answers)-1))
    return answers[idx]

# =============================
# Utilities
# =============================
def parse_choice(text: str) -> str:
    """
    Parse model output into 'J' or 'F'.
    Looks for 'Option J' / 'Option F' / single 'J'/'F' tokens.
    Defaults to 'J' if unclear (conservative towards cooperation/coordination).
    """
    if not isinstance(text, str):
        return "J"
    t = text.strip().upper()
    for token in ["OPTION J", " CHOOSE J", " ANSWER J", " A: J", " A: OPTION J"]:
        if token in t:
            return "J"
    for token in ["OPTION F", " CHOOSE F", " ANSWER F", " A: F", " A: OPTION F"]:
        if token in t:
            return "F"
    # fallback: last non-empty char J/F
    for ch in reversed(t):
        if ch in ("J", "F"):
            return ch
    return "J"

def payoff(game_type: str, a1: str, a2: str, perspective: int = 0) -> Tuple[int, int]:
    """
    Returns (p1, p2) payoff.
    For BoS, perspective parameter is ignored—matrix already asymmetric.
    """
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

def generate_round_indices(horizon_type: str, T: int = None, p: float = None, max_round: int = 100) -> List[int]:
    if horizon_type == "fixed":
        assert T is not None and T > 0
        return list(range(1, T + 1))
    elif horizon_type == "continuation":
        assert p is not None and 0.0 < p < 1.0
        rounds = []
        i = 0
        while True:
            i += 1
            rounds.append(i)
            # continue with probability p
            if random.random() > p or i >= max_round:
                break
        return rounds
    else:
        raise ValueError("Unknown horizon_type")

def expected_T_from_p(p: float) -> float:
    if p <= 0 or p >= 1:
        return float("inf")
    return 1.0 / (1.0 - p)

def build_prompt_PDbos(game_type: str, horizon_type: str, T: int = None, p: float = None, num_interactions_hint: int = None):
    """
    Returns (prompt_p1, prompt_p2) where for PD the two prompts are identical,
    and for BoS they reflect asymmetric preferences.
    """
    if game_type == "PD":
        rules = (
            "The rules of the game are as follows:\n"
            "If you choose Option J and the other player chooses Option J, then you both win 8 points.\n"
            "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
            "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
            "If both choose Option F, then you both win 5 points.\n\n"
        )
        horizon_text = ""
        if horizon_type == "fixed":
            horizon_text = f"You will play exactly {T} rounds with the same player.\n"
        else:
            horizon_text = f"After each round, with probability {p} the game continues, otherwise it ends immediately.\n"
        base = (
            "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n"
            + horizon_text + rules
        )
        return base, base

    elif game_type == "BoS":
        rules_p1 = (
            "The rules of the game are as follows:\n"
            "If both choose Option J, you win 10 points and the other player wins 7 points.\n"
            "If both choose Option F, you win 7 points and the other player wins 10 points.\n"
            "If the options differ (J/F or F/J), you both win 0 points.\n\n"
        )
        rules_p2 = (
            "The rules of the game are as follows:\n"
            "If both choose Option J, you win 7 points and the other player wins 10 points.\n"
            "If both choose Option F, you win 10 points and the other player wins 7 points.\n"
            "If the options differ (J/F or F/J), you both win 0 points.\n\n"
        )
        horizon_text = ""
        if horizon_type == "fixed":
            horizon_text = f"You will play exactly {T} rounds with the same player.\n"
        else:
            horizon_text = f"After each round, with probability {p} the game continues, otherwise it ends immediately.\n"
        base1 = "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n" + horizon_text + rules_p1
        base2 = "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n" + horizon_text + rules_p2
        return base1, base2
    else:
        raise ValueError("Unknown game_type")

def get_action(player_func: Callable, prompt_base: str, history: str, round_i: int, reasoning_mode: str, scot_k: int = 5) -> str:
    """
    Query one agent according to the reasoning mode.
    Returns parsed 'J' or 'F'.
    """
    if reasoning_mode == "baseline":
        prompt = (
            prompt_base + history +
            f"Round {round_i}. Which Option do you choose, Option J or Option F?\n" +
            "Answer format: 'Option J' or 'Option F' only.\nA: Option"
        )
        raw = player_func(prompt, round_i-1) if player_func.__code__.co_argcount >= 2 else player_func(prompt, "Option")
        return parse_choice(raw)

    elif reasoning_mode == "cot":
        prompt = (
            prompt_base + history +
            f"Round {round_i}. Briefly explain your reasoning, then state a final choice.\n" +
            "Final answer format: 'Option J' or 'Option F' only.\nA:"
        )
        raw = player_func(prompt, round_i-1) if player_func.__code__.co_argcount >= 2 else player_func(prompt, "Option")
        return parse_choice(raw)

    elif reasoning_mode == "scot":
        votes = []
        for _ in range(scot_k):
            prompt = (
                prompt_base + history +
                f"Round {round_i}. Briefly consider the other player's reasoning and incentives, then choose.\n" +
                "Final answer format: 'Option J' or 'Option F' only.\nA:"
            )
            raw = player_func(prompt, round_i-1) if player_func.__code__.co_argcount >= 2 else player_func(prompt, "Option")
            votes.append(parse_choice(raw))
        # majority vote (tie -> J)
        j_count = votes.count("J")
        f_count = votes.count("F")
        return "J" if j_count >= f_count else "F"
    else:
        raise ValueError("Unknown reasoning_mode")

def run_match(game_type: str,
              horizon_type: str,
              T: int,
              p: float,
              player_1: Callable,
              player_2: Callable,
              reasoning_mode: str,
              seed: int,
              scot_k: int,
              max_round: int) -> List[List[Any]]:
    """
    Runs one match between player_1 and player_2 under a specified condition.
    Returns list of per-round rows for CSV.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Build prompts (PD symmetric, BoS asymmetric)
    prompt1, prompt2 = build_prompt_PDbos(game_type, horizon_type, T=T, p=p)

    # Order randomization for option mentions to reduce priming
    choice_options = ["J", "F"]

    # Conversation history (held separately for two players)
    history1, history2 = "", ""

    totals = [0, 0]
    rows = []

    rounds = generate_round_indices(horizon_type, T=T, p=p, max_round=max_round)
    for i in rounds:
        # randomize order of options in the question line
        order = [0, 1]
        random.shuffle(order)
        # optA, optB = choice_options[order[0]], choice_options[order[1]]

        # Player actions
        q1 = prompt1 + history1 + f"\nYou are currently playing round {i}.\n"
        q2 = prompt2 + history2 + f"\nYou are currently playing round {i}.\n"

        a1 = get_action(player_1, q1, "", i, reasoning_mode, scot_k)
        a2 = get_action(player_2, q2, "", i, reasoning_mode, scot_k)

        # Payoffs
        r1, r2 = payoff(game_type, a1, a2)

        totals[0] += r1
        totals[1] += r2

        # Update visible histories (for next round prompts; concise)
        history1 += f"In round {i}, you chose Option {a1} and the other player chose Option {a2}. You won {r1} points and they won {r2} points.\n"
        history2 += f"In round {i}, you chose Option {a2} and the other player chose Option {a1}. You won {r2} points and they won {r1} points.\n"

        rows.append([
            game_type, horizon_type, T, p, expected_T_from_p(p) if p else None,
            reasoning_mode, seed,
            getattr(player_1, "__name__", "player1"),
            getattr(player_2, "__name__", "player2"),
            i, a1, a2, r1, r2, totals[0], totals[1],
            order[0], order[1]
        ])

    return rows

def main():
    # Register agents here
    agents = [
        act_doubao,
        act_kimi,
        act_deepseek,
        act_gpt4,
        act_J,
        act_F,
        # act_defect_once,
        # Optional: anthropic-style two-arg function
        # wrap Claude2 into a two-arg adapter for consistency
    ]

    # Build pairings
    pairings = []
    for a in agents:
        pairings.append((a, a))  # A vs A
    if CONFIG["include_cross_model"]:
        for a, b in itertools.permutations(agents, 2):
            pairings.append((a, b))  # A vs B

    out_rows: List[List[Any]] = []

    for game_type in CONFIG["games"]:
        for horizon_type in CONFIG["horizon_types"]:
            for reasoning_mode in CONFIG["reasoning_modes"]:
                for seed in CONFIG["seeds"]:
                    # Iterate parameter grids depending on horizon type
                    if horizon_type == "fixed":
                        grid = CONFIG["T_list"]
                        for T in grid:
                            for _m in range(CONFIG["matches_per_seed"]):
                                for (p1, p2) in pairings:
                                    rows = run_match(
                                        game_type=game_type,
                                        horizon_type=horizon_type,
                                        T=T, p=None,
                                        player_1=p1, player_2=p2,
                                        reasoning_mode=reasoning_mode,
                                        seed=seed,
                                        scot_k=CONFIG["scot_k"],
                                        max_round=CONFIG["max_round"]
                                    )
                                    print(rows)
                                    out_rows.extend(rows)
                    else:
                        grid = CONFIG["p_list"]
                        for p in grid:
                            for _m in range(CONFIG["matches_per_seed"]):
                                for (p1, p2) in pairings:
                                    rows = run_match(
                                        game_type=game_type,
                                        horizon_type=horizon_type,
                                        T=None, p=p,
                                        player_1=p1, player_2=p2,
                                        reasoning_mode=reasoning_mode,
                                        seed=seed,
                                        scot_k=CONFIG["scot_k"],
                                        max_round=CONFIG["max_round"]
                                    )
                                    print(rows)
                                    out_rows.extend(rows)

    # Save CSV
    columns = [
        "game_type", "horizon_type", "T", "p", "E_T",
        "reasoning_mode", "seed",
        "player1", "player2",
        "round", "answer1", "answer2",
        "points1", "points2", "total1", "total2",
        "optionA_index", "optionB_index"
    ]
    df = pd.DataFrame(out_rows, columns=columns)
    df.to_csv(CONFIG["out_csv"], index=False)
    print(f"Saved {len(df)} rows to {CONFIG['out_csv']}")

if __name__ == "__main__":
    main()