#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp1_interactive_llm.py

交互式：
- 选博弈：PD / BoS
- 选地平线：fixed（有限轮，输入 T，可选把 p 仅写入 Prompt）/ continuation（未知轮，按 p 决定是否继续，可选把 p 写入 Prompt）
- 是否只跑 baseline（默认：是；也可切换到 cot / scot）
- 选模型配对：自博弈 A vs A 或 跨模型 A vs B，并从注册表选择具体两个“大模型”
- 设定重复次数 / 随机种子
- 跑完将逐轮数据 **追加** 写入 CSV（含 p_in_prompt, p_prompt）

⚠️ API 说明：
- 下面提供了主流大模型的最小可用封装（OpenAI, DeepSeek, Moonshot/Kimi, Doubao, Anthropic Claude）。
- 需自行在环境变量里配置 API Key：
  OPENAI_API_KEY / DEEPSEEK_API_KEY / MOONSHOT_API_KEY / DOUBAO_API_KEY / ANTHROPIC_API_KEY
- 不同版本/地域的接口路径或参数名可能略有差异；如有 4xx/404，请对照你账户的官方文档把 endpoint/model 名称对齐即可。
"""

import os
import time
import json
import random
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# =============================
# 载入 .env（若存在）
# =============================
load_dotenv()

# =============================
# —— 工具：解析、收益、轮数、Prompt 生成 ——
# =============================

def parse_choice(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "J"
    t = text.strip().upper()
    if "OPTION J" in t or t.endswith(" J"):
        return "J"
    if "OPTION F" in t or t.endswith(" F"):
        return "F"
    for ch in reversed(t):
        if ch in ("J", "F"):
            return ch
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


def expected_T_from_p(p: float) -> float:
    return float("inf") if (p is None or p <= 0 or p >= 1) else 1.0 / (1.0 - p)


def generate_round_indices(horizon_type: str, T: int = None, p: float = None, max_round: int = 200) -> List[int]:
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
            if random.random() > p or i >= max_round:
                break
        return rounds
    else:
        raise ValueError("Unknown horizon_type")


def build_prompt_PDbos(game_type: str, horizon_type: str, T: int = None, p: float = None,
                       *, p_in_prompt: bool = False, p_for_prompt: float = None) -> Tuple[str, str]:
    if game_type == "PD":
        rules = (
            "The rules of the game are as follows:\n"
            "If you choose Option J and the other player chooses Option J, then you both win 8 points.\n"
            "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
            "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
            "If both choose Option F, then you both win 5 points.\n\n"
        )
        if horizon_type == "fixed":
            horizon_text = f"You will play exactly {T} rounds with the same player.\n"
            if p_in_prompt and p_for_prompt is not None:
                horizon_text += (
                    f"For your reasoning only, also consider a hypothetical continuation probability p={p_for_prompt}. "
                    f"This indicates the likelihood of future interactions, but in THIS session the number of rounds is fixed.\n"
                )
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
        if horizon_type == "fixed":
            horizon_text = f"You will play exactly {T} rounds with the same player.\n"
            if p_in_prompt and p_for_prompt is not None:
                horizon_text += (
                    f"For your reasoning only, also consider a hypothetical continuation probability p={p_for_prompt}. "
                    f"This indicates the likelihood of future interactions, but in THIS session the number of rounds is fixed.\n"
                )
        else:
            horizon_text = f"After each round, with probability {p} the game continues, otherwise it ends immediately.\n"
        base1 = "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n" + horizon_text + rules_p1
        base2 = "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n" + horizon_text + rules_p2
        return base1, base2
    else:
        raise ValueError("Unknown game_type")


def get_action(player_func: Callable, prompt_base: str, round_i: int, reasoning_mode: str) -> str:
    if reasoning_mode == "baseline":
        prompt = (
            prompt_base +
            f"Round {round_i}. Which Option do you choose, Option J or Option F?\n"
            "Answer format: 'Option J' or 'Option F' only.\nA: Option"
        )
    elif reasoning_mode == "cot":
        prompt = (
            prompt_base +
            f"Round {round_i}. Briefly explain your reasoning, then state a final choice.\n"
            "Final answer format: 'Option J' or 'Option F' only.\nA:"
        )
    elif reasoning_mode == "scot":
        prompt = (
            prompt_base +
            f"Round {round_i}. Consider the other player's incentives, then choose.\n"
            "Final answer format: 'Option J' or 'Option F' only.\nA:"
        )
    else:
        raise ValueError("Unknown reasoning_mode")

    raw = player_func(prompt, round_i-1) if player_func.__code__.co_argcount >= 2 else player_func(prompt)
    return parse_choice(raw)

# =============================
# —— 大模型封装（最小可用版） ——
# =============================

def _retry_request(do_call: Callable[[], str], *, retries: int = 3, backoff: float = 1.5) -> str:
    last_err = None
    for i in range(retries):
        try:
            return do_call()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    # 失败时保底返回 J，避免整局中断；同时把错误打印出来便于定位
    print("[WARN] LLM call failed, fallback to 'Option J'. Error:", repr(last_err))
    return "Option J"

# --- OpenAI Chat Completions (兼容 gpt-4o / gpt-4o-mini 等) ---

def act_openai_gpt(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _call() -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful decision maker. Output exactly 'Option J' or 'Option F' in the final line."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.environ.get("OPENAI_TEMPERATURE", 0)),
            "max_tokens": 16,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        txt = resp.json()["choices"][0]["message"]["content"]
        return txt

    return _retry_request(_call)

# --- DeepSeek ---

def act_deepseek(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1/chat/completions")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    def _call() -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Output exactly 'Option J' or 'Option F' in the final line."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.environ.get("DEEPSEEK_TEMPERATURE", 0)),
            "max_tokens": 16,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _retry_request(_call)

# --- Moonshot (Kimi) ---

def act_kimi(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY not set")
    url = os.environ.get("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1/chat/completions")
    model = os.environ.get("MOONSHOT_MODEL", "moonshot-v1-8k")

    def _call() -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "输出最后一行必须是 'Option J' 或 'Option F'"},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.environ.get("MOONSHOT_TEMPERATURE", 0)),
            "max_tokens": 16,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _retry_request(_call)

# --- Doubao ---

def act_doubao(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("DOUBAO_API_KEY")
    if not api_key:
        raise RuntimeError("DOUBAO_API_KEY not set")
    url = os.environ.get("DOUBAO_API_BASE", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    model = os.environ.get("DOUBAO_MODEL", "doubao-pro-32k")

    def _call() -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "最终一行只输出 'Option J' 或 'Option F'"},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.environ.get("DOUBAO_TEMPERATURE", 0)),
            "max_tokens": 16,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return _retry_request(_call)

# --- Anthropic Claude ---

def act_claude(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    url = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1/messages")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    def _call() -> str:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 32,
            "temperature": float(os.environ.get("ANTHROPIC_TEMPERATURE", 0)),
            "system": "Output exactly 'Option J' or 'Option F' in the final line.",
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Claude messages 返回 content 列表
        content = "".join([blk.get("text", "") for blk in data.get("content", [])])
        return content

    return _retry_request(_call)

# =============================
# —— 主流程：一局 & CSV ——
# =============================

def get_action_with_mode(player_func: Callable, prompt_base: str, round_i: int, reasoning_mode: str) -> str:
    return get_action(player_func, prompt_base, round_i, reasoning_mode)


def run_match(game_type: str,
              horizon_type: str,
              T: int,
              p: float,
              player_1: Callable,
              player_2: Callable,
              reasoning_mode: str,
              seed: int,
              max_round: int,
              *,
              p_in_prompt: bool = False,
              p_for_prompt: float = None) -> List[List[Any]]:
    random.seed(seed)
    np.random.seed(seed)

    prompt1, prompt2 = build_prompt_PDbos(game_type, horizon_type, T=T, p=p, p_in_prompt=p_in_prompt, p_for_prompt=p_for_prompt)

    history1, history2 = "", ""
    totals = [0, 0]
    rows = []

    rounds = generate_round_indices(horizon_type, T=T, p=p, max_round=max_round)
    for i in rounds:
        q1 = prompt1 + history1 + f"\nYou are currently playing round {i}.\n"
        q2 = prompt2 + history2 + f"\nYou are currently playing round {i}.\n"

        a1 = get_action_with_mode(player_1, q1, i, reasoning_mode)
        a2 = get_action_with_mode(player_2, q2, i, reasoning_mode)

        r1, r2 = payoff(game_type, a1, a2)
        totals[0] += r1
        totals[1] += r2

        history1 += f"In round {i}, you chose Option {a1} and the other player chose Option {a2}. You won {r1} points and they won {r2} points.\n"
        history2 += f"In round {i}, you chose Option {a2} and the other player chose Option {a1}. You won {r2} points and they won {r1} points.\n"

        rows.append([
            game_type, horizon_type, T, p, expected_T_from_p(p) if p else None,
            p_in_prompt, p_for_prompt,
            reasoning_mode, seed,
            getattr(player_1, "__name__", "player1"),
            getattr(player_2, "__name__", "player2"),
            i, a1, a2, r1, r2, totals[0], totals[1]
        ])

    return rows


def append_rows_to_csv(rows: List[List[Any]], out_csv: str):
    columns = [
        "game_type", "horizon_type", "T", "p", "E_T",
        "p_in_prompt", "p_prompt",
        "reasoning_mode", "seed",
        "player1", "player2",
        "round", "answer1", "answer2",
        "points1", "points2", "total1", "total2",
    ]
    df = pd.DataFrame(rows, columns=columns)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=header, index=False)
    print(f"[CSV] Appended {len(df)} rows to {out_csv}")

# =============================
# —— 交互菜单 ——
# =============================

def choose_from_list(title: str, options: List[str], default_index: int = 0) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options):
        mark = "*" if i == default_index else " "
        print(f"  [{i}] {opt} {mark}")
    while True:
        s = input(f"选择序号(默认 {default_index}): ").strip()
        if s == "":
            return options[default_index]
        if s.isdigit() and 0 <= int(s) < len(options):
            return options[int(s)]
        print("无效输入，请重试。")


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    while True:
        s = input(f"{prompt} ({default}): ").strip().lower()
        if s == "":
            return default_yes
        if s in ("y", "yes"): return True
        if s in ("n", "no"): return False
        print("无效输入，请输入 y 或 n。")


def input_int(prompt: str, default: int = None, min_v: int = None, max_v: int = None) -> int:
    while True:
        s = input(f"{prompt}{'' if default is None else f' [默认 {default}]'}: ").strip()
        if s == "" and default is not None:
            return default
        try:
            v = int(s)
            if (min_v is not None and v < min_v) or (max_v is not None and v > max_v):
                raise ValueError
            return v
        except Exception:
            print("无效整数，请重试。")


def input_float(prompt: str, default: float = None, min_v: float = None, max_v: float = None) -> float:
    while True:
        s = input(f"{prompt}{'' if default is None else f' [默认 {default}]'}: ").strip()
        if s == "" and default is not None:
            return default
        try:
            v = float(s)
            if (min_v is not None and v < min_v) or (max_v is not None and v > max_v):
                raise ValueError
            return v
        except Exception:
            print("无效浮点数，请重试。")


def main():
    print("=== LLM × Game Theory — 交互式大模型对战 ===")

    # 环境变量可用性检测
    def env_ok(name: str) -> bool:
        v = os.environ.get(name)
        return bool(v and v.strip())

    # 1) 可选大模型注册表（只收录已配置 API Key 的模型）
    agent_registry: Dict[str, Callable] = {}
    if env_ok("OPENAI_API_KEY"):
        agent_registry["openai:gpt"] = act_openai_gpt
    if env_ok("DEEPSEEK_API_KEY"):
        agent_registry["deepseek"] = act_deepseek
    if env_ok("MOONSHOT_API_KEY"):
        agent_registry["moonshot:kimi"] = act_kimi
    if env_ok("DOUBAO_API_KEY"):
        agent_registry["doubao"] = act_doubao
    if env_ok("ANTHROPIC_API_KEY"):
        agent_registry["anthropic:claude"] = act_claude

    if not agent_registry:
        print("[ERROR] 没有检测到可用的大模型 API Key。请在 .env 或系统环境中设置至少一个：")
        print("        OPENAI_API_KEY / DEEPSEEK_API_KEY / MOONSHOT_API_KEY / DOUBAO_API_KEY / ANTHROPIC_API_KEY")
        return

    # 2) 选择博弈
    game_type = choose_from_list("选择博弈类型", ["PD", "BoS"], default_index=0)

    # 3) 选择地平线
    horizon_type = choose_from_list("选择地平线（有限轮或未知轮）", ["fixed", "continuation"], default_index=0)

    T = None
    p = None
    p_in_prompt_flag = False
    p_for_prompt = None
    if horizon_type == "fixed":
        T = input_int("输入固定轮数 T", default=10, min_v=1, max_v=10000)
        add_p_prompt = ask_yes_no("是否在提示中加入 p 信息（仅提示，不影响轮数）？", default_yes=False)
        if add_p_prompt:
            if ask_yes_no("是否按 T 推导 p（p = 1 - 1/T）？", default_yes=True):
                p_for_prompt = 1.0 - 1.0 / T
            else:
                p_for_prompt = input_float("输入用于提示的 p_prompt (0<p<1)", default=0.9, min_v=1e-6, max_v=0.999999)
            p_in_prompt_flag = True
        max_round = 200
    else:
        if ask_yes_no("是否按 T 推导 p（p = 1 - 1/T）？", default_yes=True):
            T_for_p = input_int("输入用于推导的 T", default=10, min_v=2)
            p = 1.0 - 1.0 / T_for_p
            print(f"已设置 p = 1 - 1/T = {p:.4f}")
        else:
            p = input_float("输入继续概率 p (0<p<1)", default=0.9, min_v=1e-6, max_v=0.999999)
        p_in_prompt_flag = ask_yes_no("是否把 p 写入提示中？", default_yes=True)
        p_for_prompt = p if p_in_prompt_flag else None
        max_round = input_int("未知轮安全上限（最大轮数）", default=200, min_v=1)

    # 4) 推理范式
    only_baseline = ask_yes_no("是否只运行 baseline（无 CoT/SCoT）？", default_yes=True)
    if only_baseline:
        reasoning_modes = ["baseline"]
    else:
        candidates = ["baseline", "cot", "scot"]
        print("可选推理模式：", candidates)
        pick = input("请输入以逗号分隔的模式（回车默认 baseline）: ").strip()
        if pick == "":
            reasoning_modes = ["baseline"]
        else:
            reasoning_modes = [m.strip() for m in pick.split(",") if m.strip() in candidates] or ["baseline"]
    print("已选择模式:", reasoning_modes)

    # 5) 选择模型配对（自博弈 or 跨模型）
    pairing_type = choose_from_list("选择模型配对类型", ["A_vs_A (自博弈)", "A_vs_B (跨模型)"], default_index=1)

    names = list(agent_registry.keys())
    if pairing_type.startswith("A_vs_A"):
        nameA = choose_from_list("选择模型 A (自博弈将使用同一模型)", names, default_index=0)
        nameB = nameA
    else:
        nameA = choose_from_list("选择模型 A", names, default_index=0)
        default_B = 1 if len(names) > 1 else 0
        nameB = choose_from_list("选择模型 B", names, default_index=default_B)
    player_1 = agent_registry[nameA]
    player_2 = agent_registry[nameB]

    # 6) 重复次数 / 随机种子
    reps = input_int("每个 seed 下重复对局次数", default=1, min_v=1)
    seed_text = input("输入随机种子列表（逗号分隔，默认 0,1,2,3,4）: ").strip()
    seeds = [0, 1, 2, 3, 4] if seed_text == "" else [int(x.strip()) for x in seed_text.split(",") if x.strip()]
    if not seeds:
        seeds = [0]

    # 7) 输出 CSV
    out_csv = input("输出 CSV 文件名（默认 experiments_llm.csv）: ").strip() or "experiments_llm.csv"

    # 8) 确认
    print("\n===== 运行确认 =====")
    print("Game:", game_type)
    print("Horizon:", horizon_type, "T=", T, "p=", p, "E[T]=", None if p is None else f"{expected_T_from_p(p):.2f}")
    print("p_in_prompt:", p_in_prompt_flag, "p_prompt:", p_for_prompt)
    print("Reasoning modes:", reasoning_modes)
    print("Pairing:", f"{nameA} vs {nameB}")
    print("Seeds:", seeds, "reps:", reps)
    print("CSV:", out_csv)
    if not ask_yes_no("确认开始运行？", default_yes=True):
        print("已取消。")
        return

    # 9) 开跑
    total_rows = 0
    for mode in reasoning_modes:
        for sd in seeds:
            for _ in range(reps):
                rows = run_match(
                    game_type=game_type,
                    horizon_type=horizon_type,
                    T=T if horizon_type == "fixed" else None,
                    p=p if horizon_type == "continuation" else None,
                    player_1=player_1,
                    player_2=player_2,
                    reasoning_mode=mode,
                    seed=sd,
                    max_round=max_round,
                    p_in_prompt=p_in_prompt_flag,
                    p_for_prompt=p_for_prompt,
                )
                append_rows_to_csv(rows, out_csv)
                total_rows += len(rows)

    print("\n完成。共写入行数：", total_rows)


if __name__ == "__main__":
    main()
