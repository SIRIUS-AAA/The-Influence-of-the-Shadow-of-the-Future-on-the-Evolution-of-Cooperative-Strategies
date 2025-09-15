#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 2 (exp2.py): Fixed vs Continuation
- Focus: 制度机制对激励的影响（BI vs Shadow of the Future）
- 交互式 + 批量跑批，两种方式都支持
- 支持 PD / BoS；支持多模型自博弈或跨模型
- CSV 逐轮写出，带 ended_by(continuation 的停机原因) 等字段，便于统计

用法见文件底部 `if __name__ == "__main__": main()`
"""

import os
import time
import json
import random
from typing import List, Tuple, Dict, Callable, Any, Optional

import numpy as np
import pandas as pd
import requests


def inject_api_keys():
    # 用 setdefault 避免覆盖外部已设置的环境变量
    os.environ.setdefault("OPENAI_API_KEY",    "sk-vWPlVDMJizXLti9l69ah2W5l0bdGKxwh1OnBBRuhwtPJL4ed")
    os.environ.setdefault("DEEPSEEK_API_KEY",  "sk-42baf13ec4b24c80a751d3a22e5ea8cf")
    os.environ.setdefault("ANTHROPIC_API_KEY", "")
    os.environ.setdefault("MOONSHOT_API_KEY",  "sk-8dW8q949DUpVaWjKtATKUovMvabr0VlU0cE3WTFYsdu0Ugwz")
    os.environ.setdefault("DOUBAO_API_KEY",    "068a6b39-c13c-4df2-8253-7d5bfe1cca51")

inject_api_keys()
# =============================
# —— 基础工具：解析、收益、轮数、Prompt ——
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

def expected_T_from_p(p: Optional[float]) -> Optional[float]:
    if p is None: return None
    if p <= 0 or p >= 1: return None
    return 1.0 / (1.0 - p)

def generate_round_indices(horizon_type: str, T: int = None, p: float = None, max_round: int = 200) -> Tuple[List[int], str]:
    """
    返回 (round_indices, ended_by)
    ended_by: "fixed", "prob_stop", "cap"
    """
    if horizon_type == "fixed":
        assert T is not None and T > 0
        return list(range(1, T + 1)), "fixed"
    elif horizon_type == "continuation":
        assert p is not None and 0.0 < p < 1.0
        rounds = []
        i = 0
        ended_by = "prob_stop"
        while True:
            i += 1
            rounds.append(i)
            # 几何停机：每轮以概率 (1-p) 终止
            if random.random() > p:
                ended_by = "prob_stop"
                break
            if i >= max_round:
                ended_by = "cap"
                break
        return rounds, ended_by
    else:
        raise ValueError("Unknown horizon_type")

def build_prompt_PDbos(
    game_type: str,
    horizon_type: str,
    *,
    T: Optional[int] = None,
    p: Optional[float] = None,
    show_T_in_prompt: bool = True,
    show_p_in_prompt: bool = True
) -> Tuple[str, str]:
    if game_type == "PD":
        rules = (
            "The rules of the game are as follows:\n"
            "If you choose Option J and the other player chooses Option J, then you both win 8 points.\n"
            "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
            "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
            "If both choose Option F, then you both win 5 points.\n\n"
        )
        if horizon_type == "fixed":
            horizon_text = ""
            if show_T_in_prompt:
                horizon_text = f"You will play exactly {T} rounds with the same player.\n"
        else:
            # continuation：制度机制明确告知 p（强制）
            if not show_p_in_prompt and p is not None:
                # 出于实验一致性建议总是展示 p；但仍允许研究者手动隐藏
                pass
            horizon_text = f"After each round, with probability {p} the game continues, otherwise it ends immediately.\n" if (p is not None and show_p_in_prompt) else "After each round, the game may continue with some probability.\n"
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
            horizon_text = f"You will play exactly {T} rounds with the same player.\n" if show_T_in_prompt else ""
        else:
            horizon_text = f"After each round, with probability {p} the game continues, otherwise it ends immediately.\n" if (p is not None and show_p_in_prompt) else "After each round, the game may continue with some probability.\n"
        base1 = "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n" + horizon_text + rules_p1
        base2 = "You are playing a game repeatedly with another player. In this game, you can choose between Option J and Option F.\n" + horizon_text + rules_p2
        return base1, base2
    else:
        raise ValueError("Unknown game_type")

def _retry_request(do_call: Callable[[], str], *, retries: int = 3, backoff: float = 1.5) -> str:
    last_err = None
    for i in range(retries):
        try:
            return do_call()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    print("[WARN] LLM call failed, fallback to 'Option J'. Error:", repr(last_err))
    return "Option J"

# =============================
# —— 模型最小封装（示例：OpenAI/DeepSeek/Kimi/Doubao/Claude） ——
#     * 必须从环境变量读取，不再硬编码 *
# =============================

def act_openai_gpt(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = os.environ.get("OPENAI_API_BASE", "https://api.chatanywhere.tech/v1/chat/completions")
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

def act_deepseek(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/chat/completions")
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
        content = "".join([blk.get("text", "") for blk in data.get("content", [])])
        return content

    return _retry_request(_call)

# =============================
# —— 对局执行 & CSV ——
# =============================

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

def run_match(
    *,
    game_type: str,
    horizon_type: str,
    T: Optional[int],
    p: Optional[float],
    player_1: Callable,
    player_2: Callable,
    reasoning_mode: str,
    seed: int,
    max_round: int,
    show_T_in_prompt: bool,
    show_p_in_prompt: bool,
    match_id: str
) -> List[List[Any]]:
    random.seed(seed)
    np.random.seed(seed)

    prompt1, prompt2 = build_prompt_PDbos(
        game_type, horizon_type, T=T, p=p,
        show_T_in_prompt=show_T_in_prompt,
        show_p_in_prompt=show_p_in_prompt
    )

    rounds, ended_by = generate_round_indices(horizon_type, T=T, p=p, max_round=max_round)

    history1, history2 = "", ""
    totals = [0, 0]
    rows = []

    for i in rounds:
        q1 = prompt1 + history1 + f"\nYou are currently playing round {i}.\n"
        q2 = prompt2 + history2 + f"\nYou are currently playing round {i}.\n"

        a1 = get_action(player_1, q1, i, reasoning_mode)
        a2 = get_action(player_2, q2, i, reasoning_mode)

        r1, r2 = payoff(game_type, a1, a2)
        totals[0] += r1
        totals[1] += r2

        # 只保留 concise 历史，避免模型输出被“污染”
        history1 += f"In round {i}, you chose Option {a1}, other chose {a2}. You won {r1}.\n"
        history2 += f"In round {i}, you chose Option {a2}, other chose {a1}. You won {r2}.\n"

        rows.append([
            match_id,
            game_type, horizon_type, T, p, expected_T_from_p(p),
            int(show_T_in_prompt), int(show_p_in_prompt),
            reasoning_mode, seed, max_round, ended_by,
            getattr(player_1, "__name__", "player1"),
            getattr(player_2, "__name__", "player2"),
            i, a1, a2, r1, r2, totals[0], totals[1]
        ])

    return rows

def append_rows_to_csv(rows: List[List[Any]], out_csv: str):
    columns = [
        "match_id",
        "game_type", "horizon_type", "T", "p", "E_T",
        "T_in_prompt", "p_in_prompt",
        "reasoning_mode", "seed", "max_round", "ended_by",
        "player1", "player2",
        "round", "answer1", "answer2",
        "points1", "points2", "total1", "total2",
    ]
    df = pd.DataFrame(rows, columns=columns)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=header, index=False)
    print(f"[CSV] Appended {len(df)} rows to {out_csv}")

# =============================
# —— 交互式与批量模式 ——
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

def make_registry() -> Dict[str, Callable]:
    return {
        "openai:gpt": act_openai_gpt,
        "deepseek":   act_deepseek,
        "moonshot:kimi": act_kimi,
        "doubao":     act_doubao,
        "anthropic:claude": act_claude,
        # 你也可以在这里加入规则基线策略：
        # "J_always": lambda *_: "Option J",
        # "F_always": lambda *_: "Option F",
    }

def main():
    print("\n=== Experiment 2 — Fixed vs Continuation (交互式/批量) ===")

    agent_registry = make_registry()

    # 选择模式：交互式单局/批量 Sweep
    run_mode = choose_from_list("选择运行模式", ["interactive_single", "batch_grid"], default_index=1)

    game_type = choose_from_list("选择博弈类型", ["PD", "BoS"], default_index=0)

    if run_mode == "interactive_single":
        horizon_type = choose_from_list("选择地平线（有限轮或未知轮）", ["fixed", "continuation"], default_index=0)

        T = None
        p = None
        show_T_in_prompt = True
        show_p_in_prompt = True  # continuation: 建议始终 True

        if horizon_type == "fixed":
            T = input_int("输入固定轮数 T", default=10, min_v=1, max_v=10000)
            show_T_in_prompt = ask_yes_no("是否在提示中明确告知 T？", default_yes=True)
            max_round = T  # fixed 情况，max_round 无意义，但设为 T 便于写出
        else:
            # continuation：制度机制为真且告知 p
            if ask_yes_no("是否按 T 推导 p（p = 1 - 1/T）？", default_yes=True):
                T_for_p = input_int("输入用于推导的 T", default=10, min_v=2)
                p = 1.0 - 1.0 / T_for_p
                print(f"已设置 p = 1 - 1/T = {p:.4f}")
            else:
                p = input_float("输入继续概率 p (0<p<1)", default=0.9, min_v=1e-6, max_v=0.999999)
            show_p_in_prompt = ask_yes_no("是否把 p 写入提示中？(建议 Yes 以符合制度告知)", default_yes=True)
            max_round = input_int("未知轮安全上限（最大轮数）", default=200, min_v=1)

        # 推理范式
        modes = ["baseline"] if ask_yes_no("是否只运行 baseline？", default_yes=True) else ["baseline", "cot", "scot"]

        # 模型配对
        names = list(agent_registry.keys())
        pairing_type = choose_from_list("选择模型配对类型", ["A_vs_A (自博弈)", "A_vs_B (跨模型)"], default_index=1)
        if pairing_type.startswith("A_vs_A"):
            nameA = choose_from_list("选择模型 A", names, default_index=0)
            nameB = nameA
        else:
            nameA = choose_from_list("选择模型 A", names, default_index=0)
            nameB = choose_from_list("选择模型 B", names, default_index=1 if len(names) > 1 else 0)
        player_1 = agent_registry[nameA]
        player_2 = agent_registry[nameB]

        reps = input_int("同一 seed 重复对局次数", default=1, min_v=1)
        seed_text = input("输入随机种子列表（逗号分隔，默认 0,1,2,3,4）: ").strip()
        seeds = [0,1,2,3,4] if seed_text == "" else [int(x) for x in seed_text.split(",") if x.strip()]
        out_csv = input("输出 CSV 文件名（默认 exp2_results.csv）: ").strip() or "exp2_results.csv"

        print("\n===== 运行确认 =====")
        print("Game:", game_type)
        print("Horizon:", horizon_type, "T=", T, "p=", p, "E[T]=", None if p is None else f"{expected_T_from_p(p):.2f}")
        print("T_in_prompt:", show_T_in_prompt, "p_in_prompt:", show_p_in_prompt)
        print("Reasoning modes:", modes)
        print("Pairing:", f"{nameA} vs {nameB}")
        print("Seeds:", seeds, "reps:", reps)
        print("CSV:", out_csv)
        if not ask_yes_no("确认开始运行？", default_yes=True):
            print("已取消。"); return

        total_rows = 0
        for mode in modes:
            for sd in seeds:
                for k in range(reps):
                    match_id = f"single-{game_type}-{horizon_type}-{mode}-seed{sd}-rep{k}"
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
                        show_T_in_prompt=show_T_in_prompt,
                        show_p_in_prompt=show_p_in_prompt,
                        match_id=match_id
                    )
                    append_rows_to_csv(rows, out_csv)
                    total_rows += len(rows)
        print("\n完成。共写入行数：", total_rows)

    else:
        # ===== 批量网格：系统比较 fixed vs continuation =====
        print("\n— 批量模式：你可以设置若干 T 与 p 的列表，程序会网格地跑完 —")
        # grid for fixed
        run_fixed = ask_yes_no("是否包含 fixed（有限轮）网格？", default_yes=True)
        T_list = []
        if run_fixed:
            T_text = input("输入 T 列表（逗号分隔，默认 10,30,50,100）: ").strip()
            T_list = [10,30,50,100] if T_text == "" else [int(x) for x in T_text.split(",") if x.strip()]

        # grid for continuation
        run_cont = ask_yes_no("是否包含 continuation（未知轮）网格？", default_yes=True)
        p_list = []
        p_from_T = False
        if run_cont:
            p_from_T = ask_yes_no("continuation 的 p 是否由 T 反推（p = 1 - 1/T）？", default_yes=True)
            if p_from_T:
                print("将对上面的 T_list 逐一映射为 p=1-1/T。")
            else:
                p_text = input("输入 p 列表（逗号分隔，默认 0.8,0.9,0.95）: ").strip()
                p_list = [0.8,0.9,0.95] if p_text == "" else [float(x) for x in p_text.split(",") if x.strip()]
        max_round = input_int("未知轮安全上限（最大轮数，建议≥200）", default=200, min_v=1)

        # models & modes
        names = list(agent_registry.keys())
        nameA = choose_from_list("选择模型 A", names, default_index=0)
        nameB = choose_from_list("选择模型 B", names, default_index=1 if len(names) > 1 else 0)
        player_1 = agent_registry[nameA]
        player_2 = agent_registry[nameB]
        modes = ["baseline"] if ask_yes_no("是否只运行 baseline？", default_yes=True) else ["baseline", "cot", "scot"]

        reps = input_int("同一 seed 重复对局次数", default=1, min_v=1)
        seed_text = input("输入随机种子列表（逗号分隔，默认 0,1,2,3,4）: ").strip()
        seeds = [0,1,2,3,4] if seed_text == "" else [int(x) for x in seed_text.split(",") if x.strip()]
        out_csv = input("输出 CSV 文件名（默认 exp2_results.csv）: ").strip() or "exp2_results.csv"

        print("\n===== 运行确认（批量） =====")
        print("Game:", game_type)
        print("Fixed T_list:", T_list if run_fixed else "N/A")
        print("Continuation:", "p from T (p=1-1/T)" if (run_cont and p_from_T) else (p_list if run_cont else "N/A"))
        print("E[T] for p_list:", [f"{expected_T_from_p(p):.1f}" for p in ([(1-1/t) for t in T_list] if (run_cont and p_from_T) else p_list)] if run_cont else "N/A")
        print("max_round:", max_round)
        print("Modes:", modes)
        print("Pairing:", f"{nameA} vs {nameB}")
        print("Seeds:", seeds, "reps:", reps)
        print("CSV:", out_csv)
        if not ask_yes_no("确认开始运行？", default_yes=True):
            print("已取消。"); return

        total_rows = 0
        # fixed grid
        if run_fixed:
            for T in T_list:
                for mode in modes:
                    for sd in seeds:
                        for k in range(reps):
                            match_id = f"grid-fixed-T{T}-{game_type}-{mode}-seed{sd}-rep{k}"
                            rows = run_match(
                                game_type=game_type,
                                horizon_type="fixed",
                                T=T,
                                p=None,
                                player_1=player_1,
                                player_2=player_2,
                                reasoning_mode=mode,
                                seed=sd,
                                max_round=T,
                                show_T_in_prompt=True,   # fixed：默认清楚告知 T
                                show_p_in_prompt=False,  # 无需告知 p
                                match_id=match_id
                            )
                            append_rows_to_csv(rows, out_csv)
                            total_rows += len(rows)

        # continuation grid
        if run_cont:
            the_p_list = [(1.0 - 1.0 / T) for T in T_list] if p_from_T else p_list
            for p in the_p_list:
                for mode in modes:
                    for sd in seeds:
                        for k in range(reps):
                            match_id = f"grid-cont-p{p:.3f}-{game_type}-{mode}-seed{sd}-rep{k}"
                            rows = run_match(
                                game_type=game_type,
                                horizon_type="continuation",
                                T=None,
                                p=p,
                                player_1=player_1,
                                player_2=player_2,
                                reasoning_mode=mode,
                                seed=sd,
                                max_round=max_round,
                                show_T_in_prompt=False,  # continuation 无 T
                                show_p_in_prompt=True,   # 制度机制明确告知 p（本实验核心）
                                match_id=match_id
                            )
                            append_rows_to_csv(rows, out_csv)
                            total_rows += len(rows)

        print("\n完成（批量）。共写入行数：", total_rows)

if __name__ == "__main__":
    main()
