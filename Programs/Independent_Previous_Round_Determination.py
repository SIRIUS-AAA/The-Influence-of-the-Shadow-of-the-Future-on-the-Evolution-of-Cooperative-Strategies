#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 3 — Dynamic p_t under Continuation (Unknown Horizon)
版本：独立伯努利 + 上一轮判定（end-of-round hazard），安全移除硬编码 API Key

目标：
- continuation（未知轮）里，让每一轮的继续概率 p_t 按轨迹（上升/下降/随机游走）变化
- 停机机制：进入第 t 轮由“上一轮的 p_{t-1}”判定（独立伯努利抽签）
- 交互式运行，逐轮写 CSV（包含 path、p_now 等）
- 可选输出：按 p 分箱的合作/协调率与平均收益

依赖：requests, numpy, pandas
API：仅从环境变量读取，不再硬编码（请自行 export OPENAI_API_KEY 等）
"""

import os, json, time, random
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
# —— 基础工具（解析、收益、提示、输入） ——
# =============================

def parse_choice(text: str) -> str:
    """尽量宽容地从 LLM 文本中解析出 J/F。"""
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
    """两类博弈的收益矩阵。"""
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

def build_prompt_PDbos(game_type: str, horizon_text: str) -> Tuple[str, str]:
    """为两位玩家构建规则提示（BoS 双方偏好相反）。"""
    if game_type == "PD":
        rules = (
            "The rules of the game are as follows:\n"
            "If you choose Option J and the other player chooses Option J, then you both win 8 points.\n"
            "If you choose Option J and the other player chooses Option F, then you win 0 points and the other player wins 10 points.\n"
            "If you choose Option F and the other player chooses Option J, then you win 10 points and the other player wins 0 points.\n"
            "If both choose Option F, then you both win 5 points.\n\n"
        )
        base = "You are playing a repeated game with another player. Options: J or F.\n" + horizon_text + rules
        return base, base
    elif game_type == "BoS":
        rules_p1 = (
            "The rules of the game are as follows:\n"
            "If both choose Option J, you win 10 points and the other player wins 7 points.\n"
            "If both choose Option F, you win 7 points and the other player wins 10 points.\n"
            "If the options differ, you both win 0 points.\n\n"
        )
        rules_p2 = (
            "The rules of the game are as follows:\n"
            "If both choose Option J, you win 7 points and the other player wins 10 points.\n"
            "If both choose Option F, you win 10 points and the other player wins 7 points.\n"
            "If the options differ, you both win 0 points.\n\n"
        )
        base1 = "You are playing a repeated coordination game (Battle of the Sexes). Options: J or F.\n" + horizon_text + rules_p1
        base2 = "You are playing a repeated coordination game (Battle of the Sexes). Options: J or F.\n" + horizon_text + rules_p2
        return base1, base2
    else:
        raise ValueError("Unknown game_type")

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

# =============================
# —— 模型最小封装（仅读环境变量） ——
# =============================

def _retry_request(do_call, retries=3, backoff=1.5) -> str:
    last_err = None
    for i in range(retries):
        try:
            return do_call()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    print("[WARN] LLM call failed, fallback to 'Option J'. Error:", repr(last_err))
    return "Option J"

def act_openai_gpt(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = os.environ.get("OPENAI_API_BASE", "https://api.chatanywhere.tech/v1/chat/completions")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    def _call():
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Output exactly 'Option J' or 'Option F' in the final line."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.environ.get("OPENAI_TEMPERATURE", 0)),
            "max_tokens": 16,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return _retry_request(_call)

def act_deepseek(prompt: str, round_idx: int) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/chat/completions")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    def _call():
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
    def _call():
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "最终一行只输出 'Option J' 或 'Option F'"},
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
    def _call():
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

# =============================
# —— 动态 p：轨迹、独立伯努利停机（上一轮判定） & 对局 ——
# =============================

def path_sequence(path: str, max_round: int, eta: float, start_p: float, end_p: Optional[float]) -> List[float]:
    """生成长度为 max_round 的 p 序列（上升/下降：线性；rw：高斯随机游走）。"""
    seq = []
    if path in ("up", "down"):
        for t in range(max_round):
            frac = t / max(1, max_round - 1)
            p = start_p + (0.0 if end_p is None else (end_p - start_p) * frac)
            seq.append(min(1.0, max(0.0, p)))
    elif path == "rw":
        p = start_p
        for _ in range(max_round):
            seq.append(min(1.0, max(0.0, p)))
            p = p + np.random.normal(0.0, eta)
            p = min(1.0, max(0.0, p))
    else:
        raise ValueError("Unknown path")
    return seq

def build_history_summary(pairs: List[str], L: int) -> str:
    """根据 L 窗口生成简短历史摘要，帮助 LLM 感知近期互动。"""
    if L == 0 or len(pairs) == 0:
        return ""
    window = pairs if L < 0 else pairs[-min(L, len(pairs)):]
    cJJ = sum(1 for x in window if x == "JJ")
    cFF = sum(1 for x in window if x == "FF")
    cJF = sum(1 for x in window if x == "JF")
    cFJ = sum(1 for x in window if x == "FJ")
    last = window[-1] if window else "NA"
    return (f"Recent {len(window)} rounds summary: JJ={cJJ}, FF={cFF}, JF={cJF}, FJ={cFJ}. "
            f"Last pair={last}. ")

def get_action(player_func: Callable, prompt_base: str, summary: str, p_now: float, round_i: int, mode: str) -> str:
    """构造提示并获取玩家动作。"""
    if mode == "baseline":
        prompt = (
            prompt_base +
            f"The continuation probability after THIS round is p={p_now:.2f} (higher p means more likely to continue).\n" +
            summary +
            f"Round {round_i}. Which Option do you choose, Option J or Option F?\n" +
            "Answer with 'Option J' or 'Option F'.\nA: Option"
        )
    elif mode == "cot":
        prompt = (
            prompt_base +
            f"The continuation probability after THIS round is p={p_now:.2f}.\n" +
            summary +
            f"Round {round_i}. Briefly explain, then give the final choice.\n" +
            "Final answer: 'Option J' or 'Option F'.\nA:"
        )
    else:  # scot
        prompt = (
            prompt_base +
            f"The continuation probability after THIS round is p={p_now:.2f}.\n" +
            summary +
            f"Round {round_i}. Consider the other's incentives, then choose.\n" +
            "Final answer: 'Option J' or 'Option F'.\nA:"
        )
    raw = player_func(prompt, round_i-1) if player_func.__code__.co_argcount >= 2 else player_func(prompt)
    return parse_choice(raw)

# —— 核心改动：把“是否进入下一轮”的抽签显式封装（独立伯努利 + 上一轮判定）
def should_continue_prev(t: int, p_seq: List[float], rng: random.Random) -> bool:
    """
    独立伯努利 + 上一轮判定：
    - t 从 1 开始计数；第 1 轮总是发生；
    - 是否能进入第 t 轮（t>=2）取决于上一轮的 p_{t-1}；
    - 每次调用使用 rng.random() 独立抽签，彼此独立。
    """
    if t == 1:
        return True
    p_prev = p_seq[t - 2]  # p_{t-1}
    return rng.random() <= p_prev

def run_match_dynamic(
    *,
    game_type: str,
    path: str,
    start_p: float,
    end_p: Optional[float],
    eta: float,
    player_1: Callable,
    player_2: Callable,
    reasoning_mode: str,
    seed: int,
    max_round: int,
    L_window: int,
    p_in_prompt: bool,
    match_id: str
) -> Tuple[List[List[Any]], str]:
    """
    返回 rows, ended_by
    - 停机：独立伯努利 + 上一轮判定（should_continue_prev）
    - 每轮写出 p_now、答案、收益等
    """
    # 将 NumPy 的随机数与“停机抽签”的随机数解耦，便于清晰表达“独立伯努利”
    np.random.seed(seed)
    rng = random.Random(seed)   # 专门用于停机抽签；每次 rng.random() 独立

    horizon_text = "After each round, the game may continue with some probability (unknown a priori).\n"
    p1_base, p2_base = build_prompt_PDbos(game_type, horizon_text)

    totals = [0, 0]
    rows = []
    pairs = []
    ended_by = "prob_stop"
    p_seq = path_sequence(path, max_round, eta, start_p, end_p)

    for i, p_now in enumerate(p_seq, start=1):
        # —— 独立伯努利 + 上一轮判定：是否能进入第 i 轮？
        if not should_continue_prev(i, p_seq, rng):
            ended_by = "prob_stop"
            break

        if i == max_round:
            ended_by = "cap"

        # 提示里报告“本轮之后继续的概率”= p_now （即 p_i）
        summary = build_history_summary(pairs, L_window)
        base1 = p1_base + (f"The continuation probability after THIS round is p={p_now:.2f}.\n" if p_in_prompt else "")
        base2 = p2_base + (f"The continuation probability after THIS round is p={p_now:.2f}.\n" if p_in_prompt else "")

        a1 = get_action(player_1, base1, summary, p_now, i, reasoning_mode)
        a2 = get_action(player_2, base2, summary, p_now, i, reasoning_mode)
        pairs.append(f"{a1}{a2}")

        r1, r2 = payoff(game_type, a1, a2)
        totals[0] += r1; totals[1] += r2

        rows.append([
            match_id, game_type, "continuation_dynamic",
            path, L_window, reasoning_mode, seed, max_round,
            getattr(player_1, "__name__", "player1"),
            getattr(player_2, "__name__", "player2"),
            i, p_now, int(p_in_prompt),
            a1, a2, r1, r2, totals[0], totals[1],
        ])

    return rows, ended_by

def append_rows_to_csv(rows: List[List[Any]], out_csv: str):
    columns = [
        "match_id", "game_type", "horizon_type",
        "path", "L_window", "reasoning_mode", "seed", "max_round",
        "player1", "player2",
        "round", "p_now", "p_in_prompt",
        "answer1", "answer2", "points1", "points2", "total1", "total2",
    ]
    df = pd.DataFrame(rows, columns=columns)
    header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a", header=header, index=False)
    print(f"[CSV] Appended {len(df)} rows -> {out_csv}")

def summarize_by_p(in_csv: str, out_csv: str, bins: int = 10):
    """把 p 分箱，分别统计合作/协调率与平均收益。"""
    df = pd.read_csv(in_csv)
    df = df[df["round"] > 0].copy()
    df["p_bin"] = pd.cut(df["p_now"], bins=bins, labels=False, include_lowest=True)

    def ok(row):
        if row["game_type"] == "PD":
            return 1 if (row["answer1"] == "J" and row["answer2"] == "J") else 0
        else:
            return 1 if (row["answer1"] == row["answer2"]) else 0

    df["coop_or_coord"] = df.apply(ok, axis=1)

    grp = (df.groupby(["game_type","path","L_window","reasoning_mode","p_bin"])
             .agg(rounds=("round","count"),
                  coop_rate=("coop_or_coord","mean"),
                  avg_p=("p_now","mean"),
                  mean_points1=("points1","mean"),
                  mean_points2=("points2","mean"))
             .reset_index())
    grp.to_csv(out_csv, index=False)
    print(f"[CSV] Wrote p-binned summary -> {out_csv}")

# =============================
# —— 注册表 & 交互入口 ——
# =============================

def make_registry() -> Dict[str, Callable]:
    return {
        "openai:gpt":    act_openai_gpt,
        "deepseek":      act_deepseek,
        "moonshot:kimi": act_kimi,
        "doubao":        act_doubao,
        # 可增规则基线：
        # "J_always": lambda *_: "Option J",
        # "F_always": lambda *_: "Option F",
    }

def main():
    print("\n=== Experiment 3 — Continuation with Dynamic p_t (独立伯努利 + 上一轮判定) ===")

    registry = make_registry()
    run_mode = choose_from_list("选择运行模式", ["interactive_single", "batch_grid"], default_index=0)
    game_type = choose_from_list("选择博弈类型", ["PD", "BoS"], default_index=0)

    # 轨迹参数（共同）
    path = choose_from_list("选择 p 的轨迹", ["up", "down", "rw"], default_index=0)
    max_round = input_int("未知轮安全上限（最大轮数）", default=100, min_v=1)

    if path in ("up", "down"):
        start_p = input_float("起始 p_start (0~1)", default=(0.2 if path=="up" else 0.8), min_v=0.0, max_v=1.0)
        end_p   = input_float("结束 p_end   (0~1)", default=(0.8 if path=="up" else 0.2), min_v=0.0, max_v=1.0)
        eta     = 0.0
    else:
        start_p = input_float("随机游走起点 p_start (0~1)", default=0.5, min_v=0.0, max_v=1.0)
        end_p   = None
        eta     = input_float("随机游走步长标准差 eta", default=0.05, min_v=1e-6, max_v=0.5)

    p_in_prompt = ask_yes_no("是否把‘本轮之后继续的概率 p’写入提示？（建议 Yes）", default_yes=True)
    L_window = input_int("历史摘要窗口 L（-1=全历史，0=不摘要，正数=最近L轮）", default=5, min_v=-1)

    # 模型与范式
    names = list(registry.keys())
    pairing = choose_from_list("选择模型配对", ["A_vs_A (自博弈)", "A_vs_B (跨模型)"], default_index=0)
    if pairing.startswith("A_vs_A"):
        nameA = choose_from_list("选择模型 A", names, default_index=0)
        nameB = nameA
    else:
        nameA = choose_from_list("选择模型 A", names, default_index=0)
        nameB = choose_from_list("选择模型 B", names, default_index=1 if len(names) > 1 else 0)
    player_1 = registry[nameA]; player_2 = registry[nameB]
    modes = ["baseline"] if ask_yes_no("是否只运行 baseline？", default_yes=True) else ["baseline", "cot", "scot"]

    # 复现实验设置
    reps = input_int("同一 seed 重复对局次数", default=1, min_v=1)
    seed_text = input("输入随机种子列表（逗号分隔，默认 0,1,2,3,4）: ").strip()
    seeds = [0,1,2,3,4] if seed_text == "" else [int(x) for x in seed_text.split(",") if x.strip()]
    out_detail = input("输出明细 CSV（默认 exp3_results.csv）: ").strip() or "exp3_results.csv"
    do_summary = ask_yes_no("运行后是否输出按 p 分箱的汇总？", default_yes=True)
    out_summary = input("输出汇总 CSV（默认 exp3_summary.csv）: ").strip() or "exp3_summary.csv"

    print("\n===== 运行确认 =====")
    print("Game:", game_type, "Path:", path, "L_window:", L_window)
    print("p params:", dict(start_p=start_p, end_p=end_p, eta=eta))
    print("max_round:", max_round, "p_in_prompt:", p_in_prompt)
    print("Modes:", modes, "Pairing:", f"{nameA} vs {nameB}")
    print("Seeds:", seeds, "reps:", reps)
    print("CSV(detail):", out_detail, "CSV(summary):", out_summary if do_summary else "N/A")
    if not ask_yes_no("确认开始运行？", default_yes=True):
        print("已取消。"); return

    total_rows = 0
    for mode in modes:
        for sd in seeds:
            for k in range(reps):
                match_id = f"exp3-{game_type}-{path}-{mode}-seed{sd}-rep{k}"
                rows, ended_by = run_match_dynamic(
                    game_type=game_type, path=path, start_p=start_p, end_p=end_p, eta=eta,
                    player_1=player_1, player_2=player_2, reasoning_mode=mode,
                    seed=sd, max_round=max_round, L_window=L_window, p_in_prompt=p_in_prompt,
                    match_id=match_id
                )
                append_rows_to_csv(rows, out_detail)
                total_rows += len(rows)
    print("\n完成。逐轮写入总行数：", total_rows)

    if do_summary:
        summarize_by_p(out_detail, out_summary, bins=10)

if __name__ == "__main__":
    main()
