import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

print("=== [Pipeline V2 Step 3.5] Day=0 双源与录入错误核验 ===")

# ============================== 路径配置 ==============================
# 可通过命令行传入任意流程后的 CSV：
#   python 03_5_day0双源核验_v2.py "HIS系统\清洗流程_v2\04_孕前与早孕异动修复版.csv"
DEFAULT_INPUT_CSV = r"03_全局与阶跃修复版.csv"

# 初检原始文件：用于读取「孕前体重」「体重」和「身高」。BMI 核验只使用初检身高。
INIT_FILES = [
    r"..\初检\孕妇初检-第五批-to liu-20260319.xlsx",
    r"..\初检\孕期初检-第1-4批-to liu-20260319.xlsx",
]

DEFAULT_OUT_DIR = r"."
DEFAULT_OUT_CSV = os.path.join(DEFAULT_OUT_DIR, "03_5_day0双源核验版.csv")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_OUT_DIR, "03_5_day0双源核验_日志.txt")

# ============================== 参数 ==============================
EARLY_DAY_MAX = 98        # 孕早期上限（天），用于寻找最近早孕参照点
SAME_WEIGHT_TOL = 0.5     # 认定两个体重几乎相同的阈值（kg）
ALT_MAX_DIFF = 5.0        # 早孕重复时，另一候选与最接近候选差距不超过该值才可优先采用
BMI_MIN = 14.0            # 生理可信 BMI 下限
BMI_MAX = 50.0            # 生理可信 BMI 上限
JIN_MIN_WEIGHT = 75.0     # 大于该体重且 /2 后 BMI 合理时，才考虑斤/公斤录入错误


def parse_args():
    parser = argparse.ArgumentParser(
        description="Day=0 双源核验：按孕早期参照从初检孕前体重/建册体重中重选孕前体重。"
    )
    parser.add_argument("input_csv", nargs="?", default=None, help="任意流程后的输入 CSV。")
    parser.add_argument("--input", dest="input_opt", default=None, help="输入 CSV（等价于位置参数）。")
    parser.add_argument("--output", default=None, help="输出 CSV；不填则使用默认输出。")
    parser.add_argument("--log", default=None, help="日志文件；不填则使用默认日志。")
    return parser.parse_args()


def safe_float(v):
    try:
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def normalize_height_cm(v):
    h = safe_float(v)
    if np.isnan(h):
        return np.nan
    if 1.0 < h < 3.0:
        return round(h * 100, 1)
    if 100 <= h <= 230:
        return round(h, 1)
    return np.nan


def calc_bmi(weight_kg, height_cm):
    w = safe_float(weight_kg)
    h = safe_float(height_cm)
    if np.isnan(w) or np.isnan(h) or h <= 0:
        return np.nan
    return round(w / ((h / 100.0) ** 2), 2)


def classify_weight(raw_weight, height_cm):
    """返回候选体重的录入状态。BMI 核验仅基于初检身高。"""
    raw = safe_float(raw_weight)
    h = safe_float(height_cm)
    if np.isnan(raw):
        return {
            "status": "missing",
            "usable": False,
            "kg": np.nan,
            "bmi": np.nan,
            "reason": "体重缺失",
        }
    if np.isnan(h):
        return {
            "status": "no_init_height",
            "usable": False,
            "kg": raw,
            "bmi": np.nan,
            "reason": "初检身高缺失，无法核验BMI",
        }

    bmi_raw = calc_bmi(raw, h)
    if BMI_MIN <= bmi_raw <= BMI_MAX:
        return {
            "status": "valid_kg",
            "usable": True,
            "kg": raw,
            "bmi": bmi_raw,
            "reason": f"BMI={bmi_raw:.1f}，kg值可信",
        }

    half = round(raw / 2.0, 2)
    bmi_half = calc_bmi(half, h)
    if raw >= JIN_MIN_WEIGHT and BMI_MIN <= bmi_half <= BMI_MAX:
        return {
            "status": "likely_jin",
            "usable": True,
            "kg": half,
            "bmi": bmi_half,
            "reason": f"原值BMI={bmi_raw:.1f}异常，/2后BMI={bmi_half:.1f}，疑似斤录入",
        }

    return {
        "status": "invalid_bmi",
        "usable": False,
        "kg": raw,
        "bmi": bmi_raw,
        "reason": f"BMI={bmi_raw:.1f}超出生理范围[{BMI_MIN},{BMI_MAX}]",
    }


def source_label(src):
    return {
        "pre_weight": "孕前体重",
        "weight_col": "建册体重",
    }.get(src, src or "unknown")


def find_day0_row(group):
    day0_mask = group["gestation_day"] == 0
    if "type" in group.columns:
        initial_mask = day0_mask & (group["type"].astype(str) == "Initial_Raw")
        if initial_mask.any():
            return group[initial_mask].index[0]
    if day0_mask.any():
        return group[day0_mask].index[0]
    return None


def find_nearest_early(group):
    early_mask = (group["gestation_day"] > 0) & (group["gestation_day"] <= EARLY_DAY_MAX)
    early_rows = group[early_mask].sort_values("gestation_day")
    if early_rows.empty:
        return np.nan, np.nan
    for _, row in early_rows.iterrows():
        w = safe_float(row.get("weight", np.nan))
        if not np.isnan(w):
            return int(row["gestation_day"]), w
    return np.nan, np.nan


def build_init_candidates(init_info):
    candidates = {}
    for src in ["pre_weight", "weight_col"]:
        raw = init_info.get(src, np.nan)
        audit = classify_weight(raw, init_info.get("height_cm", np.nan))
        if not audit["usable"]:
            continue
        candidates[src] = {
            "src": src,
            "kg": audit["kg"],
            "audit": audit,
        }
    return candidates


def choose_prepreg_weight(init_info, nearest_weight):
    """
    按新规则从初检双源中选择 day=0 孕前体重：
    1. 先选择最靠近孕早期体重的候选。
    2. 如果最接近候选与孕早期体重几乎相同，且另一候选与它相差不超过 5kg，
       优先选择与孕早期体重不同的另一候选，避免复制早孕体重。
    3. 其他情况选择最接近孕早期体重的候选。
    """
    candidates = build_init_candidates(init_info)
    if not candidates:
        return None
    if np.isnan(nearest_weight):
        return None
    if len(candidates) == 1:
        only = next(iter(candidates.values()))
        return {
            **only,
            "decision": "single_usable_candidate",
            "distance_to_early": abs(only["kg"] - nearest_weight),
        }

    ranked = sorted(
        candidates.values(),
        key=lambda c: (abs(c["kg"] - nearest_weight), 0 if c["src"] == "pre_weight" else 1),
    )
    nearest = ranked[0]
    other = ranked[1]
    nearest_same_as_early = abs(nearest["kg"] - nearest_weight) < SAME_WEIGHT_TOL
    other_same_as_early = abs(other["kg"] - nearest_weight) < SAME_WEIGHT_TOL
    candidates_close = abs(other["kg"] - nearest["kg"]) <= ALT_MAX_DIFF

    if nearest_same_as_early and not other_same_as_early and candidates_close:
        return {
            **other,
            "decision": "avoid_early_duplicate_within_5kg",
            "distance_to_early": abs(other["kg"] - nearest_weight),
        }

    return {
        **nearest,
        "decision": "nearest_to_early",
        "distance_to_early": abs(nearest["kg"] - nearest_weight),
    }


def load_init_lookup():
    print("正在加载初检数据（取身高、孕前体重、建册体重）...")
    init_frames = []
    for f in INIT_FILES:
        if os.path.exists(f):
            init_frames.append(pd.read_excel(f))
        else:
            print(f"  警告：初检文件不存在 {f}")

    if not init_frames:
        raise FileNotFoundError("所有初检文件均不存在，无法进行 day=0 双源核验。")

    init_df = pd.concat(init_frames, ignore_index=True)
    init_df["项目流水号"] = init_df["项目流水号"].astype(str).str.strip()
    init_df = init_df.drop_duplicates("项目流水号")
    print(f"  初检记录共 {len(init_df)} 条（去重后）")

    lookup = {}
    for _, row in init_df.iterrows():
        nid = str(row["项目流水号"]).strip()
        lookup[nid] = {
            "pre_weight": safe_float(row.get("孕前体重", np.nan)),
            "weight_col": safe_float(row.get("体重", np.nan)),
            "height_cm": normalize_height_cm(row.get("身高", np.nan)),
        }
    return lookup


def ensure_audit_columns(df):
    audit_cols = {
        "day0_audit_flag": "",
        "day0_audit_reason": "",
        "day0_init_height_cm": np.nan,
        "day0_pre_weight_raw": np.nan,
        "day0_weight_col_raw": np.nan,
        "day0_nearest_early_day": np.nan,
        "day0_nearest_early_weight": np.nan,
    }
    for col, default in audit_cols.items():
        if col not in df.columns:
            df[col] = default
    return df


def main():
    args = parse_args()
    input_csv = args.input_opt or args.input_csv or DEFAULT_INPUT_CSV
    output_csv = args.output or DEFAULT_OUT_CSV
    log_file = args.log or DEFAULT_LOG_FILE
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    init_lookup = load_init_lookup()

    print(f"正在加载: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    required_cols = {"项目流水号", "gestation_day", "weight"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"输入表缺少必要列: {sorted(missing_cols)}")

    df["项目流水号"] = df["项目流水号"].astype(str).str.strip()
    df = ensure_audit_columns(df)
    print(f"  共 {len(df)} 行，{df['项目流水号'].nunique()} 个样本")

    logs_all = []
    fixed_count = 0
    audit_count = 0
    skipped_count = 0
    frames = []

    grouped = df.sort_values(["项目流水号", "gestation_day"]).groupby("项目流水号", observed=True)
    total = df["项目流水号"].nunique()
    for i, (nid, group) in enumerate(grouped):
        if i % 20000 == 0:
            print(f"  3.5 处理进度: {i}/{total}...")

        group = group.copy()
        day0_idx = find_day0_row(group)
        if day0_idx is None:
            frames.append(group)
            skipped_count += 1
            continue

        init_info = init_lookup.get(str(nid), {})
        init_height = init_info.get("height_cm", np.nan)
        pre_raw = init_info.get("pre_weight", np.nan)
        weight_raw = init_info.get("weight_col", np.nan)
        day0_weight = safe_float(group.at[day0_idx, "weight"])
        day0_source = str(group.at[day0_idx, "weight_source"]) if "weight_source" in group.columns else ""
        nearest_day, nearest_weight = find_nearest_early(group)

        group.at[day0_idx, "day0_init_height_cm"] = init_height
        group.at[day0_idx, "day0_pre_weight_raw"] = pre_raw
        group.at[day0_idx, "day0_weight_col_raw"] = weight_raw
        group.at[day0_idx, "day0_nearest_early_day"] = nearest_day
        group.at[day0_idx, "day0_nearest_early_weight"] = nearest_weight

        if np.isnan(day0_weight):
            group.at[day0_idx, "day0_audit_flag"] = "day0_missing"
            group.at[day0_idx, "day0_audit_reason"] = "day=0体重缺失"
            frames.append(group)
            audit_count += 1
            continue

        current_audit = classify_weight(day0_weight, init_height)
        pre_audit = classify_weight(pre_raw, init_height)
        weight_audit = classify_weight(weight_raw, init_height)

        reasons = [
            f"初检身高={init_height if not np.isnan(init_height) else 'NA'}cm",
            f"当前day0={day0_weight:.1f}kg({current_audit['reason']})",
            f"孕前体重={pre_raw if not np.isnan(pre_raw) else 'NA'}({pre_audit['status']})",
            f"建册体重={weight_raw if not np.isnan(weight_raw) else 'NA'}({weight_audit['status']})",
        ]
        if not np.isnan(nearest_weight):
            reasons.append(f"早孕参照=day{nearest_day}d {nearest_weight:.1f}kg")

        flag = "ok"
        new_weight = None
        new_source = None

        selected = choose_prepreg_weight(init_info, nearest_weight)
        if selected is None:
            if np.isnan(nearest_weight):
                flag = "no_early_reference"
                reasons.append("无可用孕早期参照，无法按双源距离规则选择孕前体重")
            else:
                flag = "no_usable_init_candidate"
                reasons.append("初检孕前体重和建册体重均不可用，无法替换day=0")
        else:
            selected_src = selected["src"]
            selected_weight = selected["kg"]
            selected_audit = selected["audit"]
            decision = selected["decision"]
            reasons.append(
                f"双源选择={source_label(selected_src)} {selected_weight:.1f}kg"
                f"({decision}, 距早孕={selected['distance_to_early']:.1f}kg; {selected_audit['reason']})"
            )

            if abs(selected_weight - day0_weight) >= SAME_WEIGHT_TOL:
                new_weight = selected_weight
                new_source = (
                    selected_src
                    if selected_audit["status"] == "valid_kg"
                    else f"{selected_src}_unit_corrected"
                )
                flag = "day0_reselected_from_init"
            elif decision == "avoid_early_duplicate_within_5kg":
                flag = "day0_already_selected_nonduplicate"
            elif not np.isnan(nearest_weight) and abs(day0_weight - nearest_weight) < SAME_WEIGHT_TOL:
                flag = "day0_same_as_early_but_selected"
                reasons.append("双源规则仍选择最接近项，因另一候选缺失、不可用、同样重复或差距超过5kg")

        # 初检源中存在疑似斤录入或 BMI 异常，也记录到日志，方便人工核对原始表。
        init_source_errors = []
        for src, audit in [("pre_weight", pre_audit), ("weight_col", weight_audit)]:
            if audit["status"] in {"likely_jin", "invalid_bmi"}:
                init_source_errors.append(f"{source_label(src)}:{audit['reason']}")
        if init_source_errors and flag == "ok":
            flag = "init_candidate_input_error"
        if init_source_errors:
            reasons.extend(init_source_errors)

        if new_weight is not None:
            old_weight = day0_weight
            group.at[day0_idx, "weight"] = round(float(new_weight), 2)
            if "weight_source" in group.columns:
                group.at[day0_idx, "weight_source"] = new_source
            new_bmi = calc_bmi(new_weight, init_height)
            if not np.isnan(new_bmi) and "BMI" in group.columns:
                group.at[day0_idx, "BMI"] = new_bmi
            fixed_count += 1
            reasons.append(f"修正day=0: {old_weight:.1f}->{new_weight:.1f}kg")

        group.at[day0_idx, "day0_audit_flag"] = flag
        group.at[day0_idx, "day0_audit_reason"] = " | ".join(reasons)

        if flag != "ok":
            audit_count += 1
            logs_all.append(f"[{nid}] {flag} | " + " | ".join(reasons))

        frames.append(group)

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    log_content = (
        f"Step 3.5 日志 ({datetime.now()})\n"
        + "=" * 80
        + "\n"
        + f"输入: {input_csv}\n"
        + f"输出: {output_csv}\n"
        + f"修正 day=0: {fixed_count} 例\n"
        + f"审计标记: {audit_count} 例\n"
        + f"无 day=0 跳过: {skipped_count} 例\n"
        + "=" * 80
        + "\n"
        + "\n".join(logs_all)
    )
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)

    print("\nSTEP 3.5 完成！")
    print(f"  修正 day=0: {fixed_count} 例")
    print(f"  审计标记: {audit_count} 例")
    print(f"  无 day=0 跳过: {skipped_count} 例")
    print(f"  输出 -> {output_csv}")
    print(f"  日志 -> {log_file}")


if __name__ == "__main__":
    main()
