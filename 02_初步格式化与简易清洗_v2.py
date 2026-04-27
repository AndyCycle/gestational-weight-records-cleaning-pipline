import pandas as pd
import numpy as np
import os
import math
from datetime import datetime

print("=== [Pipeline V2 Step 2] 初步格式化与简易清洗 + day=0 极端BMI兜底 ===")

INPUT_CSV = r"01_合并后底表_带初检.csv"
OUT_DIR   = r"."
OUT_CSV   = os.path.join(OUT_DIR, "02_初步清洗_去低级失误版.csv")
LOG_FILE  = os.path.join(OUT_DIR, "02_初步清洗_日志.txt")

# =================== 简易清洗（原逻辑保留不变）===================

def simple_clean(group):
    """处理单一产妇的纯量级录入错漏。"""
    w_orig = group['weight'].values.copy()
    days   = group['gestation_day'].values
    n      = len(w_orig)
    logs   = []
    changed = False

    for i in range(n):
        curr_w = w_orig[i]
        if pd.isna(curr_w):
            continue

        # 寻找最近邻锚点
        anchors = []
        for j in range(i - 1, -1, -1):
            if not pd.isna(w_orig[j]):
                anchors.append(w_orig[j])
                break
        for j in range(i + 1, n):
            if not pd.isna(w_orig[j]):
                anchors.append(w_orig[j])
                break

        if not anchors:
            continue
        ref_w = np.mean(anchors)

        # 1. 键盘敲多一个 0 溢出（例如 500）
        if curr_w >= 280:
            cw_div10 = curr_w / 10.0
            if abs(cw_div10 - ref_w) < 25:
                w_orig[i] = round(cw_div10, 2)
                logs.append(
                    f"Day {days[i]}d: 多0溢出 | {curr_w} -> {w_orig[i]} "
                    f"(邻居参考 {ref_w:.1f})"
                )
                changed = True
                continue

        # 2. 漏敲十位（如 6.5，真实应为 56.5）
        if curr_w < 25 and ref_w >= 40:
            c_mod    = curr_w % 10
            ref_tens = (int(ref_w) // 10) * 10
            cands    = [
                ref_tens + c_mod,
                (ref_tens - 10) + c_mod,
                (ref_tens + 10) + c_mod,
            ]
            best_c = min(cands, key=lambda x: abs(x - ref_w))
            if abs(best_c - ref_w) < 8:
                w_orig[i] = round(best_c, 2)
                logs.append(
                    f"Day {days[i]}d: 丢十位数 | {curr_w} -> {w_orig[i]} "
                    f"(邻居参考 {ref_w:.1f})"
                )
                changed = True
                continue

    group['weight_cleaned'] = w_orig
    return group, logs, changed


# =================== 新增：day=0 极端 BMI 兜底 ===================

def drop_extreme_day0(df, bmi_low=10.0, bmi_high=50.0):
    """
    对所有 gestation_day == 0 的行，计算 BMI。
    若 BMI < bmi_low 或 BMI > bmi_high，则删除该行（仅 day=0）。
    返回 (cleaned_df, drop_log_list)
    """
    day0_mask = df['gestation_day'] == 0
    drop_logs = []
    drop_idx  = []

    for idx, row in df[day0_mask].iterrows():
        w = row['weight']
        h = row['height']
        if pd.isna(w) or pd.isna(h):
            continue
        h_m = h / 100.0 if h > 3 else h
        if h_m <= 0:
            continue
        bmi = round(w / (h_m * h_m), 2)
        if bmi < bmi_low or bmi > bmi_high:
            drop_idx.append(idx)
            drop_logs.append(
                f"[{row['项目流水号']}] day=0 极端BMI删除 | "
                f"BMI={bmi:.2f}, weight={w:.1f}kg, height={h:.1f}"
            )

    cleaned_df = df.drop(index=drop_idx)
    return cleaned_df, drop_logs


# =================== 主函数 ===================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到输入文件: {INPUT_CSV}")
        return

    print(f"正在加载: {INPUT_CSV}")
    df     = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    df[id_col] = df[id_col].astype(str).str.strip()

    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)

    frames        = []
    all_logs      = []
    total         = len(grouped)
    changed_count = 0

    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  进度: {i}/{total}...")
        c_group, logs, changed = simple_clean(group.copy())
        frames.append(c_group)
        if changed:
            changed_count += 1
            all_logs.append(f"[{nid}] 修正明细:")
            all_logs.extend(["  " + l for l in logs])

    final_df = pd.concat(frames, ignore_index=True)
    # 用清理后体重覆盖原值
    final_df.rename(
        columns={'weight': 'weight_raw_p1', 'weight_cleaned': 'weight'},
        inplace=True
    )

    print(f"简易清洗完成，共修复 {changed_count} 例。")
    print("正在进行 day=0 极端BMI兜底删除...")

    final_df, drop_logs = drop_extreme_day0(final_df)
    all_logs = drop_logs + ["--- 以下为简易清洗明细 ---"] + all_logs

    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Step 02 日志 ({datetime.now()})\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(all_logs))

    print(f"  day=0 极端BMI删除: {len(drop_logs)} 条")
    print(f"STEP 2 完成！输出 -> {OUT_CSV}\n")


if __name__ == '__main__':
    main()
