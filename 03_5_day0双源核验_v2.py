import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=== [Pipeline V2 Step 3.5] Day=0 双源一致性核验 ===")

# ============================== 路径配置 ==============================
# 输入：Step 03 的输出
INPUT_CSV  = r"03_全局与阶跃修复版.csv"

# 初检原始文件（与 Step 01 相同，用于取候选列的原始值）
INIT_FILES = [
    r"..\初检\孕妇初检-第五批-to liu-20260319.xlsx",
    r"..\初检\孕期初检-第1-4批-to liu-20260319.xlsx",
]

OUT_DIR  = r"."
OUT_CSV  = os.path.join(OUT_DIR, "03_5_day0双源核验版.csv")
LOG_FILE = os.path.join(OUT_DIR, "03_5_day0双源核验_日志.txt")

# ============================== 参数 ==============================
EARLY_DAY_MAX = 98    # 孕早期上限（天），用于寻找参照点
TOLERANCE     = 0.5   # 认定"相等"的体重差阈值（kg）
BMI_MIN       = 14.0  # 替代体重的生理下限
BMI_MAX       = 50.0  # 替代体重的生理上限

# ============================== 工具函数 ==============================

def calc_bmi(w, h_cm):
    """返回 BMI；任一值无效则返回 None。"""
    if w is None or h_cm is None:
        return None
    try:
        w, h_cm = float(w), float(h_cm)
    except (ValueError, TypeError):
        return None
    if pd.isna(w) or pd.isna(h_cm) or h_cm <= 0:
        return None
    h_m = h_cm / 100.0
    return round(w / (h_m * h_m), 2)


def safe_float(v):
    """将值转为 float，无效则返回 NaN。"""
    try:
        f = float(v)
        return f if not np.isnan(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


# ============================== 加载初检数据 ==============================
print("正在加载初检数据（取双候选列）...")
init_frames = []
for f in INIT_FILES:
    if os.path.exists(f):
        init_frames.append(pd.read_excel(f))
    else:
        print(f"  警告：初检文件不存在 {f}")

if not init_frames:
    print("错误：所有初检文件均不存在，退出。")
    exit(1)

init_df = pd.concat(init_frames, ignore_index=True)
init_df['项目流水号'] = init_df['项目流水号'].astype(str).str.strip()
init_df = init_df.drop_duplicates('项目流水号')
print(f"  初检记录共 {len(init_df)} 条（去重后）")

# 构建 lookup：项目流水号 -> {孕前体重, 体重}
init_lookup: dict = {}
for _, row in init_df.iterrows():
    nid = str(row['项目流水号']).strip()
    init_lookup[nid] = {
        'pre_weight': safe_float(row.get('孕前体重', np.nan)),
        'weight_col': safe_float(row.get('体重', np.nan)),
    }

# ============================== 加载主数据 ==============================
print(f"正在加载: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV, low_memory=False)
df['项目流水号'] = df['项目流水号'].astype(str).str.strip()
print(f"  共 {len(df)} 行，{df['项目流水号'].nunique()} 个样本")

# ============================== 逐样本处理 ==============================
logs_all      = []
replaced_count   = 0   # 成功替换 day=0 体重
dup_no_alt_count = 0   # 发现相等但无可用替代
skip_count       = 0   # 无 day=0 或无早期参照

grouped = df.sort_values(['项目流水号', 'gestation_day']).groupby('项目流水号', observed=True)
frames  = []

total = df['项目流水号'].nunique()
for i, (nid, group) in enumerate(grouped):
    if i % 20000 == 0:
        print(f"  3.5 处理进度: {i}/{total}...")

    group = group.copy().reset_index(drop=True)
    nid   = str(nid)

    # ---- 定位 day=0 的 Initial_Raw 行 ----
    day0_mask = (group['gestation_day'] == 0) & (group['type'] == 'Initial_Raw')
    if not day0_mask.any():
        frames.append(group)
        skip_count += 1
        continue

    day0_idx    = group[day0_mask].index[0]
    day0_weight = safe_float(group.at[day0_idx, 'weight'])
    day0_source = str(group.at[day0_idx, 'weight_source']) if 'weight_source' in group.columns else ''
    height_cm   = safe_float(group.at[day0_idx, 'height'])

    if np.isnan(day0_weight):
        frames.append(group)
        skip_count += 1
        continue

    # ---- 寻找孕早期最近一次非 day=0 体重 ----
    early_mask = (group['gestation_day'] > 0) & (group['gestation_day'] <= EARLY_DAY_MAX)
    early_rows = group[early_mask].sort_values('gestation_day')
    if early_rows.empty:
        frames.append(group)
        skip_count += 1
        continue

    nearest_row    = early_rows.iloc[0]
    nearest_weight = safe_float(nearest_row['weight'])
    nearest_day    = int(nearest_row['gestation_day'])

    if np.isnan(nearest_weight):
        frames.append(group)
        skip_count += 1
        continue

    diff = abs(day0_weight - nearest_weight)

    # ---- 若差值 < 阈值，认为二者来源相同，尝试替换 ----
    if diff >= TOLERANCE:
        frames.append(group)
        continue

    # 从初检 lookup 取候选值
    alt_pool = init_lookup.get(nid, {})

    if day0_source == 'weight_col':
        # 当前用了建册体重，尝试切换为孕前体重
        candidate_val = alt_pool.get('pre_weight', np.nan)
        candidate_src = 'pre_weight'
    else:
        # 当前用了孕前体重（或其他），尝试切换为建册体重
        candidate_val = alt_pool.get('weight_col', np.nan)
        candidate_src = 'weight_col'

    # 验证候选值
    new_weight = None
    reject_reason = ''
    if np.isnan(candidate_val):
        reject_reason = '候选值缺失(NaN)'
    elif abs(candidate_val - nearest_weight) < TOLERANCE:
        reject_reason = f'候选值({candidate_val:.1f})与早期参照({nearest_weight:.1f})也相等，无法区分'
    else:
        bmi_cand = calc_bmi(candidate_val, height_cm)
        if bmi_cand is None:
            reject_reason = '身高缺失，无法验算BMI'
        elif bmi_cand < BMI_MIN or bmi_cand > BMI_MAX:
            reject_reason = f'候选值BMI={bmi_cand:.1f} 超出生理范围[{BMI_MIN},{BMI_MAX}]'
        else:
            new_weight = candidate_val

    if new_weight is not None:
        old_weight = day0_weight
        group.at[day0_idx, 'weight'] = round(float(new_weight), 2)
        if 'weight_source' in group.columns:
            group.at[day0_idx, 'weight_source'] = candidate_src
        new_bmi = calc_bmi(new_weight, height_cm)
        if new_bmi is not None and 'BMI' in group.columns:
            group.at[day0_idx, 'BMI'] = new_bmi

        replaced_count += 1
        logs_all.append(
            f"[{nid}] 替换 day=0: {old_weight:.1f}→{new_weight:.1f}kg "
            f"({day0_source}→{candidate_src}) | 早期参照=day{nearest_day}d w={nearest_weight:.1f}kg"
        )
    else:
        dup_no_alt_count += 1
        logs_all.append(
            f"[{nid}] day=0({day0_weight:.1f}kg)≈早期参照 day{nearest_day}d({nearest_weight:.1f}kg) "
            f"| 无可用替代: {reject_reason}"
        )

    frames.append(group)

# ============================== 输出 ==============================
out_df = pd.concat(frames, ignore_index=True)
out_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

log_content = (
    f"Step 3.5 日志 ({datetime.now()})\n"
    + "=" * 60 + "\n"
    + "\n".join(logs_all)
)
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(log_content)

print(f"\nSTEP 3.5 完成！")
print(f"  day=0 成功替换: {replaced_count} 例")
print(f"  发现相等但无可用替代: {dup_no_alt_count} 例")
print(f"  无 day=0 或无早期参照: {skip_count} 例")
print(f"  输出 -> {OUT_CSV}")
print(f"  日志 -> {LOG_FILE}")

if __name__ == '__main__':
    pass
