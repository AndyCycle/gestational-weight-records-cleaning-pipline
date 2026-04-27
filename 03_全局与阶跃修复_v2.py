import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline V2 Step 3] 全局斤系与长程阶跃修复（含分娩记录外部验证）===")

INPUT_CSV     = r"02_初步清洗_去低级失误版.csv"
DELIVERY_XLSX = r"..\宝安妇幼数据搜索\清洗任务\Baoan合并校验\Baoan分娩记录-第1-5批-清洗地址后-icd11_mapped-20260410.xlsx"
OUT_DIR       = r"."
OUT_CSV       = os.path.join(OUT_DIR, "03_全局与阶跃修复版.csv")
LOG_FILE      = os.path.join(OUT_DIR, "03_全局与阶跃修复_日志.txt")
PLOT_DIR      = os.path.join(OUT_DIR, "03_Plots_全局与阶跃")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================== 分娩记录关键词 ==============================
OBESITY_KW  = ['肥胖', '胖', '超重']
LEAN_KW     = ['消瘦', '瘦']
BMI28_KW    = ['BMI≥28', 'BMI>=28', 'BMI＞28']
BMI2518_KW  = ['BMI＞25或＜18.5', 'BMI>25或<18.5', 'BMI＞25', 'BMI>25']

# 突变点检测参数（实际测试中可调整）
SPIKE_RATIO    = 1.5   # 相邻体重之比 >= 此值视为突变
SPIKE_ABS_DIFF = 20.0  # 且绝对差 >= 此值（kg）

# BMI 阈值
HIGH_BMI_SUSPECT = 25.0
LOW_BMI_SUSPECT  = 13.0
BMI_EXTREME_HIGH = 37.0
BMI_EXTREME_LOW  = 12.5

# ============================== 工具函数 ==============================

def load_delivery(xlsx_path):
    """
    加载分娩记录，返回以项目流水号为 key 的字典，value 为行数据 dict。
    注：若文件不存在返回空字典。
    """
    if not os.path.exists(xlsx_path):
        print(f"  警告：分娩记录文件不存在: {xlsx_path}，将跳过外部验证。")
        return {}
    df = pd.read_excel(xlsx_path, dtype={'项目流水号': str})
    df['项目流水号'] = df['项目流水号'].astype(str).str.strip()
    result = {}
    for _, row in df.iterrows():
        nid = row['项目流水号']
        if nid not in result:
            result[nid] = row.to_dict()
    print(f"  分娩记录加载完成，共 {len(result)} 条（按项目流水号去重）")
    return result


def _contains_any(text, keywords):
    if pd.isna(text):
        return False
    text = str(text)
    return any(kw in text for kw in keywords)


def query_delivery_labels(nid, delivery_dict):
    """
    以项目流水号查询分娩记录。
    返回:
        found          : bool，是否在分娩记录中找到该个体
        obesity_label  : bool，是否有肥胖/超重/胖 标注
        lean_label     : bool，是否有消瘦/瘦 标注
        bmi_risk_25_18 : bool，孕期风险项是否含"BMI>25或<18.5"模糊字段
        bmi28_confirmed: bool，孕期风险项是否含 BMI>=28 精确字段
        raw_info       : str，原始字段拼接（用于日志）
    """
    if nid not in delivery_dict:
        return False, False, False, False, False, ""

    row = delivery_dict[nid]
    risk_field  = row.get('孕期风险项', '')
    surgery     = row.get('手术适应症', '')
    obstetric   = row.get('产科合并症', '')

    bmi28_confirmed = _contains_any(risk_field, BMI28_KW)
    bmi_risk_25_18  = _contains_any(risk_field, BMI2518_KW)

    obesity_label = (
        bmi28_confirmed or
        _contains_any(surgery, OBESITY_KW) or
        _contains_any(obstetric, OBESITY_KW)
    )
    lean_label = (
        _contains_any(surgery, LEAN_KW) or
        _contains_any(obstetric, LEAN_KW) or
        (bmi_risk_25_18 and not obesity_label)   # 模糊字段下消瘦端
    )

    raw_info = f"风险项=[{risk_field}] 手术适应症=[{surgery}] 产科合并症=[{obstetric}]"
    return True, obesity_label, lean_label, bmi_risk_25_18, bmi28_confirmed, raw_info


def has_spike(v_w, spike_ratio=SPIKE_RATIO, min_abs_diff=SPIKE_ABS_DIFF):
    """
    检测时序中是否存在突变点：
    相邻体重比 >= spike_ratio 且绝对差 >= min_abs_diff。
    """
    for i in range(1, len(v_w)):
        a, b = v_w[i - 1], v_w[i]
        if a <= 0 or b <= 0:
            continue
        ratio = max(a, b) / min(a, b)
        if ratio >= spike_ratio and abs(a - b) >= min_abs_diff:
            return True
    return False


def _spike_pairs(v_w, spike_ratio=SPIKE_RATIO, min_abs_diff=SPIKE_ABS_DIFF):
    """返回所有满足突变条件的相邻对索引集合（右侧索引）。"""
    result = set()
    for i in range(1, len(v_w)):
        a, b = v_w[i - 1], v_w[i]
        if a <= 0 or b <= 0:
            continue
        if max(a, b) / min(a, b) >= spike_ratio and abs(a - b) >= min_abs_diff:
            result.add(i)
    return result


def classify_spike_pattern(v_w, v_days):
    """
    对有效体重序列的突变模式进行精细分类，返回 (pattern, meta)。

    pattern 取值：
      'tail_drop'            末尾1-2点下降跳变，最后点 <= 倒数第二点
                             → 更倾向于真实孕晚期/产后数据，保留原值
      'tail_isolated_high'   末尾单个点异常高（上跳），其余轨迹无下降趋势
                             → 单点录入错误，留给后续局部尖峰步骤处理
      'day0_isolated_high'   仅第一对跳变，day=0 约为后续中值的 2 倍，
                             后续序列内部一致 → 仅对 day=0 /2
      'two_point_front_high' 仅两点，后点 ≈ 前点 / 2 → 仅对前点 /2
      'mid_isolated_low'     中段（非首尾）有且仅有一个孤立低点 ≈ 其余点中值/2，
                             其余点内部一致 → 保留低点，其余 /2
      'global'               其余有效突变 → 全局 /2
    """
    n = len(v_w)
    meta = {}

    # ---- 两点序列 ----
    if n == 2:
        a, b = v_w[0], v_w[1]
        if a > 0 and b > 0:
            ratio = max(a, b) / min(a, b)
            if 1.6 <= ratio <= 2.8 and abs(a - b) >= SPIKE_ABS_DIFF and a > b:
                return 'two_point_front_high', meta
        return 'global', meta

    # ---- 三点及以上 ----
    spikes = _spike_pairs(v_w)

    # tail_drop / tail_isolated_high：突变仅在最后一对
    if spikes == {n - 1}:
        if v_w[-1] <= v_w[-2]:
            # 最后一点低于倒数第二点 → 末尾下降，保留
            return 'tail_drop', meta
        else:
            # 最后一点高于倒数第二点 → 末尾孤立上突变
            # 进一步确认：前 n-1 个点之间无突变（避免误判）
            inner_spikes = _spike_pairs(v_w[:-1])
            if not inner_spikes:
                return 'tail_isolated_high', meta
            # 如果前段也有突变，退化为 global
            return 'global', meta

    # day0_isolated_high：突变仅在第一对，且 v_w[0] > v_w[1]
    if spikes == {1} and v_w[0] > v_w[1]:
        ratio = v_w[0] / v_w[1]
        if 1.6 <= ratio <= 2.8:
            # 后续部分内部是否稳定（无其他突变）
            inner_spikes = _spike_pairs(v_w[1:])
            if not inner_spikes:
                return 'day0_isolated_high', meta

    # mid_isolated_low：中段有且仅有一个孤立低点
    # 定义：位于序列内部（非首尾），周围点与之比值满足突变条件，
    #       且去掉该点后其余序列内部无突变
    if n >= 4:
        for j in range(1, n - 1):          # 只看内部点
            left_spike  = (j in spikes)    # j 是右侧突变点
            right_spike = (j + 1 in spikes)  # j 也是左侧（j+1 的左邻）
            if left_spike and right_spike and v_w[j] < v_w[j - 1] and v_w[j] < v_w[j + 1]:
                # 候选：j 处是低谷
                without_j = np.concatenate([v_w[:j], v_w[j + 1:]])
                med_rest  = float(np.median(without_j))
                ratio_j   = med_rest / v_w[j] if v_w[j] > 0 else 0
                if 1.6 <= ratio_j <= 2.8:
                    # 去掉低点后，其余序列是否稳定
                    inner_spikes = _spike_pairs(without_j)
                    if not inner_spikes:
                        meta['low_j'] = j
                        return 'mid_isolated_low', meta

    return 'global', meta


def apply_spike_fix(pattern, meta, w_orig, v_idx, v_w, days, logs):
    """
    根据 classify_spike_pattern 的结果执行精准修正。
    返回 (w_orig, logs, changed, error_type)
    """
    if pattern == 'tail_drop':
        logs.append('  → 末尾单侧下降（非单位错误）→ 保留原值，不做/2')
        return w_orig, logs, False, ''

    elif pattern == 'tail_isolated_high':
        logs.append('  → 末尾孤立上突变（单点录入偏高）→ 不做全局/2，留给后续局部尖峰步骤')
        return w_orig, logs, False, ''

    elif pattern == 'day0_isolated_high':
        i0  = v_idx[0]
        old = w_orig[i0]
        w_orig[i0] = round(old / 2.0, 2)
        logs.append(f'  → day=0 孤立高点 /2 | Day {days[i0]}d: {old:.1f} -> {w_orig[i0]:.1f}')
        return w_orig, logs, True, 'Day0_Isolated_High'

    elif pattern == 'two_point_front_high':
        i0  = v_idx[0]
        old = w_orig[i0]
        w_orig[i0] = round(old / 2.0, 2)
        logs.append(f'  → 两点序列前高后低，仅/2前点 | Day {days[i0]}d: {old:.1f} -> {w_orig[i0]:.1f}')
        return w_orig, logs, True, 'TwoPoint_Front_Div2'

    elif pattern == 'mid_isolated_low':
        low_j   = meta['low_j']
        changed = False
        for ii, curr_i in enumerate(v_idx):
            if ii == low_j:
                logs.append(f'  Day {days[curr_i]}d: 保留孤立低点（真实值）{v_w[ii]:.1f}')
                continue
            old = w_orig[curr_i]
            w_orig[curr_i] = round(old / 2.0, 2)
            logs.append(f'  Day {days[curr_i]}d: /2 | {old:.1f} -> {w_orig[curr_i]:.1f}')
            changed = True
        return w_orig, logs, changed, 'Mid_Isolated_Low_Kept'

    else:  # 'global'
        return _apply_global_div2(w_orig, v_idx, v_w, days, logs, 'Global_Div2')


def get_height(group):
    """返回身高（m），优先用群体 height 列众数。"""
    h_series = group['height'].dropna() if 'height' in group.columns else pd.Series(dtype=float)
    if len(h_series) == 0:
        return None
    h_val = float(h_series.mode().iloc[0])
    if h_val > 100:
        return h_val / 100.0
    elif 1.0 < h_val < 3.0:
        return h_val
    return None


def plot_repair(nid, days, w_raw, w_clean, error_type, logs):
    plt.figure(figsize=(10, 6))
    plt.plot(days, w_raw, color='lightgray', linestyle='--', marker='o', label='原始 (Raw)')
    plt.plot(days, w_clean, color='blue', linestyle='-', marker='D', alpha=0.7, label='修复后 (Cleaned)')
    changed_days, changed_cleans = [], []
    for d, r, c in zip(days, w_raw, w_clean):
        if not pd.isna(r) and not pd.isna(c) and abs(r - c) > 0.01:
            changed_days.append(d)
            changed_cleans.append(c)
    if changed_days:
        plt.scatter(changed_days, changed_cleans, color='red', s=100, zorder=5, label='修复点')
    plt.title(f"ID: {nid} | {error_type}")
    plt.xlabel("孕周(天)")
    plt.ylabel("体重 (kg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    info_text = "\n".join(logs[:8])
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{nid}_{error_type}.png"), dpi=100)
    plt.close()

# ============================== 核心清洗逻辑 ==============================

def clean_global_and_step_v2(group, nid, delivery_dict):
    """
    v2 全局斤系与阶跃修复，引入分娩记录外部验证。
    返回 (cleaned_group, logs, changed, error_type, flag)
    flag: 标注字段（inconclusive / None 等）
    """
    w_orig = group['weight'].values.copy()
    days   = group['gestation_day'].values

    valid_mask = ~pd.isna(w_orig)
    if valid_mask.sum() < 2:
        group['weight_cleaned'] = w_orig
        return group, [], False, "", None

    v_idx = np.where(valid_mask)[0]
    v_w   = w_orig[valid_mask]

    logs       = []
    changed    = False
    error_type = ""
    flag       = None

    H = get_height(group)

    median_bmi = None
    if H and H > 0:
        median_bmi = round(float(np.median(v_w)) / (H * H), 2)

    # ---------- 高 BMI 可疑路径 ----------
    if median_bmi is not None and median_bmi > HIGH_BMI_SUSPECT:
        found, obesity_label, lean_label, bmi_risk_25_18, bmi28_confirmed, raw_info = \
            query_delivery_labels(nid, delivery_dict)
        logs.append(
            f"高BMI可疑(中位BMI={median_bmi:.1f}) | 分娩记录found={found} "
            f"obesity={obesity_label} bmi28={bmi28_confirmed} bmi2518={bmi_risk_25_18}"
        )

        if not found:
            # 无分娩记录
            if has_spike(v_w):
                pattern, pmeta = classify_spike_pattern(v_w, days)
                logs.append(f"  → 无分娩记录但存在突变点，模式={pattern}")
                w_orig, logs, changed, error_type = apply_spike_fix(
                    pattern, pmeta, w_orig, v_idx, v_w, days, logs
                )
                if changed:
                    error_type = f"无分娩记录_{error_type}"
            else:
                logs.append("  → 无分娩记录且无突变点 → 保留原值")

        elif obesity_label:
            # 有分娩记录且确认肥胖
            if median_bmi >= BMI_EXTREME_HIGH:
                logs.append(f"  → 肥胖确认 (BMI>={BMI_EXTREME_HIGH}) → 跳过/2")
            else:
                logs.append("  → 肥胖确认 (obesity_label=True) → 跳过/2")

        elif bmi_risk_25_18 and not obesity_label:
            # 有模糊字段"BMI>25或<18.5"
            if median_bmi >= BMI_EXTREME_HIGH:
                logs.append(f"  → BMI>={BMI_EXTREME_HIGH} + 模糊字段 → 认定肥胖，跳过/2")
            elif median_bmi > HIGH_BMI_SUSPECT:
                # BMI ∈ (25, 37)，依赖突变检测
                if has_spike(v_w):
                    pattern, pmeta = classify_spike_pattern(v_w, days)
                    logs.append(f"  → 模糊字段 BMI(25,37) + 突变点，模式={pattern}")
                    w_orig, logs, changed, error_type = apply_spike_fix(
                        pattern, pmeta, w_orig, v_idx, v_w, days, logs
                    )
                    if changed:
                        error_type = f"模糊字段_{error_type}"
                    elif not changed and pattern in ('tail_drop', 'tail_isolated_high'):
                        # 突变模式确认为非单位错误，仍标记 inconclusive 供人工审查
                        flag = 'BMI_25_37_inconclusive'
                else:
                    logs.append("  → 模糊字段 BMI(25,37) + 无突变 → flag=BMI_25_37_inconclusive")
                    flag = 'BMI_25_37_inconclusive'

        else:
            # 分娩记录存在但无相关标注
            if has_spike(v_w):
                pattern, pmeta = classify_spike_pattern(v_w, days)
                logs.append(f"  → 分娩记录无相关标注 + 突变点，模式={pattern}")
                w_orig, logs, changed, error_type = apply_spike_fix(
                    pattern, pmeta, w_orig, v_idx, v_w, days, logs
                )
                if changed:
                    error_type = f"无标注_{error_type}"
                elif not changed and pattern in ('tail_drop', 'tail_isolated_high'):
                    flag = 'high_BMI_no_delivery_label'
            else:
                logs.append("  → 分娩记录无相关标注 + 无突变 → flag=high_BMI_no_delivery_label")
                flag = 'high_BMI_no_delivery_label'

        group['weight_cleaned'] = w_orig
        return group, logs, changed, error_type, flag

    # ---------- 低 BMI 可疑路径 ----------
    if median_bmi is not None and median_bmi < LOW_BMI_SUSPECT:
        found, obesity_label, lean_label, bmi_risk_25_18, bmi28_confirmed, raw_info = \
            query_delivery_labels(nid, delivery_dict)
        logs.append(
            f"低BMI可疑(中位BMI={median_bmi:.1f}) | 分娩记录found={found} "
            f"lean={lean_label}"
        )

        drop_idx_list = []

        if found:
            if lean_label:
                # 确认消瘦
                if median_bmi < BMI_EXTREME_LOW:
                    logs.append(
                        f"  → 消瘦确认但BMI<{BMI_EXTREME_LOW}（极端）→ 删除异常时序点"
                    )
                    drop_idx_list = list(v_idx)  # 删除所有极端低 BMI 点
                else:
                    logs.append(f"  → 消瘦确认 BMI in [{BMI_EXTREME_LOW},{LOW_BMI_SUSPECT}) → flag=lean_confirmed_borderline")
                    flag = 'lean_confirmed_borderline'
            else:
                logs.append("  → 分娩记录无消瘦标注 → 疑似记录错误 → 删除异常时序点")
                drop_idx_list = list(v_idx)
        else:
            if median_bmi < BMI_EXTREME_LOW:
                logs.append(f"  → 无分娩记录且BMI<{BMI_EXTREME_LOW} → 删除异常时序点")
                drop_idx_list = list(v_idx)
            else:
                logs.append(f"  → 无分娩记录且BMI in [{BMI_EXTREME_LOW},{LOW_BMI_SUSPECT}) → flag=very_low_BMI_no_delivery_record")
                flag = 'very_low_BMI_no_delivery_record'

        if drop_idx_list:
            for di in drop_idx_list:
                if not pd.isna(w_orig[di]):
                    logs.append(f"  删除 Day {days[di]}d w={w_orig[di]:.1f}kg (BMI可疑低)")
                    w_orig[di] = np.nan
            changed    = True
            error_type = "Low_BMI_Suspect_Drop"

        group['weight_cleaned'] = w_orig
        return group, logs, changed, error_type, flag

    # ---------- 原有全局斤系逻辑（未触发高/低 BMI 路径时执行）----------
    n_v = len(v_w)

    # 1. 全局斤系（最低 >= 80，中位数 >= 105）
    if np.min(v_w) >= 80 and np.median(v_w) >= 105:
        safe_to_convert = True
        if H:
            sim_min_bmi = (np.min(v_w) / 2.0) / (H * H)
            if sim_min_bmi < 13.5:
                safe_to_convert = False
        if safe_to_convert:
            for curr_i in v_idx:
                original = w_orig[curr_i]
                w_orig[curr_i] = round(original / 2.0, 2)
                logs.append(f"Day {days[curr_i]}d: 全局斤 | {original:.1f} -> {w_orig[curr_i]:.1f}")
            changed    = True
            error_type = "Global_Jin_全局斤系"
            group['weight_cleaned'] = w_orig
            return group, logs, changed, error_type, flag

    # 2. 长程阶跃
    if n_v >= 3:
        best_split = -1
        best_score = float('inf')
        for i in range(1, n_v):
            part1 = v_w[:i]
            part2 = v_w[i:]
            if not len(part1) or not len(part2):
                continue
            med1, med2 = np.median(part1), np.median(part2)
            if abs(med1 - med2) > 30:
                ratio = max(med1, med2) / min(med1, med2)
                if 1.6 <= ratio <= 2.8:
                    std1 = np.std(part1) if len(part1) > 1 else 0
                    std2 = np.std(part2) if len(part2) > 1 else 0
                    if std1 < 18 and std2 < 22:
                        safe_to_convert = True
                        if med1 > med2:
                            if H and (np.min(part1) / 2.0) / (H * H) < 13.5:
                                safe_to_convert = False
                        else:
                            if H and (np.min(part2) / 2.0) / (H * H) < 13.5:
                                safe_to_convert = False
                        if safe_to_convert:
                            score = (std1 * len(part1) + std2 * len(part2)) / n_v
                            if score < best_score:
                                best_score = score
                                best_split = i

        if best_split != -1:
            part1 = v_w[:best_split]
            part2 = v_w[best_split:]
            med1, med2 = np.median(part1), np.median(part2)
            if med1 > med2:
                corrected_med = med1 / 2.0
                plausible = abs(corrected_med - med2) < 15
            else:
                corrected_med = med2 / 2.0
                plausible = abs(corrected_med - med1) < 15
            if plausible:
                if med1 > med2:
                    error_type = "Step_FrontJin_前程斤系"
                    for i_v in range(best_split):
                        curr_i = v_idx[i_v]
                        old = w_orig[curr_i]
                        w_orig[curr_i] = round(old / 2.0, 2)
                        logs.append(f"Day {days[curr_i]}d: 前程阶跃 | {old:.1f} -> {w_orig[curr_i]:.1f}")
                    changed = True
                else:
                    error_type = "Step_BackJin_后程斤系"
                    for i_v in range(best_split, n_v):
                        curr_i = v_idx[i_v]
                        old = w_orig[curr_i]
                        w_orig[curr_i] = round(old / 2.0, 2)
                        logs.append(f"Day {days[curr_i]}d: 后程阶跃 | {old:.1f} -> {w_orig[curr_i]:.1f}")
                    changed = True
                if changed:
                    group['weight_cleaned'] = w_orig
                    return group, logs, changed, error_type, flag

    group['weight_cleaned'] = w_orig
    return group, [], False, "", None


def _apply_global_div2(w_orig, v_idx, v_w, days, logs, error_type):
    """对所有有效体重执行全局 /2 修正的辅助函数。"""
    for ii, curr_i in enumerate(v_idx):
        original = w_orig[curr_i]
        w_orig[curr_i] = round(original / 2.0, 2)
        logs.append(f"  Day {days[curr_i]}d: /2 | {original:.1f} -> {w_orig[curr_i]:.1f}")
    return w_orig, logs, True, error_type


# ============================== 主函数 ==============================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到输入文件: {INPUT_CSV}")
        return

    print(f"正在加载: {INPUT_CSV}")
    df     = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    if id_col not in df.columns:
        df.rename(columns={df.columns[0]: id_col}, inplace=True)
    df[id_col] = df[id_col].astype(str).str.strip()

    print(f"正在加载分娩记录...")
    delivery_dict = load_delivery(DELIVERY_XLSX)

    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    frames, all_logs = [], []
    changed_count = 0
    total = len(grouped)
    flag_counts = {}

    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"  03 处理进度: {i}/{total}...")
        c_group, logs, changed, err_type, flag = clean_global_and_step_v2(
            group.copy(), nid, delivery_dict
        )
        frames.append(c_group)
        if changed:
            changed_count += 1
            all_logs.append(f"[{nid}] 修复类型: {err_type}")
            all_logs.extend(["  " + l for l in logs])
            plot_repair(
                nid,
                c_group['gestation_day'].values,
                c_group['weight_raw_p1'].values if 'weight_raw_p1' in c_group.columns
                else c_group['weight'].values,
                c_group['weight_cleaned'].values,
                err_type, logs
            )
        elif logs:
            all_logs.append(f"[{nid}] 无修改 | " + " | ".join(logs[:2]))
        if flag:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
            all_logs.append(f"  flag={flag}")

    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(
        columns={'weight': 'weight_raw_p3', 'weight_cleaned': 'weight'},
        inplace=True
    )
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Step 03 日志 ({datetime.now()})\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(all_logs))

    print(f"\nSTEP 3 完成！修复: {changed_count} 例")
    print("  待人工核查 flag 统计:")
    for k, v in flag_counts.items():
        print(f"    {k}: {v}")
    print(f"  输出 -> {OUT_CSV}\n")


if __name__ == '__main__':
    main()
