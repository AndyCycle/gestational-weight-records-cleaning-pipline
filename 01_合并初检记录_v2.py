import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=== [Pipeline V2 Step 1] 合并初检记录（四路决策树 + 身高统一化）===")

# ============================== 配置路径 ==============================
# 初检文件列表（放在宝安数据根目录，与本脚本的工作目录一致）
INIT_FILES = [
    r"初检\孕妇初检-第五批-to liu-20260319.xlsx",
    r"初检\孕期初检-第1-4批-to liu-20260319.xlsx",
]
HIS_PATH   = r"HIS系统\宝安_HIS前处理_合并表.csv"

OUT_DIR  = r"HIS系统\清洗流程_v2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "01_合并后底表_带初检.csv")
LOG_PATH = os.path.join(OUT_DIR, "01_合并初检_日志.txt")

# ============================== 参数 ==============================
BMI_VALID_LOW    = 14.0   # 低于此值视为体重/BMI 可疑
EARLY_PREG_DAYS  = 98     # 孕早期上限（天），用于仲裁时寻找 HIS 参照点
BMI_DIFF_THRESH  = 1.0    # |BMI0 - BMI_t| 超过此值触发仲裁

# ============================== 工具函数 ==============================

def get_unified_height(his_group):
    """
    从 HIS 时序中获取统一身高（cm）。
    取众数；若众数为 NaN，返回 None。
    支持 HIS 中身高以 cm（>3）或 m（1~3）两种记录形式。
    """
    if 'height' not in his_group.columns:
        return None
    h_vals = his_group['height'].dropna().values
    if len(h_vals) == 0:
        return None
    # 统一到 cm
    normalized = []
    for h in h_vals:
        if 1.0 < h < 3.0:
            normalized.append(round(h * 100, 1))
        elif h > 100:
            normalized.append(round(h, 1))
        # 其余忽略（异常值）
    if not normalized:
        return None
    # 取众数
    from scipy.stats import mode as scipy_mode
    try:
        result = scipy_mode(normalized, keepdims=True)
        return float(result.mode[0])
    except Exception:
        return float(pd.Series(normalized).mode().iloc[0])


def height_from_init(row):
    """从初检行读取身高并统一到 cm。"""
    h = row.get('身高', np.nan)
    if pd.isna(h):
        return None
    h = float(h)
    if 1.0 < h < 3.0:
        return round(h * 100, 1)
    elif h > 100:
        return h
    return None


def calc_bmi(weight_kg, height_cm):
    """返回 BMI；任一值无效则返回 None。"""
    if weight_kg is None or height_cm is None:
        return None
    h_m = height_cm / 100.0
    if h_m <= 0:
        return None
    return round(weight_kg / (h_m * h_m), 2)


def get_nearest_early_weight(his_group, max_day=EARLY_PREG_DAYS):
    """
    取 HIS 时序中 gestation_day < max_day 内按天数升序的第一条有效 weight。
    返回 (day, weight) 或 (None, None)。
    """
    # 防御：空 DataFrame（该个体在 HIS 中无记录）或缺少必要列时直接返回
    if his_group.empty or 'gestation_day' not in his_group.columns or 'weight' not in his_group.columns:
        return None, None
    early = his_group[
        (his_group['gestation_day'] > 0) &
        (his_group['gestation_day'] < max_day) &
        (his_group['weight'].notna())
    ].sort_values('gestation_day')
    if early.empty:
        return None, None
    row = early.iloc[0]
    return int(row['gestation_day']), float(row['weight'])


def decide_day0_weight(pre_w, enroll_w, H_cm, his_group, nid, logs):
    """
    四路决策树核心：
    返回 (chosen_weight, weight_source, H_cm)
    weight_source in {'pre_weight', 'weight_col', 'weight_col_enrollment',
                      'his_early', None}
    """
    # 路径 A：孕前体重存在
    if pre_w is not None and not pd.isna(pre_w):
        try:
            pre_w = float(pre_w)
        except (ValueError, TypeError):
            pre_w = None

    if pre_w is not None:
        bmi0 = calc_bmi(pre_w, H_cm)

        # A1: BMI0 < 14，孕前体重可疑
        if bmi0 is not None and bmi0 < BMI_VALID_LOW:
            logs.append(f"[{nid}] A1: 孕前体重 BMI={bmi0:.1f}<{BMI_VALID_LOW}，转路径B")
            return _path_b(enroll_w, H_cm, his_group, nid, logs)

        # A2: BMI0 >= 14
        # 将 enroll_w 统一处理为 None（包含 NaN 情况），避免后续 NaN 比较失效
        if enroll_w is not None and not pd.isna(enroll_w):
            try:
                enroll_w = float(enroll_w)
            except (ValueError, TypeError):
                enroll_w = None
        else:
            enroll_w = None  # NaN/None 统一设为 None，使 A2-C 正确识别为"缺失"

        bmi_t = calc_bmi(enroll_w, H_cm) if enroll_w is not None else None

        # A2-C: 体重列缺失 or BMI 差 <= 阈值
        if bmi_t is None or (bmi0 is not None and abs(bmi0 - bmi_t) <= BMI_DIFF_THRESH):
            logs.append(f"[{nid}] A2-C: 采用孕前体重 {pre_w:.1f}kg (BMI={bmi0})")
            return pre_w, 'pre_weight', H_cm

        # A2-D: BMI 差 > 阈值，仲裁
        ref_day, ref_w = get_nearest_early_weight(his_group)
        if ref_w is not None:
            diff_pre    = abs(pre_w - ref_w)
            diff_enroll = abs(enroll_w - ref_w)
            if diff_pre <= diff_enroll:
                chosen, src = pre_w, 'pre_weight'
            else:
                chosen, src = enroll_w, 'weight_col'
            logs.append(
                f"[{nid}] A2-D: 仲裁 | 孕前体重={pre_w:.1f}(diff={diff_pre:.1f}), "
                f"建册体重={enroll_w:.1f}(diff={diff_enroll:.1f}), "
                f"参照day={ref_day}d w={ref_w:.1f} → 选{src}"
            )
            return chosen, src, H_cm
        else:
            # 无 HIS 早孕记录，退化为 A2-C 逻辑
            logs.append(f"[{nid}] A2-D: 无HIS早孕参照，退化采用孕前体重 {pre_w:.1f}kg")
            return pre_w, 'pre_weight', H_cm

    # 路径 B：孕前体重缺失
    logs.append(f"[{nid}] 路径B: 孕前体重缺失，尝试使用建册体重")
    return _path_b(enroll_w, H_cm, his_group, nid, logs)


def _path_b(enroll_w, H_cm, his_group, nid, logs):
    """路径 B 的实现（复用于 A1 → B 的跳转）。"""
    if enroll_w is not None and not pd.isna(enroll_w):
        try:
            enroll_w = float(enroll_w)
        except (ValueError, TypeError):
            enroll_w = None

    if enroll_w is not None:
        bmi_t = calc_bmi(enroll_w, H_cm)

        # B1: 建册体重 BMI 也可疑
        if bmi_t is not None and bmi_t < BMI_VALID_LOW:
            logs.append(
                f"[{nid}] B1: 建册体重 BMI={bmi_t:.1f}<{BMI_VALID_LOW}，"
                f"尝试HIS孕早期"
            )
            ref_day, ref_w = get_nearest_early_weight(his_group)
            if ref_w is not None:
                logs.append(
                    f"[{nid}] B1→HIS: 使用HIS day={ref_day}d 体重={ref_w:.1f}kg"
                )
                return ref_w, 'his_early', H_cm
            else:
                logs.append(f"[{nid}] B1: 无HIS早孕记录，不生成day=0")
                return None, None, H_cm

        # B2: 建册体重合理
        logs.append(
            f"[{nid}] B2: 使用建册体重 {enroll_w:.1f}kg "
            f"(BMI={bmi_t}, weight_source=weight_col_enrollment)"
        )
        return enroll_w, 'weight_col_enrollment', H_cm

    # 体重列也缺失
    logs.append(f"[{nid}] B: 建册体重也缺失，不生成day=0")
    return None, None, H_cm


# ============================== 主函数 ==============================

def main():
    print("正在加载初检数据...")
    init_frames = []
    for f in INIT_FILES:
        if os.path.exists(f):
            init_frames.append(pd.read_excel(f))
        else:
            print(f"  警告：初检文件不存在 {f}")
    if not init_frames:
        print("错误：所有初检文件均不存在，退出。")
        return
    init_df = pd.concat(init_frames, ignore_index=True)
    init_df['项目流水号'] = init_df['项目流水号'].astype(str).str.strip()
    init_df = init_df.drop_duplicates('项目流水号')
    print(f"  初检记录共 {len(init_df)} 条（去重后）")

    print(f"正在加载HIS时序表: {HIS_PATH}")
    sys_df = pd.read_csv(HIS_PATH, low_memory=False)
    sys_df['项目流水号'] = sys_df['项目流水号'].astype(str).str.strip()
    print(f"  HIS 记录共 {len(sys_df)} 条")

    # 按个体分组 HIS，便于后续查询
    his_grouped = {nid: grp for nid, grp in sys_df.groupby('项目流水号')}

    seeds_list = []
    logs_all   = []
    stat = {'pre_weight': 0, 'weight_col': 0, 'weight_col_enrollment': 0,
            'his_early': 0, 'none': 0}

    total = len(init_df)
    print(f"正在逐一处理 {total} 位孕妇的孕前体重决策...")

    for idx, row in init_df.iterrows():
        nid = row['项目流水号']
        logs = []

        # 从 HIS 获取统一身高
        his_grp = his_grouped.get(nid, pd.DataFrame())
        H_cm = get_unified_height(his_grp)  if not his_grp.empty else None
        # 若 HIS 无身高，取初检表身高
        if H_cm is None:
            H_cm = height_from_init(row)

        if H_cm is None:
            logs.append(f"[{nid}] 警告：身高缺失，无法计算BMI，跳过")
            logs_all.extend(logs)
            stat['none'] += 1
            continue

        pre_w    = row.get('孕前体重', np.nan)
        enroll_w = row.get('体重', np.nan)

        chosen_w, src, unified_H = decide_day0_weight(
            pre_w, enroll_w, H_cm, his_grp, nid, logs
        )
        logs_all.extend(logs)

        # NaN 与 None 均视为无效（NaN 在 A2-D 仲裁 bug 情况下可能漏出）
        if chosen_w is None or (isinstance(chosen_w, float) and pd.isna(chosen_w)):
            stat['none'] += 1
            continue

        stat[src] = stat.get(src, 0) + 1

        # 附加列（其余字段从初检行取，后续步骤会补全）
        seed_row = {
            '项目流水号':   nid,
            'weight':       round(float(chosen_w), 2),
            'height':       unified_H,
            'BMI':          calc_bmi(float(chosen_w), unified_H),
            'gestation_day': 0,
            'type':         'Initial_Raw',
            'weight_source': src,
        }
        seeds_list.append(seed_row)

    seeds_df = pd.DataFrame(seeds_list)
    print(f"\n  决策结果统计:")
    for k, v in stat.items():
        print(f"    {k}: {v}")
    print(f"  成功生成 day=0 锚点: {len(seeds_df)} 条")

    # 剔除 HIS 中已有 day=0 且该个体在 seeds_df 中的记录（防止双录）
    sys_df = sys_df[
        ~((sys_df['项目流水号'].isin(seeds_df['项目流水号'])) &
          (sys_df['gestation_day'] == 0))
    ]

    # 合并列对齐
    for col in sys_df.columns:
        if col not in seeds_df.columns:
            seeds_df[col] = np.nan

    # 诊断：打印 HIS 表中与体重相关的列名，用于排查列名不对齐问题
    weight_related = [c for c in sys_df.columns if 'weight' in c.lower() or '体重' in c]
    print(f"\n  [诊断] HIS 中 weight 相关列: {weight_related}")
    print(f"  [诊断] seeds_df 列: {list(seeds_df.columns)}")
    if 'weight' not in sys_df.columns:
        print("  [警告] HIS 表中不存在 'weight' 列！seeds_df 中的 weight 将在 concat 后独立存在，"
              "但 HIS 行对应 weight 为 NaN。请检查 HIS CSV 中体重列的实际列名。")

    final = pd.concat([sys_df, seeds_df], ignore_index=True)\
              .sort_values(['项目流水号', 'gestation_day'])

    final.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Step 01 日志 ({datetime.now()})\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(logs_all))

    print(f"\nSTEP 1 完成！输出 -> {OUT_PATH}")
    print(f"日志   -> {LOG_PATH}\n")


if __name__ == '__main__':
    main()
