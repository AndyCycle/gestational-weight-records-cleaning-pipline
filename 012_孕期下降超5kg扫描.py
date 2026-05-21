"""
=== [Pipeline Step 012] 孕期体重下降 >5kg 扫描 ===

在 10_孕前与早孕异动修复版.csv 上执行扫描：
1) 仅保留孕期点（默认 0~294 天）；
2) 若存在 is_postpartum_normal 列，则排除产后标记点；
3) 计算每个样本孕期体重序列的最大回撤（历史峰值 - 后续低点）；
4) 标记最大回撤 > 5kg 的样本；
5) 输出被标记样本清单与轨迹可视化。
"""

import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ─── 路径 ───────────────────────────────────────────────
INPUT_CSV = r"HIS系统\清洗流程_v2/10_孕前与早孕异动修复版_人工校正_20260507_112155.csv"
OUT_DIR = r"HIS系统\清洗流程_v2"
PLOT_DIR = os.path.join(OUT_DIR, "012_Plots_孕期下降超5kg")
REPORT_CSV = os.path.join(OUT_DIR, "012_孕期下降超5kg_样本清单.csv")
LOG_FILE = os.path.join(OUT_DIR, "012_孕期下降超5kg_扫描报告.txt")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── 参数 ───────────────────────────────────────────────
DROP_THRESHOLD_KG = 5.0
PREGNANCY_DAY_MIN = 0
PREGNANCY_DAY_MAX = 294
MIN_POINTS = 2
PROGRESS_EVERY = 20000


def safe_filename(text):
    s = str(text).strip()
    if not s:
        return "unknown_id"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def parse_bool_series(series):
    """将混合类型（bool/字符串/数字）统一为布尔序列。"""
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin({"true", "1", "yes", "y", "t"})


def find_max_drawdown(days, weights):
    """
    计算序列最大回撤（历史峰值->后续低点）与最大单步下降。
    返回：
      max_drop, peak_idx, trough_idx, max_single_step_drop
    """
    n = len(weights)
    if n < 2:
        return 0.0, 0, 0, 0.0

    peak_idx = 0
    peak_w = weights[0]
    best_drop = 0.0
    best_peak_idx = 0
    best_trough_idx = 0

    for i in range(1, n):
        w = weights[i]

        drop = peak_w - w
        if drop > best_drop:
            best_drop = float(drop)
            best_peak_idx = peak_idx
            best_trough_idx = i

        if w > peak_w:
            peak_w = w
            peak_idx = i

    step_drops = weights[:-1] - weights[1:]
    max_single_step_drop = float(np.max(step_drops)) if len(step_drops) > 0 else 0.0

    return best_drop, best_peak_idx, best_trough_idx, max_single_step_drop


def plot_sample(nid, days, weights, peak_idx, trough_idx, max_drop, max_step_drop, out_dir):
    """绘制单样本轨迹图，并高亮最大下降区间。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(
        days,
        weights,
        color="#4C78A8",
        linestyle="-",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="体重序列",
        zorder=2,
    )

    peak_day = days[peak_idx]
    peak_w = weights[peak_idx]
    trough_day = days[trough_idx]
    trough_w = weights[trough_idx]

    ax.scatter(
        [peak_day],
        [peak_w],
        color="#FF8C00",
        s=120,
        edgecolors="#8B4000",
        linewidths=1.2,
        zorder=5,
        label="下降起点(峰值)",
    )
    ax.scatter(
        [trough_day],
        [trough_w],
        color="red",
        s=120,
        edgecolors="darkred",
        linewidths=1.2,
        zorder=5,
        label="下降终点(低点)",
    )
    ax.plot(
        [peak_day, trough_day],
        [peak_w, trough_w],
        linestyle="--",
        linewidth=1.5,
        color="red",
        alpha=0.8,
        zorder=4,
    )

    ax.annotate(
        f"峰值\nDay {int(peak_day)}\n{peak_w:.1f}kg",
        (peak_day, peak_w),
        textcoords="offset points",
        xytext=(8, 12),
        fontsize=8,
        color="#8B4000",
        arrowprops=dict(arrowstyle="->", color="#8B4000", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFF2CC", alpha=0.9),
    )
    ax.annotate(
        f"低点\nDay {int(trough_day)}\n{trough_w:.1f}kg",
        (trough_day, trough_w),
        textcoords="offset points",
        xytext=(8, -46),
        fontsize=8,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFECEC", alpha=0.9),
    )

    ax.set_title(
        f"项目流水号: {nid} | 最大下降: {max_drop:.1f}kg | 最大单步下降: {max_step_drop:.1f}kg",
        fontsize=12,
    )
    ax.set_xlabel("孕龄 (天)")
    ax.set_ylabel("体重 (kg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    fname = safe_filename(nid) + "_drop_gt5kg.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=120)
    plt.close(fig)


def main():
    print("=== [Pipeline Step 012] 孕期体重下降 >5kg 扫描 ===")
    print(f"  输入文件: {INPUT_CSV}")
    print(f"  阈值: 最大回撤 > {DROP_THRESHOLD_KG} kg")
    print(f"  孕期范围: {PREGNANCY_DAY_MIN} ~ {PREGNANCY_DAY_MAX} 天")

    if not os.path.exists(INPUT_CSV):
        print(f"  错误: 找不到输入文件 {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    id_col = "项目流水号"
    if id_col not in df.columns:
        if "project_id" in df.columns:
            id_col = "project_id"
        else:
            id_col = df.columns[0]

    day_col = "gestation_day"
    wt_col = "weight"

    if day_col not in df.columns or wt_col not in df.columns:
        print(f"  错误: 输入文件缺少必要字段（{day_col}, {wt_col}）")
        return

    df[id_col] = df[id_col].astype(str).str.strip()
    df[day_col] = pd.to_numeric(df[day_col], errors="coerce")
    df[wt_col] = pd.to_numeric(df[wt_col], errors="coerce")
    df = df.dropna(subset=[id_col, day_col, wt_col])
    total_rows_after_basic_clean = len(df)

    # 仅保留孕期记录
    mask = (df[day_col] >= PREGNANCY_DAY_MIN) & (df[day_col] <= PREGNANCY_DAY_MAX)
    if "is_postpartum_normal" in df.columns:
        pp_mask = parse_bool_series(df["is_postpartum_normal"])
        mask = mask & (~pp_mask)

    preg_df = df.loc[mask, [id_col, day_col, wt_col]].copy()
    preg_df = preg_df.sort_values([id_col, day_col])

    grouped = preg_df.groupby(id_col, sort=False)
    total_groups = len(grouped)

    flagged_records = []
    flagged_count = 0
    skipped_too_few_points = 0

    for i, (nid, grp) in enumerate(grouped):
        if i % PROGRESS_EVERY == 0:
            print(f"  扫描进度: {i}/{total_groups} ...")

        days = grp[day_col].values.astype(float)
        weights = grp[wt_col].values.astype(float)

        if len(days) < MIN_POINTS:
            skipped_too_few_points += 1
            continue

        max_drop, peak_idx, trough_idx, max_step_drop = find_max_drawdown(days, weights)
        if max_drop <= DROP_THRESHOLD_KG:
            continue

        flagged_count += 1

        peak_day = int(days[peak_idx])
        peak_w = float(weights[peak_idx])
        trough_day = int(days[trough_idx])
        trough_w = float(weights[trough_idx])
        day_span = trough_day - peak_day

        flagged_records.append(
            {
                "项目流水号": nid,
                "孕期数据点数": len(days),
                "下降起点_day": peak_day,
                "下降起点_weight": round(peak_w, 3),
                "下降终点_day": trough_day,
                "下降终点_weight": round(trough_w, 3),
                "最大下降kg": round(max_drop, 3),
                "下降跨越天数": int(day_span),
                "最大单步下降kg": round(max_step_drop, 3),
            }
        )

        plot_sample(
            nid=nid,
            days=days,
            weights=weights,
            peak_idx=peak_idx,
            trough_idx=trough_idx,
            max_drop=max_drop,
            max_step_drop=max_step_drop,
            out_dir=PLOT_DIR,
        )

    flagged_df = pd.DataFrame(flagged_records)
    flagged_df = flagged_df.sort_values(
        by=["最大下降kg", "最大单步下降kg"], ascending=[False, False]
    ).reset_index(drop=True)
    flagged_df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    ratio = (flagged_count / total_groups * 100.0) if total_groups > 0 else 0.0
    lines = [
        "=== 孕期体重下降 >5kg 扫描报告 ===",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"输入文件: {INPUT_CSV}",
        f"基础清洗后记录数: {total_rows_after_basic_clean:,}",
        f"孕期记录数(0~294天, 去产后标记): {len(preg_df):,}",
        f"参与扫描样本数: {total_groups:,}",
        f"点数过少样本(<{MIN_POINTS}点): {skipped_too_few_points:,}",
        f"命中样本数(最大下降>{DROP_THRESHOLD_KG}kg): {flagged_count:,} ({ratio:.2f}%)",
        "",
        "参数:",
        f"  最大下降阈值: {DROP_THRESHOLD_KG} kg",
        f"  孕期范围: {PREGNANCY_DAY_MIN}~{PREGNANCY_DAY_MAX} 天",
        "",
        "输出:",
        f"  样本清单: {REPORT_CSV}",
        f"  轨迹图目录: {PLOT_DIR}/",
    ]

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\n扫描完成。请结合轨迹图进行人工复核。")


if __name__ == "__main__":
    main()

