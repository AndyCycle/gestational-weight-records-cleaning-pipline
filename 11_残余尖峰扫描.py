"""
=== [Pipeline Step 10] 残余尖峰扫描 ===

在 09_day0双源核验版.csv 上执行最终扫描，检测逃过前序清洗步骤的
孤立异常体重记录。仅输出可视化与清单，不修改原始数据，
供人工排查后决定是否手动修正。

检测策略：
  A) 去趋势残差法：对每人的体重序列做线性回归去趋势，再用
     MAD（中位绝对偏差）标准化残差，标记 |z_mad| > 阈值 的点。
     能捕捉"偏离个体自身趋势"的孤立异常。
  B) 局部上下文法：比较每个点与左右邻居的差异，使用自适应阈值
     （基于个体体重量级和邻居间距）。
     能捕捉"相对邻居跳变过大"的尖峰/深谷。
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ─── 路径 ───────────────────────────────────────────────
INPUT_CSV  = r"HIS系统\清洗流程_v2/10_孕前与早孕异动修复版.csv"
OUT_DIR    = r"HIS系统\清洗流程_v2"
PLOT_DIR   = os.path.join(OUT_DIR, "11_Plots_残余尖峰扫描")
REPORT_CSV = os.path.join(OUT_DIR, "11_残余尖峰扫描_可疑清单.csv")
LOG_FILE   = os.path.join(OUT_DIR, "11_残余尖峰扫描_报告.txt")
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── 参数 ───────────────────────────────────────────────
# 策略A: 去趋势残差 MAD z-score 阈值
DETREND_Z_THRESHOLD = 5.0
# 策略A: 最少数据点数（太少则线性回归无意义）
DETREND_MIN_POINTS  = 5

# 策略B: 局部跳变的绝对差阈值（kg）
LOCAL_ABS_THRESHOLD = 15.0
# 策略B: 局部跳变的相对比率阈值（curr/neighbor_median）
LOCAL_RATIO_LOW     = 0.60   # 向下跳变
LOCAL_RATIO_HIGH    = 1.50   # 向上跳变
# 策略B: 邻居之间的最大合理差异（如果邻居本身差太大则跳过）
NEIGHBOR_MAX_DIFF   = 12.0


def mad_zscore(residuals: np.ndarray) -> np.ndarray:
    """用 MAD（中位绝对偏差）计算稳健 z-score，比标准差更抗极端值。"""
    med = np.nanmedian(residuals)
    mad = np.nanmedian(np.abs(residuals - med))
    if mad < 1e-6:
        # MAD 接近零说明序列几乎无波动，回退到标准差
        std = np.nanstd(residuals)
        if std < 1e-6:
            return np.zeros_like(residuals)
        return (residuals - med) / std
    return (residuals - med) / (mad * 1.4826)  # 1.4826 使 MAD 与高斯 SD 等效


def strategy_a_detrend(days: np.ndarray, weights: np.ndarray) -> list[int]:
    """策略A：线性去趋势 + MAD z-score 检测。返回异常点的局部索引列表。"""
    if len(days) < DETREND_MIN_POINTS:
        return []

    # 线性回归去趋势
    coeffs = np.polyfit(days, weights, deg=1)
    trend = np.polyval(coeffs, days)
    residuals = weights - trend

    z = mad_zscore(residuals)
    flags = np.where(np.abs(z) > DETREND_Z_THRESHOLD)[0].tolist()
    return flags


def strategy_b_local(days: np.ndarray, weights: np.ndarray) -> list[int]:
    """策略B：局部上下文跳变检测。返回异常点的局部索引列表。"""
    n = len(days)
    if n < 3:
        return []

    flags = []
    for i in range(1, n - 1):
        left_w  = weights[i - 1]
        curr_w  = weights[i]
        right_w = weights[i + 1]

        # 邻居一致性：如果邻居本身差异过大，无法判断谁是异常
        if abs(left_w - right_w) > NEIGHBOR_MAX_DIFF:
            continue

        ref = np.median([left_w, right_w])
        diff = abs(curr_w - ref)
        ratio = curr_w / ref if ref > 0 else 1.0

        if diff > LOCAL_ABS_THRESHOLD and (ratio > LOCAL_RATIO_HIGH or ratio < LOCAL_RATIO_LOW):
            flags.append(i)

    # 首点检测
    if n >= 3:
        ref = np.median(weights[1:min(4, n)])
        diff = abs(weights[0] - ref)
        ratio = weights[0] / ref if ref > 0 else 1.0
        if diff > LOCAL_ABS_THRESHOLD and (ratio > LOCAL_RATIO_HIGH or ratio < LOCAL_RATIO_LOW):
            flags.append(0)

    # 尾点检测
    if n >= 3:
        ref = np.median(weights[max(0, n - 4):n - 1])
        diff = abs(weights[-1] - ref)
        ratio = weights[-1] / ref if ref > 0 else 1.0
        if diff > LOCAL_ABS_THRESHOLD and (ratio > LOCAL_RATIO_HIGH or ratio < LOCAL_RATIO_LOW):
            flags.append(n - 1)

    return sorted(set(flags))


def plot_flagged(nid, days, weights, flag_indices, reasons, out_dir):
    """为被标记的样本绘制时序图，高亮异常点。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(days, weights, color="#4C78A8", linestyle="-", marker="o",
            markersize=4, linewidth=1.2, label="体重序列", zorder=2)

    # 高亮异常点
    flag_days = [days[i] for i in flag_indices if i < len(days)]
    flag_wts  = [weights[i] for i in flag_indices if i < len(days)]
    ax.scatter(flag_days, flag_wts, color="red", s=120, zorder=5,
               edgecolors="darkred", linewidths=1.5, label="可疑点")

    # 标注每个异常点
    for idx in flag_indices:
        if idx >= len(days):
            continue
        d, w = days[idx], weights[idx]
        reason_str = reasons.get(idx, "")
        ax.annotate(
            f"day{d}\n{w:.1f}kg\n{reason_str}",
            (d, w), textcoords="offset points", xytext=(8, 12),
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.85),
        )

    ax.set_title(f"项目流水号: {nid}  |  可疑点: {len(flag_indices)}", fontsize=12)
    ax.set_xlabel("孕龄 (天)")
    ax.set_ylabel("体重 (kg)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{nid}_spike_scan.png"), dpi=120)
    plt.close(fig)


def main():
    print("=== [Pipeline Step 10] 残余尖峰扫描 ===")
    print(f"  输入: {INPUT_CSV}")
    print(f"  检测参数:")
    print(f"    策略A - 去趋势 MAD z-score 阈值: {DETREND_Z_THRESHOLD}")
    print(f"    策略B - 局部跳变绝对阈值: {LOCAL_ABS_THRESHOLD} kg")
    print(f"    策略B - 局部跳变比率范围: [{LOCAL_RATIO_LOW}, {LOCAL_RATIO_HIGH}]")

    if not os.path.exists(INPUT_CSV):
        print(f"  错误: 找不到 {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = "项目流水号"
    if id_col not in df.columns:
        # 尝试兼容
        if "project_id" in df.columns:
            id_col = "project_id"
        else:
            id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str).str.strip()

    day_col = "gestation_day"
    wt_col  = "weight"

    df[day_col] = pd.to_numeric(df[day_col], errors="coerce")
    df[wt_col]  = pd.to_numeric(df[wt_col],  errors="coerce")
    df = df.dropna(subset=[id_col, day_col, wt_col])

    grouped = df.sort_values([id_col, day_col]).groupby(id_col)
    total_groups = len(grouped)

    all_flags = []      # 汇总清单
    flagged_count = 0

    for i, (nid, grp) in enumerate(grouped):
        if i % 20000 == 0:
            print(f"  扫描进度: {i}/{total_groups} ...")

        days    = grp[day_col].values.astype(float)
        weights = grp[wt_col].values.astype(float)

        # 运行两个策略
        flags_a = strategy_a_detrend(days, weights)
        flags_b = strategy_b_local(days, weights)

        # 合并去重
        all_idx = sorted(set(flags_a) | set(flags_b))
        if not all_idx:
            continue

        # 构建每个异常点的标记原因
        reasons = {}
        for idx in all_idx:
            tags = []
            if idx in flags_a:
                tags.append("去趋势")
            if idx in flags_b:
                tags.append("局部跳变")
            reasons[idx] = "+".join(tags)

        flagged_count += 1

        # 记录到清单
        for idx in all_idx:
            if idx >= len(days):
                continue
            all_flags.append({
                "项目流水号": nid,
                "gestation_day": int(days[idx]),
                "weight": float(weights[idx]),
                "检测策略": reasons[idx],
                "数据点总数": len(days),
            })

        # 绘图
        plot_flagged(nid, days, weights, all_idx, reasons, PLOT_DIR)

    # 输出清单
    flag_df = pd.DataFrame(all_flags)
    flag_df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")

    # 输出报告
    lines = [
        "=== 残余尖峰扫描报告 ===",
        f"扫描样本总数: {total_groups:,}",
        f"被标记样本数: {flagged_count:,}",
        f"被标记数据点总数: {len(all_flags):,}",
        "",
        f"检测参数:",
        f"  策略A - 去趋势 MAD z-score 阈值: {DETREND_Z_THRESHOLD}",
        f"  策略B - 局部跳变绝对阈值: {LOCAL_ABS_THRESHOLD} kg",
        f"  策略B - 局部跳变比率范围: [{LOCAL_RATIO_LOW}, {LOCAL_RATIO_HIGH}]",
        f"  策略B - 邻居最大合理差异: {NEIGHBOR_MAX_DIFF} kg",
        "",
        "输出文件:",
        f"  可疑清单: {REPORT_CSV}",
        f"  可视化目录: {PLOT_DIR}/",
    ]
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\n扫描完成。请人工排查可视化图片后决定是否修正。")


if __name__ == "__main__":
    main()
