import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 8] 全面质控扫描与可视化审计 ===")

INPUT_CSV = r"E:\文件\研究生\项目\肥胖分布统计\宝安\HIS系统\清洗管线重构_三步走\07_终极清洗结果版_可用于插值.csv"
OUT_DIR = r"E:\文件\研究生\项目\肥胖分布统计\宝安\HIS系统\清洗管线重构_三步走"
REPORT_FILE = os.path.join(OUT_DIR, "08_质控扫描报告.txt")
FLAGGED_CSV = os.path.join(OUT_DIR, "08_被标记的可疑样本一览.csv")

# 各类异常的绘图输出目录
PLOT_CATEGORIES = {
    "A_极端BMI":        "08_QC_A_极端BMI",
    "B_高速增重":       "08_QC_B_高速增重",
    "C_非产后骤降":     "08_QC_C_非产后骤降",
    "D_总增重异常":     "08_QC_D_总增重异常",
    "E_残余尖峰嫌疑":   "08_QC_E_残余尖峰嫌疑",
    "F_平坦复制嫌疑":   "08_QC_F_平坦复制嫌疑",
    "G_身高体重不协调":  "08_QC_G_身高体重不协调",
    "H_数据点稀疏":     "08_QC_H_数据点稀疏",
}

for cat_key, cat_dir in PLOT_CATEGORIES.items():
    os.makedirs(os.path.join(OUT_DIR, cat_dir), exist_ok=True)

# ============= 扫描参数 =============
# A: 极端BMI阈值
BMI_LOW = 14.0
BMI_HIGH = 50.0

# B: 高速增重 (每日最大合理增重，单位 kg/day)
MAX_DAILY_GAIN = 0.25  # 约 1.75 kg/week

# C: 非产后骤降 (在没有产后标签的情况下短期暴跌)
DROP_THRESHOLD_KG = 8.0
DROP_MAX_DAYS = 28

# D: 总增重异常 (整个孕期的总增重范围)
TOTAL_GWG_LOW = -5.0   # 允许极端孕吐减重
TOTAL_GWG_HIGH = 35.0  # 极端增重上限

# E: 残余尖峰嫌疑 (与滑窗中位数的偏差)
SPIKE_RESIDUAL_THRESHOLD = 12.0

# F: 平坦复制嫌疑 (连续N次完全相同的体重)
FLAT_MIN_CONSECUTIVE = 4

# G: 身高体重不协调 (整个孕期中位BMI范围)
MEDIAN_BMI_LOW = 14.5
MEDIAN_BMI_HIGH = 48.0

# H: 数据点稀疏
MIN_VALID_POINTS = 3

# ============= 绘图函数 =============
def plot_flagged(nid, days, weights, height, flags_text, category_key):
    fig, ax1 = plt.subplots(figsize=(11, 6))
    
    valid_mask = ~pd.isna(weights)
    v_days = days[valid_mask]
    v_weights = weights[valid_mask]
    
    ax1.plot(v_days, v_weights, color='#2196F3', linestyle='-', marker='o', markersize=5, label='体重 (kg)')
    ax1.set_xlabel("孕周(天)")
    ax1.set_ylabel("体重 (kg)", color='#2196F3')
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax1.grid(True, alpha=0.2)
    
    # 如果有身高则在右轴画 BMI
    if height and height > 0:
        h_m = height / 100.0 if height > 3 else height
        bmis = v_weights / (h_m * h_m)
        ax2 = ax1.twinx()
        ax2.plot(v_days, bmis, color='#FF9800', linestyle='--', marker='s', markersize=3, alpha=0.6, label='BMI')
        ax2.set_ylabel("BMI", color='#FF9800')
        ax2.tick_params(axis='y', labelcolor='#FF9800')
        
        # BMI 红线
        ax2.axhline(y=BMI_LOW, color='red', linestyle=':', alpha=0.4, label=f'BMI={BMI_LOW}')
        ax2.axhline(y=BMI_HIGH, color='red', linestyle=':', alpha=0.4, label=f'BMI={BMI_HIGH}')
    
    plt.title(f"QC Flag | ID: {nid}", fontsize=12)
    
    # 标注标签信息
    flag_str = "\n".join(flags_text[:6])  # 最多显示6条，防止溢出
    ax1.text(0.02, 0.98, flag_str, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.9, edgecolor='#FF9800'))
    
    plt.tight_layout()
    cat_dir = PLOT_CATEGORIES[category_key]
    plt.savefig(os.path.join(OUT_DIR, cat_dir, f"{nid}.png"), dpi=100)
    plt.close()

# ============= 扫描逻辑 =============
def scan_sample(nid, group):
    """对单个样本进行全面质控扫描，返回所有发现的标记列表"""
    w = group['weight'].values.copy()
    days = group['gestation_day'].values
    valid_mask = ~pd.isna(w)
    v_idx = np.where(valid_mask)[0]
    n_v = len(v_idx)
    
    # 获取身高
    h_series = group['height'].dropna() if 'height' in group.columns else pd.Series(dtype=float)
    H_cm = None
    if len(h_series) > 0:
        h_val = float(h_series.iloc[0])
        if h_val > 100:
            H_cm = h_val
        elif 1.0 < h_val < 3.0:
            H_cm = h_val * 100
    
    # 产后标签(如果有)
    is_pp = np.zeros(len(w), dtype=bool)
    if 'is_postpartum_normal' in group.columns:
        pp_vals = group['is_postpartum_normal'].values
        is_pp = np.array([bool(v) if not pd.isna(v) else False for v in pp_vals])
    
    flags = []  # [(类别key, 描述文本)]
    
    if n_v < MIN_VALID_POINTS:
        flags.append(("H_数据点稀疏", f"仅有 {n_v} 个有效体重数据点"))
        return flags, H_cm
    
    v_w = w[v_idx]
    v_days = days[v_idx]
    
    # --- A: 极端BMI ---
    if H_cm:
        h_m = H_cm / 100.0
        bmis = v_w / (h_m * h_m)
        low_bmi_idx = np.where(bmis < BMI_LOW)[0]
        high_bmi_idx = np.where(bmis > BMI_HIGH)[0]
        for li in low_bmi_idx:
            flags.append(("A_极端BMI", f"Day {v_days[li]}d: BMI={bmis[li]:.1f} < {BMI_LOW} (w={v_w[li]:.1f}kg)"))
        for hi in high_bmi_idx:
            flags.append(("A_极端BMI", f"Day {v_days[hi]}d: BMI={bmis[hi]:.1f} > {BMI_HIGH} (w={v_w[hi]:.1f}kg)"))
    
    # --- B: 高速增重 ---
    for i in range(1, n_v):
        d_diff = v_days[i] - v_days[i-1]
        if d_diff <= 0: continue
        w_diff = v_w[i] - v_w[i-1]
        daily_rate = w_diff / d_diff
        if daily_rate > MAX_DAILY_GAIN and d_diff <= 42:
            flags.append(("B_高速增重", f"Day {v_days[i-1]}d->{v_days[i]}d: +{w_diff:.1f}kg/{d_diff}天 ({daily_rate:.3f}kg/d)"))
    
    # --- C: 非产后骤降 ---
    for i in range(1, n_v):
        d_diff = v_days[i] - v_days[i-1]
        w_drop = v_w[i-1] - v_w[i]
        if w_drop > DROP_THRESHOLD_KG and d_diff <= DROP_MAX_DAYS:
            # 排除已经被标记为产后的点
            if not is_pp[v_idx[i]]:
                flags.append(("C_非产后骤降", f"Day {v_days[i-1]}d->{v_days[i]}d: -{w_drop:.1f}kg/{d_diff}天 (非产后)"))
    
    # --- D: 总增重异常 ---
    # 寻找 day=0 的锚点和最后一个非产后点
    pre_w = None
    for i in range(n_v):
        if v_days[i] <= 7:
            pre_w = v_w[i]
            break
    
    last_ante_w = None
    for i in range(n_v - 1, -1, -1):
        if not is_pp[v_idx[i]]:
            last_ante_w = v_w[i]
            break
    
    if pre_w is not None and last_ante_w is not None:
        total_gwg = last_ante_w - pre_w
        if total_gwg < TOTAL_GWG_LOW:
            flags.append(("D_总增重异常", f"总GWG={total_gwg:.1f}kg < {TOTAL_GWG_LOW}kg (孕前{pre_w:.1f} -> 末期{last_ante_w:.1f})"))
        elif total_gwg > TOTAL_GWG_HIGH:
            flags.append(("D_总增重异常", f"总GWG={total_gwg:.1f}kg > {TOTAL_GWG_HIGH}kg (孕前{pre_w:.1f} -> 末期{last_ante_w:.1f})"))
    
    # --- E: 残余尖峰嫌疑 ---
    if n_v >= 3:
        for i in range(1, n_v - 1):
            left_w = v_w[i-1]
            right_w = v_w[i+1]
            ref = np.median([left_w, right_w])
            dev = abs(v_w[i] - ref)
            if dev > SPIKE_RESIDUAL_THRESHOLD:
                flags.append(("E_残余尖峰嫌疑", f"Day {v_days[i]}d: w={v_w[i]:.1f} 偏离邻居中位{ref:.1f} 达{dev:.1f}kg"))
    
    # --- F: 平坦复制嫌疑 ---
    if n_v >= FLAT_MIN_CONSECUTIVE:
        run_start = 0
        for i in range(1, n_v):
            if abs(v_w[i] - v_w[run_start]) < 0.01:
                if (i - run_start + 1) >= FLAT_MIN_CONSECUTIVE:
                    flags.append(("F_平坦复制嫌疑", f"Day {v_days[run_start]}d-{v_days[i]}d: 连续{i-run_start+1}次相同体重 {v_w[run_start]:.1f}kg"))
                    break
            else:
                run_start = i
    
    # --- G: 身高体重不协调 ---
    if H_cm:
        h_m = H_cm / 100.0
        med_bmi = np.median(v_w) / (h_m * h_m)
        if med_bmi < MEDIAN_BMI_LOW:
            flags.append(("G_身高体重不协调", f"中位BMI={med_bmi:.1f} < {MEDIAN_BMI_LOW} (身高{H_cm:.0f}cm, 中位体重{np.median(v_w):.1f}kg)"))
        elif med_bmi > MEDIAN_BMI_HIGH:
            flags.append(("G_身高体重不协调", f"中位BMI={med_bmi:.1f} > {MEDIAN_BMI_HIGH} (身高{H_cm:.0f}cm, 中位体重{np.median(v_w):.1f}kg)"))
    
    return flags, H_cm

# ============= 主函数 =============
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到输入文件: {INPUT_CSV}")
        return
        
    print(f"正在加载终极清洗底表: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    if id_col not in df.columns:
        df.rename(columns={df.columns[0]: id_col}, inplace=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    total = len(grouped)
    
    # 统计
    category_counts = defaultdict(int)
    category_samples = defaultdict(list)
    all_flagged_rows = []
    flagged_sample_count = 0
    clean_sample_count = 0
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0:
            print(f"08扫描进度: {i}/{total}...")
        
        flags, H_cm = scan_sample(nid, group)
        
        if flags:
            flagged_sample_count += 1
            
            # 统计各类计数
            seen_cats = set()
            for cat_key, desc in flags:
                category_counts[cat_key] += 1
                if cat_key not in seen_cats:
                    category_samples[cat_key].append(nid)
                    seen_cats.add(cat_key)
                all_flagged_rows.append({'项目流水号': nid, '类别': cat_key, '描述': desc})
            
            # 对每个被标记的类别分别绘图（取优先级最高的类别作为主绘图目录）
            primary_cat = flags[0][0]
            flags_text = [f"[{c}] {d}" for c, d in flags]
            plot_flagged(nid, group['gestation_day'].values, group['weight'].values,
                         H_cm, flags_text, primary_cat)
        else:
            clean_sample_count += 1
    
    # ============= 输出报告 =============
    report_lines = []
    report_lines.append(f"体重时序数据全面质控扫描报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append(f"总样本数: {total}")
    report_lines.append(f"通过质控（零标记）: {clean_sample_count} ({clean_sample_count/total*100:.1f}%)")
    report_lines.append(f"被标记可疑样本数: {flagged_sample_count} ({flagged_sample_count/total*100:.1f}%)")
    report_lines.append("")
    report_lines.append("--- 各类异常分布统计 ---")
    
    for cat_key in sorted(PLOT_CATEGORIES.keys()):
        cnt = category_counts.get(cat_key, 0)
        n_samples = len(category_samples.get(cat_key, []))
        report_lines.append(f"  {cat_key}: {cnt} 条标记, 涉及 {n_samples} 个独立样本")
    
    report_lines.append("")
    report_lines.append("--- 扫描阈值参数 ---")
    report_lines.append(f"  BMI下界: {BMI_LOW}")
    report_lines.append(f"  BMI上界: {BMI_HIGH}")
    report_lines.append(f"  每日最大增重: {MAX_DAILY_GAIN} kg/d")
    report_lines.append(f"  非产后骤降阈值: {DROP_THRESHOLD_KG} kg / {DROP_MAX_DAYS} 天")
    report_lines.append(f"  总GWG范围: [{TOTAL_GWG_LOW}, {TOTAL_GWG_HIGH}] kg")
    report_lines.append(f"  残余尖峰偏离阈值: {SPIKE_RESIDUAL_THRESHOLD} kg")
    report_lines.append(f"  平坦复制连续次数: {FLAT_MIN_CONSECUTIVE}")
    report_lines.append(f"  中位BMI范围: [{MEDIAN_BMI_LOW}, {MEDIAN_BMI_HIGH}]")
    report_lines.append(f"  最少有效数据点: {MIN_VALID_POINTS}")
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 输出标记清单CSV
    if all_flagged_rows:
        flagged_df = pd.DataFrame(all_flagged_rows)
        flagged_df.to_csv(FLAGGED_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\nSTEP 8 质控扫描完成！")
    print(f"  报告文件: {REPORT_FILE}")
    print(f"  标记清单: {FLAGGED_CSV}")
    print(f"  各类绘图目录: {OUT_DIR}/08_QC_*")

if __name__ == '__main__':
    main()
