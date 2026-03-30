import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 3] 全局斤系与长程阶跃修复 ===")

INPUT_CSV = r"02_初步清洗_去低级失误版.csv"
OUT_DIR = r"gestational-weight-records-cleaning-pipline"
OUT_CSV = os.path.join(OUT_DIR, "03_全局与阶跃修复版.csv")
LOG_FILE = os.path.join(OUT_DIR, "03_全局与阶跃修复_日志.txt")
PLOT_DIR = os.path.join(OUT_DIR, "03_Plots_全局与阶跃")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_repair(nid, days, w_raw, w_clean, error_type, logs):
    plt.figure(figsize=(10, 6))
    plt.plot(days, w_raw, color='lightgray', linestyle='--', marker='o', label='原始体重 (Raw)')
    plt.plot(days, w_clean, color='blue', linestyle='-', marker='D', alpha=0.7, label='修复后 (Cleaned)')
    
    # 标出被修改的点
    changed_days = []
    changed_cleans = []
    for d, r, c in zip(days, w_raw, w_clean):
        if not pd.isna(r) and not pd.isna(c) and abs(r - c) > 0.01:
            changed_days.append(d)
            changed_cleans.append(c)
            
    if changed_days:
        plt.scatter(changed_days, changed_cleans, color='red', s=100, zorder=5, label='修复点')
        
    plt.title(f"ID: {nid} | 类型: {error_type}")
    plt.xlabel("孕周(天) Gestation Day")
    plt.ylabel("体重 Weight (kg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    info_text = "\n".join(logs)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{nid}_{error_type}.png"), dpi=100)
    plt.close()

def clean_global_and_step(group):
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    
    valid_mask = ~pd.isna(w_orig)
    if valid_mask.sum() < 2:
        return group, [], False, ""
        
    v_idx = np.where(valid_mask)[0]
    v_w = w_orig[valid_mask]
    
    logs = []
    changed = False
    error_type = ""
    
    h_series = group['height'].dropna() if 'height' in group.columns else []
    H = None
    if len(h_series) > 0:
        h_val = float(h_series.max())
        if h_val > 100: H = h_val / 100.0
        elif 1.0 < h_val < 3.0: H = h_val

    # 1. 全局斤系 (最低 > 80，中位数 > 105)
    if np.min(v_w) >= 80 and np.median(v_w) >= 105:
        # BMI保护校验
        safe_to_convert = True
        if H:
            simulated_min_bmi = (np.min(v_w) / 2.0) / (H * H)
            if simulated_min_bmi < 13.5:
                safe_to_convert = False
                
        if safe_to_convert:
            for curr_i in v_idx:
                w_orig[curr_i] = round(w_orig[curr_i] / 2.0, 2)
                logs.append(f"Day {days[curr_i]}d: 全局斤 | {v_w[np.where(v_idx==curr_i)[0][0]]:.1f} -> {w_orig[curr_i]:.1f}")
            changed = True
            error_type = "Global_Jin_全局斤系"
            group['weight_cleaned'] = w_orig
            return group, logs, changed, error_type
        
    # 2. 长程阶跃 —— 最佳切分策略
    n_v = len(v_w)
    if n_v >= 3:
        best_split = -1
        best_score = float('inf')   # 越小越好：两段的加权方差之和
        
        for i in range(1, n_v):
            part1 = v_w[:i]
            part2 = v_w[i:]
            if len(part1) == 0 or len(part2) == 0:
                continue
            
            med1 = np.median(part1)
            med2 = np.median(part2)
            
            # ---- 放宽绝对差门槛(30)，ratio 本身已是斤/kg 的强约束 ----
            if abs(med1 - med2) > 30:
                ratio = max(med1, med2) / min(med1, med2)
                if 1.6 <= ratio <= 2.8:
                    std1 = np.std(part1) if len(part1) > 1 else 0
                    std2 = np.std(part2) if len(part2) > 1 else 0
                    if std1 < 18 and std2 < 22:
                        
                        # BMI 生理红线校验
                        safe_to_convert = True
                        if med1 > med2:
                            if H and (np.min(part1) / 2.0) / (H * H) < 13.5:
                                safe_to_convert = False
                        else:
                            if H and (np.min(part2) / 2.0) / (H * H) < 13.5:
                                safe_to_convert = False
                            
                        if safe_to_convert:
                            # 加权段内方差作为切分质量评分
                            score = (std1 * len(part1) + std2 * len(part2)) / n_v
                            if score < best_score:
                                best_score = score
                                best_split = i
        
        if best_split != -1:
            part1 = v_w[:best_split]
            part2 = v_w[best_split:]
            med1 = np.median(part1)
            med2 = np.median(part2)
            
            # ---- 修正后合理性验证 ----
            # 斤段除以2后的中位数，应与kg段中位数差值 < 15kg
            # （正常孕期增重罕有超过15kg，这里给宽松空间）
            if med1 > med2:
                corrected_med = med1 / 2.0
                plausible = abs(corrected_med - med2) < 15
            else:
                corrected_med = med2 / 2.0
                plausible = abs(corrected_med - med1) < 15
            
            if plausible:
                # 判断哪一段是斤（较大的一段是斤）
                if med1 > med2:
                    error_type = "Step_FrontJin_前程斤系"
                    for i_v in range(best_split):
                        curr_i = v_idx[i_v]
                        w_orig[curr_i] = round(w_orig[curr_i] / 2.0, 2)
                        logs.append(f"Day {days[curr_i]}d: 前程阶跃 | {v_w[i_v]:.1f} -> {w_orig[curr_i]:.1f}")
                    changed = True
                else:
                    error_type = "Step_BackJin_后程斤系"
                    for i_v in range(best_split, n_v):
                        curr_i = v_idx[i_v]
                        w_orig[curr_i] = round(w_orig[curr_i] / 2.0, 2)
                        logs.append(f"Day {days[curr_i]}d: 后程阶跃 | {v_w[i_v]:.1f} -> {w_orig[curr_i]:.1f}")
                    changed = True
                    
                if changed:
                    group['weight_cleaned'] = w_orig
                    return group, logs, changed, error_type
            
    group['weight_cleaned'] = w_orig
    return group, [], False, ""

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到输入文件: {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    if id_col not in df.columns: df.rename(columns={df.columns[0]: id_col}, inplace=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    
    frames, all_logs = [], []
    changed_count = 0
    total = len(grouped)
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0: print(f"03处理进度: {i}/{total}...")
        c_group, logs, changed, err_type = clean_global_and_step(group.copy())
        frames.append(c_group)
        if changed:
            changed_count += 1
            all_logs.append(f"[{nid}] 修复类型: {err_type}")
            all_logs.extend(["  " + l for l in logs])
            plot_repair(nid, c_group['gestation_day'].values, c_group['weight_raw_p1'] if 'weight_raw_p1' in c_group.columns else c_group['weight'].values, c_group['weight_cleaned'].values, err_type, logs)
            
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(columns={'weight': 'weight_raw_p3', 'weight_cleaned': 'weight'}, inplace=True)
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_logs))
    print(f"STEP 3 完成！修复: {changed_count} 例。")

if __name__ == '__main__':
    main()
