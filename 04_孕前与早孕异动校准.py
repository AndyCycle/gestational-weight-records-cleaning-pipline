import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 4] 孕前与早孕异动校准 ===")

INPUT_CSV = r"03_全局与阶跃修复版.csv"
OUT_DIR = r"gestational-weight-records-cleaning-pipline"
OUT_CSV = os.path.join(OUT_DIR, "04_孕前与早孕异动修复版.csv")
LOG_FILE = os.path.join(OUT_DIR, "04_早孕异动修复_日志.txt")
PLOT_DIR = os.path.join(OUT_DIR, "04_Plots_早孕异动")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_repair(nid, days, w_raw, w_clean, error_type, logs):
    plt.figure(figsize=(10, 6))
    plt.plot(days, w_raw, color='lightgray', linestyle='--', marker='o')
    plt.plot(days, w_clean, color='blue', linestyle='-', marker='D')
    changed_days, changed_cleans = [], []
    for d, r, c in zip(days, w_raw, w_clean):
        if not pd.isna(r) and not pd.isna(c) and abs(r - c) > 0.01:
            changed_days.append(d)
            changed_cleans.append(c)
    if changed_days: plt.scatter(changed_days, changed_cleans, color='red', s=100, zorder=5)
    plt.title(f"ID: {nid} | {error_type}")
    plt.xlabel("孕周(天)")
    plt.ylabel("体重 (kg)")
    plt.grid(True, alpha=0.3)
    info_text = "\\n".join(logs)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{nid}_{error_type}.png"), dpi=100)
    plt.close()

def clean_early_preg(group):
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    valid_mask = ~pd.isna(w_orig)
    if valid_mask.sum() < 2: return group, [], False, ""
        
    v_idx = np.where(valid_mask)[0]
    # pre_idx: 孕前或前20天. early_idx: 20-140天.
    pre_idx = [i for i in v_idx if days[i] <= 21]
    early_idx = [i for i in v_idx if 21 < days[i] <= 140]
    
    if not pre_idx or not early_idx:
        group['weight_cleaned'] = w_orig
        return group, [], False, ""
        
    pre_w = [w_orig[i] for i in pre_idx]
    early_w = [w_orig[i] for i in early_idx[:2]]
    
    med_pre = np.median(pre_w)
    med_early = np.median(early_w)
    logs, changed, err_type = [], False, ""
    
    h_series = group['height'].dropna()
    H = None
    if len(h_series) > 0:
        h_val = float(h_series.max())
        if h_val > 100: H = h_val / 100.0
        elif 1.0 < h_val < 3.0: H = h_val
        
    # 高开低走
    if med_pre - med_early > 30 and 1.6 <= med_pre / med_early <= 2.8:
        for i in pre_idx:
            old_w = w_orig[i]
            new_w = round(old_w / 2.0, 2)
            
            # 生理极限保护
            if H:
                if new_w / (H * H) < 14.0:
                    continue
                    
            if old_w > 75:
                w_orig[i] = new_w
                logs.append(f"Day {days[i]}d: 高开低走(孕前斤) | {old_w:.1f} -> {new_w:.1f}")
                changed = True
        err_type = "Drop_高开低走"
        
    # 火箭发射
    elif med_early - med_pre > 30 and 1.6 <= med_early / med_pre <= 3.0:
        for i in early_idx[:2]:
            old_w = w_orig[i]
            new_w = round(old_w / 2.0, 2)
            
            # 生理极限保护
            if H:
                if new_w / (H * H) < 14.0:
                    continue
                    
            if old_w > 75:
                w_orig[i] = new_w
                logs.append(f"Day {days[i]}d: 火箭发射(早期斤) | {old_w:.1f} -> {new_w:.1f}")
                changed = True
        err_type = "Spike_火箭发射"
        
    group['weight_cleaned'] = w_orig
    return group, logs, changed, err_type

def main():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    if id_col not in df.columns: df.rename(columns={df.columns[0]: id_col}, inplace=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    frames, all_logs, changed_count = [], [], 0
    total = len(grouped)
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0: print(f"04处理进度: {i}/{total}...")
        c_group, logs, changed, err_type = clean_early_preg(group.copy())
        frames.append(c_group)
        if changed:
            changed_count += 1
            all_logs.append(f"[{nid}] {err_type}")
            all_logs.extend(["  " + l for l in logs])
            plot_repair(nid, c_group['gestation_day'].values, c_group['weight'].values, c_group['weight_cleaned'].values, err_type, logs)
            
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(columns={'weight': 'weight_raw_p4', 'weight_cleaned': 'weight'}, inplace=True)
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_logs))
    print(f"STEP 4 完成！修复: {changed_count} 例。")

if __name__ == '__main__':
    main()
