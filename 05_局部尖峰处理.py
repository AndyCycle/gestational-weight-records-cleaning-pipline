import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 5] 局部尖峰与深谷插值修复 ===")

INPUT_CSV = r"E:\文件\研究生\项目\肥胖分布统计\宝安\市妇幼系统\清洗\04_孕前与早孕异动修复版.csv"
OUT_DIR = r"E:\文件\研究生\项目\肥胖分布统计\宝安\市妇幼系统\清洗"
OUT_CSV = os.path.join(OUT_DIR, "05_局部尖峰处理版.csv")
LOG_FILE = os.path.join(OUT_DIR, "05_局部尖峰修复_日志.txt")
PLOT_DIR = os.path.join(OUT_DIR, "05_Plots_局部尖峰")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_repair(nid, days, w_raw, w_clean, logs):
    plt.figure(figsize=(10, 6))
    plt.plot(days, w_raw, color='lightgray', linestyle='--', marker='o')
    plt.plot(days, w_clean, color='blue', linestyle='-', marker='D')
    changed_days, changed_cleans = [], []
    for d, r, c in zip(days, w_raw, w_clean):
        if not pd.isna(r) and not pd.isna(c) and abs(r - c) > 0.01:
            changed_days.append(d)
            changed_cleans.append(c)
    if changed_days: plt.scatter(changed_days, changed_cleans, color='red', s=100, zorder=5)
    plt.title(f"ID: {nid} | 局部尖峰/深谷修复")
    plt.xlabel("孕周(天)")
    plt.ylabel("体重 (kg)")
    plt.grid(True, alpha=0.3)
    info_text = "\n".join(logs)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{nid}_Spike.png"), dpi=100)
    plt.close()

def clean_spikes(group):
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    valid_mask = ~pd.isna(w_orig)
    if valid_mask.sum() < 3: return group, [], False
        
    v_idx = np.where(valid_mask)[0]
    n_v = len(v_idx)
    logs, changed = [], False
    
    for i in range(1, n_v - 1):
        idx_curr = v_idx[i]
        curr_w = w_orig[idx_curr]
        
        # 寻找左右邻居
        left_w = w_orig[v_idx[i-1]]
        right_w = w_orig[v_idx[i+1]]
        
        # 增加邻居一致性校验：如果左右邻居由于本身有错导致落差极大，直接放弃修复该点，留给07平滑去剔除（避免发生多米诺骨牌级联误伤）
        if abs(left_w - right_w) > 18:
            continue
            
        ref_med = np.median([left_w, right_w])
        
        # 向上孤立尖峰 (斤)
        if curr_w - ref_med > 25 and 1.6 <= curr_w / ref_med <= 3.0:
            new_w = round(curr_w / 2.0, 2)
            # 进行贴合度复检：结合天数的斜率检测 (基础允差 6kg + 0.15kg/day)
            allow_l = 6.0 + max(3, days[idx_curr] - days[v_idx[i-1]]) * 0.15
            allow_r = 6.0 + max(3, days[v_idx[i+1]] - days[idx_curr]) * 0.15
            if abs(new_w - left_w) <= allow_l and abs(new_w - right_w) <= allow_r:
                w_orig[idx_curr] = new_w
                logs.append(f"Day {days[idx_curr]}d: 向上尖峰(斤) | {curr_w:.1f} -> {new_w:.1f} (贴合参考 {ref_med:.1f})")
                changed = True
            
        # 向下孤立深谷 (公斤)
        elif ref_med - curr_w > 25 and 1.6 <= ref_med / curr_w <= 3.0:
            new_w = round(curr_w * 2.0, 2)
            allow_l = 6.0 + max(3, days[idx_curr] - days[v_idx[i-1]]) * 0.15
            allow_r = 6.0 + max(3, days[v_idx[i+1]] - days[idx_curr]) * 0.15
            if abs(new_w - left_w) <= allow_l and abs(new_w - right_w) <= allow_r:
                w_orig[idx_curr] = new_w
                logs.append(f"Day {days[idx_curr]}d: 向下深谷(公里) | {curr_w:.1f} -> {new_w:.1f} (贴合参考 {ref_med:.1f})")
                changed = True

    # 简单处理连续两个点的尖峰
    for i in range(1, n_v - 2):
        idx1 = v_idx[i]
        idx2 = v_idx[i+1]
        w1, w2 = w_orig[idx1], w_orig[idx2]
        
        left_w = w_orig[v_idx[i-1]]
        right_w = w_orig[v_idx[i+2]]
        
        if abs(left_w - right_w) > 20:
            continue
            
        ref_med = np.median([left_w, right_w])
        
        # 双尖峰
        if w1 - ref_med > 25 and w2 - ref_med > 25 and 1.6 <= w1 / ref_med <= 3.0 and 1.6 <= w2 / ref_med <= 3.0:
            nw1 = round(w1 / 2.0, 2)
            nw2 = round(w2 / 2.0, 2)
            # 同样引入动态时间斜率检测
            allow_l = 6.0 + max(3, days[idx1] - days[v_idx[i-1]]) * 0.15
            allow_r = 6.0 + max(3, days[v_idx[i+2]] - days[idx2]) * 0.15
            allow_mid = 5.0 + max(1, days[idx2] - days[idx1]) * 0.15
            # 仅当两点修补后都能合群再下手
            if abs(nw1 - left_w) <= allow_l and abs(nw2 - right_w) <= allow_r and abs(nw1 - nw2) <= allow_mid:
                w_orig[idx1] = nw1
                w_orig[idx2] = nw2
                logs.append(f"Day {days[idx1]}d & {days[idx2]}d: 双尖峰 | {w1:.1f},{w2:.1f} -> {nw1:.1f},{nw2:.1f}")
                changed = True
            
    group['weight_cleaned'] = w_orig
    return group, logs, changed

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
        if i % 10000 == 0: print(f"05处理进度: {i}/{total}...")
        c_group, logs, changed = clean_spikes(group.copy())
        frames.append(c_group)
        if changed:
            changed_count += 1
            all_logs.append(f"[{nid}] 局部尖峰/深谷")
            all_logs.extend(["  " + l for l in logs])
            plot_repair(nid, c_group['gestation_day'].values, c_group['weight'].values, c_group['weight_cleaned'].values, logs)
            
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(columns={'weight': 'weight_raw_p5', 'weight_cleaned': 'weight'}, inplace=True)
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_logs))
    print(f"STEP 5 完成！修复: {changed_count} 例。")

if __name__ == '__main__':
    main()
