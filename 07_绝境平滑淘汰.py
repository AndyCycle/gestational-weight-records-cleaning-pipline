import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 7] 终极平滑与死错淘汰 ===")

INPUT_CSV = r"E:\文件\研究生\项目\肥胖分布统计\宝安\HIS系统\清洗管线重构_三步走\06_产后断崖锁定版.csv"
OUT_DIR = r"E:\文件\研究生\项目\肥胖分布统计\宝安\HIS系统\清洗管线重构_三步走"
OUT_CSV = os.path.join(OUT_DIR, "07_终极清洗结果版_可用于插值.csv")
LOG_FILE = os.path.join(OUT_DIR, "07_极值死错清理_日志.txt")
PLOT_DIR = os.path.join(OUT_DIR, "07_Plots_终极死错剔除")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_repair(nid, days, w_raw, removed_days, removed_weights, logs):
    plt.figure(figsize=(10, 6))
    plt.plot(days, w_raw, color='blue', linestyle='-', marker='o', alpha=0.5, label='原数据')
    
    if removed_days: plt.scatter(removed_days, removed_weights, color='red', s=120, marker='X', zorder=10, label='被淘汰死错 (NaN)')
    plt.title(f"ID: {nid} | 绝境死错淘汰")
    plt.xlabel("孕周(天)")
    plt.ylabel("体重 (kg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    info_text = "\n".join(logs)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{nid}_RemoveNaN.png"), dpi=100)
    plt.close()

def remove_dead_errors(group):
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    
    # 06引入的免修金牌列 (如果有)
    is_pp = []
    if 'is_postpartum_normal' in group.columns:
        is_pp = group['is_postpartum_normal'].values
    else:
        is_pp = np.zeros(len(w_orig), dtype=bool)
        
    valid_mask = ~pd.isna(w_orig)
    v_idx = np.where(valid_mask)[0]
    n_v = len(v_idx)
    
    logs, marked = [], False
    removed_days = []
    removed_weights = []
    
    if n_v >= 3:
        for i in range(n_v):
            idx = v_idx[i]
            
            # 拥有免死金牌 (断崖合法) 直接跳过
            if is_pp[idx]: continue
            
            curr_w = w_orig[idx]
            
            # 取出当前点前后的若干个有效邻居（最多前后各2个，排除自己）
            # 这有助于避免相邻的错误点误导基准线，也可以抵抗双连死点的污染
            start = max(0, i - 2)
            end = min(n_v, i + 3)
            
            neighbors_idx = [v_idx[j] for j in range(start, end) if j != i]
            if not neighbors_idx: continue
                
            # 滤除之前作为死点被删掉的 NaN 邻居
            neighbors_w = [w_orig[j] for j in neighbors_idx if not pd.isna(w_orig[j])]
            if not neighbors_w: continue
                
            ref_med = np.median(neighbors_w)
            
            # 为了计算动态天数容忍跨度，探测它覆盖的最远有效邻居距离
            max_days_span = 3
            for j in neighbors_idx:
                if not pd.isna(w_orig[j]):
                    max_days_span = max(max_days_span, abs(days[j] - days[idx]))
                    
            # 基础宽容，并且随着天数增加，防病理性跌落误删
            allowable_jump = 16.0 + 0.25 * max_days_span
            
            if abs(curr_w - ref_med) > allowable_jump:
                old_w = w_orig[idx]
                w_orig[idx] = np.nan
                marked = True
                logs.append(f"Day {days[idx]}d: 死亡越界点剔除 | {old_w:.1f} (邻居中位 {ref_med:.1f}, 界限 {allowable_jump:.1f}) -> NaN")
                removed_days.append(days[idx])
                removed_weights.append(curr_w)

    group['weight_cleaned'] = w_orig
    return group, logs, marked, removed_days, removed_weights

def main():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    if id_col not in df.columns: df.rename(columns={df.columns[0]: id_col}, inplace=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    frames, all_logs, marked_count = [], [], 0
    total = len(grouped)
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0: print(f"07处理进度: {i}/{total}...")
        c_group, logs, marked, rd, rw = remove_dead_errors(group.copy())
        frames.append(c_group)
        if marked:
            marked_count += 1
            all_logs.append(f"[{nid}] 清理死错")
            all_logs.extend(["  " + l for l in logs])
            plot_repair(nid, c_group['gestation_day'].values, c_group['weight'].values, rd, rw, logs)
            
    final_df = pd.concat(frames, ignore_index=True)
    # 最后将最终清洗好的列赋给 weight，其它列如 weight_raw_p* 都保留。
    final_df.rename(columns={'weight': 'weight_raw_p7', 'weight_cleaned': 'weight'}, inplace=True)
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_logs))
    print(f"STEP 7 完成！共成功清理生理性死错点（设为NaN）的产妇: {marked_count} 例。")
    print(f"\n全部多脚本流水线清洗完成，最终可用底表产出: {OUT_CSV}")

if __name__ == '__main__':
    main()
