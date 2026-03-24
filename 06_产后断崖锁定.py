import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 6] 产后断崖锁定 (结合分娩记录打标免修) ===")

INPUT_CSV = r"E:\文件\研究生\项目\肥胖分布统计\宝安\市妇幼系统\清洗\05_局部尖峰处理版.csv"
OUT_DIR = r"E:\文件\研究生\项目\肥胖分布统计\宝安\市妇幼系统\清洗"
OUT_CSV = os.path.join(OUT_DIR, "06_产后断崖锁定版.csv")
LOG_FILE = os.path.join(OUT_DIR, "06_产后断崖锁定_日志.txt")
PLOT_DIR = os.path.join(OUT_DIR, "06_Plots_产后免修锁定")
os.makedirs(PLOT_DIR, exist_ok=True)

DELIVERY_FILES = [
    r"E:\文件\研究生\项目\宝安妇幼数据搜索\搜索数据\保健号编码后-给宝安\分娩记录-201707-202312-清洗地址后.csv",
    r"E:\文件\研究生\项目\宝安妇幼数据搜索\搜索数据\保健号编码后-给宝安\分娩记录-第5批-2-清洗地址-去重后-icd11_mapped.xlsx",
    r"E:\文件\研究生\项目\宝安妇幼数据搜索\搜索数据\保健号编码后-给宝安\分娩记录-第5批-1-清洗地址后-icd11_mapped.xlsx"
]

def load_delivery_data():
    print("正在加载分娩日期记录字典...")
    dfs = []
    
    # 文件1: csv
    try:
        df1 = pd.read_csv(DELIVERY_FILES[0], usecols=['项目流水号', '分娩时间'], low_memory=False)
        dfs.append(df1)
    except Exception as e:
        print(f"警告: 无法加载 {DELIVERY_FILES[0]}: {e}")
        
    # 文件2和3: xlsx
    for f in DELIVERY_FILES[1:]:
        try:
            dx = pd.read_excel(f, usecols=['项目流水号', '分娩时间'])
            dfs.append(dx)
        except Exception as e:
            print(f"警告: 无法加载 {f}: {e}")
            
    if not dfs:
        return {}
        
    delivery_df = pd.concat(dfs, ignore_index=True)
    delivery_df['项目流水号'] = delivery_df['项目流水号'].astype(str).str.strip()
    delivery_df['分娩时间'] = pd.to_datetime(delivery_df['分娩时间'], errors='coerce')
    delivery_df = delivery_df.dropna(subset=['分娩时间'])
    
    # 按照流水号去重，或者取第一条
    # 若有多次记录，取最大时间（保守起见，避免早产误判干扰前期数据）或者直接first
    delivery_map = delivery_df.sort_values(by='分娩时间').groupby('项目流水号')['分娩时间'].last().to_dict()
    print(f"成功构建分娩记录字典，共计包含 {len(delivery_map)} 个唯一病历。")
    return delivery_map

def plot_repair(nid, days, w_raw, pp_days, pp_weights, logs):
    plt.figure(figsize=(10, 6))
    plt.plot(days, w_raw, color='blue', linestyle='-', marker='o', alpha=0.6, label='当前体重')
    
    if len(pp_days) > 0: plt.scatter(pp_days, pp_weights, color='green', s=120, zorder=5, marker='*', label='产后断崖(锁定)')
    plt.title(f"ID: {nid} | 产后断崖免修锁定")
    plt.xlabel("孕周(天)")
    plt.ylabel("体重 (kg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    info_text = "\n".join(logs)
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{nid}_Postpartum.png"), dpi=100)
    plt.close()

def mark_postpartum_drops(group, nid, delivery_map):
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    valid_mask = ~pd.isna(w_orig)
    
    v_idx = np.where(valid_mask)[0]
    n_v = len(v_idx)
    logs, marked = [], False
    
    is_pp = np.zeros(len(w_orig), dtype=bool)
    
    # 结合分娩日期计算精确的阈值门槛
    delivery_date = delivery_map.get(str(nid), pd.NaT)
    delivery_gday = None
    
    if pd.notna(delivery_date):
        lmp_series = group['LMP'].dropna()
        if len(lmp_series) > 0:
            lmp = pd.to_datetime(lmp_series.iloc[0], errors='coerce')
            if pd.notna(lmp):
                delivery_gday = (delivery_date - lmp).days
                
    if delivery_gday is not None:
        threshold_day = delivery_gday - 7 # 允许一周的误差范围
        cond_str = f"分娩孕周 {delivery_gday}d"
    else:
        threshold_day = 270 # 如果没有分娩记录作兜底
        cond_str = "无分娩记录兜底 >270d"
    
    if n_v >= 2:
        for i in range(1, n_v):
            curr_idx = v_idx[i]
            prev_idx = v_idx[i-1]
            curr_w = w_orig[curr_idx]
            prev_w = w_orig[prev_idx]
            curr_day = days[curr_idx]
            prev_day = days[prev_idx]
            
            # 跌幅 > 4.5kg 且在临产期后
            if curr_day >= threshold_day and (prev_w - curr_w) > 4.5 and (prev_w - curr_w) < 22:
                for j in range(i, n_v):
                    is_pp[v_idx[j]] = True
                
                marked = True
                logs.append(f"Day {curr_day}d: 锁定产后断崖 ({cond_str}) | {prev_w:.1f} -> {curr_w:.1f} (下降 {prev_w-curr_w:.1f}kg)")
                break 
                
    group['is_postpartum_normal'] = is_pp
    group['weight_cleaned'] = w_orig 
    return group, logs, marked

def main():
    if not os.path.exists(INPUT_CSV): return
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    if id_col not in df.columns: df.rename(columns={df.columns[0]: id_col}, inplace=True)
    df[id_col] = df[id_col].astype(str).str.strip()
    
    delivery_map = load_delivery_data()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    frames, all_logs, marked_count = [], [], 0
    total = len(grouped)
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0: print(f"06处理进度: {i}/{total}...")
        c_group, logs, marked = mark_postpartum_drops(group.copy(), nid, delivery_map)
        frames.append(c_group)
        if marked:
            marked_count += 1
            all_logs.append(f"[{nid}] 锁定免修")
            all_logs.extend(["  " + l for l in logs])
            
            mask = c_group['is_postpartum_normal'].values
            plot_repair(nid, c_group['gestation_day'].values, c_group['weight'].values,
                        c_group['gestation_day'].values[mask], c_group['weight'].values[mask], logs)
            
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(columns={'weight': 'weight_raw_p6', 'weight_cleaned': 'weight'}, inplace=True)
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_logs))
    print(f"STEP 6 完成！共扫描打标护航由于分娩正常的体重断崖: {marked_count} 例。")

if __name__ == '__main__':
    main()
