import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=== [Pipeline Step 6] 产后断崖锁定 (结合分娩记录打标免修) ===")

INPUT_CSV = r"HIS系统\清洗流程_v2\05_局部尖峰处理版.csv"
OUT_DIR = r"HIS系统\清洗流程_v2"
OUT_CSV = os.path.join(OUT_DIR, "06_产后断崖锁定版.csv")
LOG_FILE = os.path.join(OUT_DIR, "06_产后断崖锁定_日志.txt")
PLOT_DIR = os.path.join(OUT_DIR, "06_Plots_产后免修锁定")
os.makedirs(PLOT_DIR, exist_ok=True)

DELIVERY_FILES = [
    r"E:\文件\研究生\项目\宝安妇幼数据搜索\清洗任务\Baoan合并校验\Baoan分娩记录-第1-5批-清洗地址后-icd11_mapped-20260410.xlsx",
]

import re

def parse_gestational_week(s):
    """解析 '40 周 1 天'、'37周3天' 等格式为总天数。无法解析或超出生理范围则返回 None。"""
    if pd.isna(s):
        return None
    s = str(s).strip()
    w_m = re.search(r'(\d+)\s*周', s)
    d_m = re.search(r'(\d+)\s*天', s)
    if w_m is None:
        return None
    weeks = int(w_m.group(1))
    extra = int(d_m.group(1)) if d_m else 0
    total = weeks * 7 + extra
    # 生理范围：20 周(140d) ~ 45 周(315d)
    return total if 140 <= total <= 315 else None


def _read_delivery_file(read_func, path, base_cols, **kwargs):
    """尝试带 '孕周' 列读取；列不存在则补 NaN。"""
    try:
        return read_func(path, usecols=base_cols + ['孕周'], **kwargs)
    except (ValueError, KeyError):
        df = read_func(path, usecols=base_cols, **kwargs)
        df['孕周'] = np.nan
        return df


def load_delivery_data():
    print("正在加载分娩记录（含孕周列）...")
    dfs = []

    # 根据扩展名自动选择读取函数，不依赖 DELIVERY_FILES 的顺序
    for f in DELIVERY_FILES:
        try:
            ext = os.path.splitext(f)[1].lower()
            if ext == '.csv':
                read_func = pd.read_csv
                kwargs = {'low_memory': False}
            else:  # .xlsx / .xls
                read_func = pd.read_excel
                kwargs = {}
            dfs.append(_read_delivery_file(read_func, f, ['项目流水号', '分娩时间'], **kwargs))
        except Exception as e:
            print(f"警告: 无法加载 {f}: {e}")

    if not dfs:
        return {}, {}

    delivery_df = pd.concat(dfs, ignore_index=True)
    delivery_df['项目流水号'] = delivery_df['项目流水号'].astype(str).str.strip()

    # ---- gweek_map：从全量记录构建，不要求分娩时间存在 ----
    # 只要 孕周 列可解析即纳入，覆盖尽可能多的病历
    delivery_df['_gdays'] = delivery_df['孕周'].apply(parse_gestational_week)
    gweek_all = delivery_df.dropna(subset=['_gdays']).copy()
    # 同一病历有多条时，取孕周天数最大的那条（最接近足月）
    gweek_map = (
        gweek_all.sort_values('_gdays')
        .groupby('项目流水号')['_gdays'].last()
        .astype(int).to_dict()
    ) if not gweek_all.empty else {}

    # ---- delivery_map：LMP推算兜底用，需要有效分娩时间 ----
    delivery_df['分娩时间'] = pd.to_datetime(delivery_df['分娩时间'], errors='coerce')
    delivery_valid = delivery_df.dropna(subset=['分娩时间'])
    delivery_map = (
        delivery_valid.sort_values('分娩时间')
        .groupby('项目流水号')['分娩时间'].last().to_dict()
    ) if not delivery_valid.empty else {}

    print(f"成功构建 gweek_map（孕周档案）: 覆盖 {len(gweek_map)} 个病历，"
          f"含有效孕周记录 {len(gweek_all)} 条。")
    print(f"成功构建 delivery_map（LMP推算兜底）: 覆盖 {len(delivery_map)} 个病历。")
    return delivery_map, gweek_map

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

def mark_postpartum_drops(group, nid, delivery_map, gweek_map):
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    valid_mask = ~pd.isna(w_orig)
    
    v_idx = np.where(valid_mask)[0]
    n_v = len(v_idx)
    v_w = w_orig[v_idx]
    v_days = days[v_idx]
    logs, marked = [], False
    
    is_pp = np.zeros(len(w_orig), dtype=bool)
    
    # ---- 确定分娩孕周天数（delivery_gday）----
    # 优先 1：分娩档案直接记录的"孕周"列（最可靠）
    delivery_gday = None
    cond_str = "无分娩记录兜底 >270d"

    gweek_days = gweek_map.get(str(nid))
    if gweek_days is not None:
        delivery_gday = gweek_days
        cond_str = f"孕周档案 {gweek_days // 7}w{gweek_days % 7}d={gweek_days}d"

    # 优先 2：分娩时间 + LMP 推算（孕周列缺失时备选）
    if delivery_gday is None:
        delivery_date = delivery_map.get(str(nid), pd.NaT)
        if pd.notna(delivery_date):
            lmp_series = group['LMP'].dropna()
            if len(lmp_series) > 0:
                lmp = pd.to_datetime(lmp_series.iloc[0], errors='coerce')
                if pd.notna(lmp):
                    calc = (delivery_date - lmp).days
                    if 140 <= calc <= 315:
                        delivery_gday = calc
                        cond_str = f"LMP推算 {calc}d"

    if delivery_gday is not None:
        threshold_day = delivery_gday - 7  # 允许一周误差
    else:
        threshold_day = 270
    
    # 获取孕前基准体重 (W0)，取前14天内的最早记录
    W0 = None
    if n_v > 0:
        early_records = [v_w_val for i_v, v_w_val in enumerate(v_w) if v_days[i_v] <= 14]
        if early_records:
            W0 = early_records[0]

    if n_v >= 2:
        # 有明确分娩记录时，不约束跌幅上限（分娩后任何跌幅都是生理正常的）
        # 无分娩记录兜底时，适当设一个宽松上限防止误标
        drop_upper = None if delivery_gday is not None else 35.0
        
        for i in range(1, n_v):
            curr_idx = v_idx[i]
            prev_idx = v_idx[i-1]
            curr_w = w_orig[curr_idx]
            prev_w = w_orig[prev_idx]
            curr_day = days[curr_idx]
            prev_day = days[prev_idx]
            
            drop = prev_w - curr_w
            # 跌幅 > 4.5kg 且在临产期后
            if curr_day >= threshold_day and drop > 4.5:
                # 极端异常校验：防止分娩大出血或严重的系统录入错误
                # 如果跌幅超过 25kg 且最终体重比孕前还低 15kg 以上，则认定为记录错误而非正常分娩
                if W0 is not None and drop > 25.0 and curr_w < (W0 - 20.0):
                    continue # 认为是错误记录，不打免修标签，留给后续 07 脚本删除
                
                if drop_upper is not None and drop > drop_upper:
                    continue  # 无分娩记录兜底时，跌幅过大可能不是分娩
                    
                for j in range(i, n_v):
                    is_pp[v_idx[j]] = True
                
                marked = True
                logs.append(f"Day {curr_day}d: 锁定产后断崖 ({cond_str}) | {prev_w:.1f} -> {curr_w:.1f} (下降 {drop:.1f}kg)")
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
    
    delivery_map, gweek_map = load_delivery_data()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    frames, all_logs, marked_count = [], [], 0
    total = len(grouped)
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0: print(f"06处理进度: {i}/{total}...")
        c_group, logs, marked = mark_postpartum_drops(group.copy(), nid, delivery_map, gweek_map)
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
