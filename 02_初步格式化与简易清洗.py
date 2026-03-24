import pandas as pd
import numpy as np
import os
import math
from datetime import datetime

print("=== [Pipeline Step 2] 初步格式化与简易错漏清洗 ===")

INPUT_CSV = r"E:\文件\研究生\项目\肥胖分布统计\宝安\市妇幼系统\清洗\01_合并后底表_带初检.csv"
OUT_DIR = r"E:\文件\研究生\项目\肥胖分布统计\宝安\市妇幼系统\清洗"
OUT_CSV = os.path.join(OUT_DIR, "02_初步清洗_去低级失误版.csv")
LOG_FILE = os.path.join(OUT_DIR, "初步清洗_日志.txt")

def simple_clean(group):
    # 处理单一病人的纯量级错漏
    w_orig = group['weight'].values.copy()
    days = group['gestation_day'].values
    n = len(w_orig)
    logs = []
    changed = False

    for i in range(n):
        curr_w = w_orig[i]
        if pd.isna(curr_w): continue
        
        # 寻找最近邻锚点
        anchors = []
        for j in range(i-1, -1, -1):
            if not pd.isna(w_orig[j]): anchors.append(w_orig[j]); break
        for j in range(i+1, n):
            if not pd.isna(w_orig[j]): anchors.append(w_orig[j]); break
            
        if not anchors: continue 
        ref_w = np.mean(anchors)
        
        # 1. 键盘敲多一个 0 溢出 (例如 500)
        if curr_w >= 280:
            cw_div10 = curr_w / 10.0
            if abs(cw_div10 - ref_w) < 25:
                w_orig[i] = round(cw_div10, 2)
                logs.append(f"Day {days[i]}d: 多0溢出 | 原值 {curr_w} -> {w_orig[i]} (邻居参考 {ref_w:.1f})")
                changed = True
                continue

        # 2. 漏敲十位 (例如 6.5)
        # 严控阈值：仅当体重低于 25kg 时才判定为缺漏十位，防止 34kg 等真实低体重人群被周围未清洗的异常锚点（如74kg的斤单位）误导
        if curr_w < 25 and ref_w >= 40:
            c_mod = curr_w % 10
            ref_tens = (int(ref_w) // 10) * 10
            cands = [ref_tens + c_mod, (ref_tens - 10) + c_mod, (ref_tens + 10) + c_mod]
            best_c = min(cands, key=lambda x: abs(x - ref_w))
            # 只有当拼补出来的数字与邻居极其接近（误差控制在8kg内），才确认为漏敲十位
            if abs(best_c - ref_w) < 8:
                w_orig[i] = round(best_c, 2)
                logs.append(f"Day {days[i]}d: 丢十位数 | 原值 {curr_w} -> {w_orig[i]} (邻居参考 {ref_w:.1f})")
                changed = True
                continue

    group['weight_cleaned'] = w_orig
    return group, logs, changed

def main():
    print(f"正在加载合并底表: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    id_col = '项目流水号'
    df[id_col] = df[id_col].astype(str).str.strip()
    
    grouped = df.sort_values([id_col, 'gestation_day']).groupby(id_col)
    
    frames = []
    all_logs = []
    total = len(grouped)
    changed_count = 0
    
    for i, (nid, group) in enumerate(grouped):
        if i % 10000 == 0: print(f"进度: {i}/{total}...")
        c_group, logs, changed = simple_clean(group)
        frames.append(c_group)
        if changed:
            changed_count += 1
            all_logs.append(f"[{nid}] 修正明细:")
            all_logs.extend(["  " + l for l in logs])
            
    final_df = pd.concat(frames, ignore_index=True)
    # 用清理后的体重覆盖原值
    final_df.rename(columns={'weight': 'weight_raw_p1', 'weight_cleaned': 'weight'}, inplace=True)
    
    final_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"初步基础洗涤日志 ({datetime.now()})\n")
        f.write("="*60 + "\n")
        f.write("\n".join(all_logs))
        
    print(f"STEP 2 完成！共修复简单遗漏产妇: {changed_count} 例。文件输出至 -> {OUT_CSV}\n")

if __name__ == '__main__':
    main()
