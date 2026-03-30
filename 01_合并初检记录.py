import pandas as pd
import numpy as np
import os

print("=== [Pipeline Step 1] 合并初检记录 ===")

# ================= 配置路径 =================
INIT_FILES = [
    r"孕妇初检-第五批.xlsx",
    r"孕期初检-2017年-2023年xlsx"
]
HIS_PATH = r"宝安_HIS前处理_合并表.csv"

OUT_DIR = r"gestational-weight-records-cleaning-pipline"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "01_合并后底表_带初检.csv")

def main():
    print("正在加载初始数据...")
    init_df = pd.concat([pd.read_excel(f) for f in INIT_FILES], ignore_index=True)
    init_df['项目流水号'] = init_df['项目流水号'].astype(str).str.strip()
    init_df = init_df.drop_duplicates('项目流水号')

    seeds_list = []
    for _, row in init_df.iterrows():
        nid = row['项目流水号']
        pre_w = row['孕前体重']
        if pd.isna(pre_w): continue
        try:
            pre_w = float(pre_w)
        except: continue

        seeds_list.append({
            '项目流水号': nid,
            'weight': pre_w,
            'height': row['身高'],
            'BMI': row['BMI'],
            'SBP': row['收缩压'],
            'DBP': row['舒张压'],
            'gestation_day': 0,
            'type': 'Initial_Raw'
        })

    seeds_df = pd.DataFrame(seeds_list)
    print(f"成功汲取可用孕前体重点数: {len(seeds_df)}")

    sys_df = pd.read_csv(HIS_PATH, low_memory=False)
    sys_df['项目流水号'] = sys_df['项目流水号'].astype(str).str.strip()

    # 防止 day=0 双重录入，剔除旧 HIS 表中的 day=0 如果该产妇在初检表有值
    sys_df = sys_df[~((sys_df['项目流水号'].isin(seeds_df['项目流水号'])) & (sys_df['gestation_day'] == 0))]

    print("执行合并拼接...")
    for col in sys_df.columns:
        if col not in seeds_df.columns:
            seeds_df[col] = np.nan

    final = pd.concat([sys_df, seeds_df], ignore_index=True).sort_values(['项目流水号', 'gestation_day'])

    final.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"STEP 1 完成！输出文件 -> {OUT_PATH}\n")

if __name__ == '__main__':
    main()
