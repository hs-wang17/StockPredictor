import os
import pandas as pd
from tqdm import tqdm

# 原始标签（10日）
factor_with_label_dir = "/home/haris/raid0/shared/haris/mydata_20251231/concat_daily_factor_with_label_10"
save_dir = "/home/haris/raid0/shared/haris/mydata_20251231/concat_daily_factor_with_label_10_index"
os.makedirs(save_dir, exist_ok=True)
index_weight_df = pd.read_feather("/home/haris/raid0/shared/haris/mydata_20251231/index_weight.fea")
factor_with_label_files = sorted(os.listdir(factor_with_label_dir))
for factor_with_label_file in tqdm(factor_with_label_files[:]):  # 只计算最后一天
    date = factor_with_label_file[:8]
    factor_with_label = pd.read_feather(os.path.join(factor_with_label_dir, factor_with_label_file))
    index_weight = index_weight_df.loc[date].dropna().to_frame(name="index_weight")
    index_weight.columns = ["index_weight"]
    common_index = factor_with_label.index.intersection(index_weight.index)
    factor_with_label_aligned = factor_with_label.loc[common_index]
    index_weight_aligned = index_weight.loc[common_index]
    factor_with_label_index_weight = pd.concat([factor_with_label_aligned, index_weight_aligned], axis=1)
    factor_with_label_index_weight.to_feather(os.path.join(save_dir, factor_with_label_file))

print("Done")
