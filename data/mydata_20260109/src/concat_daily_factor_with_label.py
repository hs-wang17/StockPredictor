import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 原始标签（20日）
factor_dir = "/home/haris/mydata_20260109/concat_daily_factor"
save_dir = "/home/haris/mydata_20260109/concat_daily_factor_with_label"
os.makedirs(save_dir, exist_ok=True)
label_df = pd.read_feather("/home/haris/mydata_20260109/label.fea")
factor_files = sorted(os.listdir(factor_dir))[:-20]
for factor_file in tqdm(factor_files):
    date = factor_file[:8]
    factor = pd.read_feather(os.path.join(factor_dir, factor_file))
    label = label_df.loc[date].dropna().to_frame(name="label")
    label.columns = ["label"]
    common_index = factor.index.intersection(label.index)
    factor_aligned = factor.loc[common_index]
    label_aligned = label.loc[common_index]
    factor_with_label = pd.concat([factor_aligned, label_aligned], axis=1)
    factor_with_label.to_feather(os.path.join(save_dir, factor_file))

factor_dir = "/home/haris/mydata_20260109/concat_daily_factor"
save_dir = "/home/haris/mydata_20260109/concat_daily_factor_with_label_dummy"
os.makedirs(save_dir, exist_ok=True)
label_df = pd.read_feather("/home/haris/mydata_20260109/label_dummy.fea")
factor_files = sorted(os.listdir(factor_dir))[:-20]
for factor_file in tqdm(factor_files):
    date = factor_file[:8]
    factor = pd.read_feather(os.path.join(factor_dir, factor_file))
    label = label_df.loc[date].dropna().to_frame(name="label")
    label.columns = ["label"]
    common_index = factor.index.intersection(label.index)
    factor_aligned = factor.loc[common_index]
    label_aligned = label.loc[common_index]
    factor_with_label = pd.concat([factor_aligned, label_aligned], axis=1)
    factor_with_label.to_feather(os.path.join(save_dir, factor_file))

factor_dir = "/home/haris/mydata_20260109/concat_daily_factor"
save_dir = "/home/haris/mydata_20260109/concat_daily_factor_with_label_beta"
os.makedirs(save_dir, exist_ok=True)
label_df = pd.read_feather("/home/haris/mydata_20260109/label_beta.fea")
factor_files = sorted(os.listdir(factor_dir))[:-20]
for factor_file in tqdm(factor_files):
    date = factor_file[:8]
    factor = pd.read_feather(os.path.join(factor_dir, factor_file))
    label = label_df.loc[date].dropna().to_frame(name="label")
    label.columns = ["label"]
    common_index = factor.index.intersection(label.index)
    factor_aligned = factor.loc[common_index]
    label_aligned = label.loc[common_index]
    factor_with_label = pd.concat([factor_aligned, label_aligned], axis=1)
    factor_with_label.to_feather(os.path.join(save_dir, factor_file))

# 新标签1（3日、5日、10日、20日等权重计算均值）
factor_dir = "/home/haris/mydata_20260109/concat_daily_factor"
save_dir = "/home/haris/mydata_20260109/concat_daily_factor_with_label_mix"
os.makedirs(save_dir, exist_ok=True)
label_df = pd.read_feather("/home/haris/mydata_20260109/label_mix.fea")
factor_files = sorted(os.listdir(factor_dir))[:-20]
for factor_file in tqdm(factor_files):
    date = factor_file[:8]
    factor = pd.read_feather(os.path.join(factor_dir, factor_file))
    label = label_df.loc[date].dropna().to_frame(name="label")
    label.columns = ["label"]
    common_index = factor.index.intersection(label.index)
    factor_aligned = factor.loc[common_index]
    label_aligned = label.loc[common_index]
    factor_with_label = pd.concat([factor_aligned, label_aligned], axis=1)
    factor_with_label.to_feather(os.path.join(save_dir, factor_file))

# 新标签2（3日、5日、10日、20日多标签向量）
factor_dir = "/home/haris/mydata_20260109/concat_daily_factor"
save_dir = "/home/haris/mydata_20260109/concat_daily_factor_with_label_vector"
os.makedirs(save_dir, exist_ok=True)
label_df = pd.read_feather("/home/haris/mydata_20260109/label_vector.fea")
factor_files = sorted(os.listdir(factor_dir))[:-20]
for factor_file in tqdm(factor_files):
    date = factor_file[:8]
    factor = pd.read_feather(os.path.join(factor_dir, factor_file))
    # label = label_df.loc[date].dropna().to_frame(name="label")
    mask = label_df.loc[date].apply(lambda x: isinstance(x, (list, np.ndarray)) and not np.isnan(x).any())
    label = label_df.loc[date][mask].to_frame(name="label")
    label.columns = ["label"]
    common_index = factor.index.intersection(label.index)
    factor_aligned = factor.loc[common_index]
    label_aligned = label.loc[common_index]
    factor_with_label = pd.concat([factor_aligned, label_aligned], axis=1)
    factor_with_label.to_feather(os.path.join(save_dir, factor_file))

print("Done")
