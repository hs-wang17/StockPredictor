import numpy as np
import pandas as pd
import os
from tqdm import tqdm

data_dir = "/home/haris/data/DailyFactors/day2day"
save_dir_copy = "/home/haris/raid0/shared/haris/mydata_20251231/concat_day2day_factor"
os.makedirs(save_dir_copy, exist_ok=True)
factor_types = [d for d in os.listdir(data_dir) if d != "consensus" and d != "finance2"]
date_list = []
for factor_type in factor_types:
    factor_type_dir = os.path.join(data_dir, factor_type)
    dates = os.listdir(factor_type_dir)
    date_list.extend(dates)
date_list = sorted(list(set(date_list)))

for date in tqdm([date for date in date_list if date >= "20181001"][-1:]):  # 只计算最后一天
    df_list = []
    for factor_type in factor_types:
        factor_path = os.path.join(data_dir, factor_type, date)
        df = pd.read_feather(factor_path)
        if "index" in df.columns:
            df = df.set_index("index")
        elif "symbol" in df.columns:
            df = df.set_index("symbol")
        df_list.append(df)
    if df_list:
        concat_df = pd.concat(df_list, axis=1)
        concat_df.columns = [f"factor_{str(i).zfill(3)}" for i in np.arange(len(concat_df.columns))]
        save_path_copy = os.path.join(save_dir_copy, f"{date}")
        concat_df.to_feather(save_path_copy)

print("Done")
