import pandas as pd
import os
from tqdm import tqdm

data_dirs = [
    "/home/haris/raid0/shared/haris/mydata_20251231/concat_day2day_factor",
    "/home/haris/data/DailyFactors/min2day2/min_fac2",
    "/home/haris/data/DailyFactors/order2day",
    "/home/haris/data/DailyFactors/ordertrans2day",
    "/home/haris/data/DailyFactors/tick2day",
    "/home/haris/data/DailyFactors/trans2day",
]
save_dir_copy = "/home/haris/raid0/shared/haris/mydata_20251231/concat_daily_factor"

os.makedirs(save_dir_copy, exist_ok=True)
date_set = set()
for d in data_dirs:
    if os.path.isdir(d):
        date_set.update(os.listdir(d))
date_list = sorted(date_set)[1:]
for date in tqdm(date_list[-1:]):  # 只计算最后一天
    df_list = []
    for d in data_dirs:
        path = os.path.join(d, date)
        if os.path.exists(path):
            try:
                df = pd.read_feather(path)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    if df_list:
        concat_df = pd.concat(df_list, axis=1)
        concat_df.columns = [f"factor_{str(i).zfill(4)}" for i in range(concat_df.shape[1])]
        concat_df.to_feather(os.path.join(save_dir_copy, date))

print("Done")
