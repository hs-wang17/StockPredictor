import numpy as np
import pandas as pd
import os
from tqdm import tqdm

data_dirs = [
    "/home/haris/raid0/shared/haris/mydata_20251231/concat_daily_factor",
    "/home/haris/data/IntraDayFactors/1130/min2day/min_fac1",
    "/home/haris/data/IntraDayFactors/1130/order2day",
    "/home/haris/data/IntraDayFactors/1130/ordertrans2day/ordertrans_fac1",
    # "/home/haris/data/IntraDayFactors/1130/ordertrans2day/ordertrans_fac2",
    "/home/haris/data/IntraDayFactors/1130/tick2day",
    "/home/haris/data/IntraDayFactors/1130/trans2day",
]
save_dir = "/home/haris/raid0/shared/haris/mydata_20260127/concat_daily_factor"
os.makedirs(save_dir, exist_ok=True)

date_set = set()

for d in data_dirs:
    if os.path.isdir(d):
        date_set.update(os.listdir(d))

date_list = sorted(date_set)

for idx_date in tqdm(range(1, len(date_list))[-1:]):  # 只计算最后一天
    date = date_list[idx_date]
    last_date = date_list[idx_date - 1]
    df_list = []

    for d in data_dirs[:1]:
        path = os.path.join(d, last_date)
        df = pd.read_feather(path)
        df_list.append(df)

    for d in data_dirs[1:]:
        path = os.path.join(d, date)
        df = pd.read_feather(path)
        if "code" in df.columns:
            df = df.set_index("code")
        df_list.append(df)

    concat_df = pd.concat(df_list, axis=1)
    concat_df.columns = [f"factor_{str(i).zfill(4)}" for i in range(concat_df.shape[1])]
    concat_df.to_feather(os.path.join(save_dir, date))

trade_dates = sorted(set([d[:8] for d in date_list]))
trade_date_df = pd.DataFrame({"trade_date": trade_dates[:-21]})
trade_date_df.to_feather("/home/haris/raid0/shared/haris/mydata_20260127/trade_date.fea")

print("Done")
