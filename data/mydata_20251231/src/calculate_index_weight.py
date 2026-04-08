import numpy as np
import pandas as pd
import os
from tqdm import tqdm

os.chdir(r"/mnt/raid0/nfs_from6_readonly/trade_support_data/trade_support7")
file_list = [file for file in sorted(os.listdir()) if file >= "20180101.fea"]

index_weight_list = []
for file in tqdm(file_list[:]):
    date = file.split(".")[0]
    index_member = pd.read_feather(date + ".fea")[["hs300_member", "zz500_member", "zz1000_member", "zz2000_member"]]
    index_weight = pd.Series(0.0, index=index_member.index)
    index_weight[index_member["hs300_member"] > 0.0] = 1.5
    index_weight[index_member["zz500_member"] > 0.0] = 1.5
    index_weight[index_member["zz1000_member"] > 0.0] = 1.5
    index_weight[index_member["zz2000_member"] > 0.0] = 1.0
    index_weight[
        (index_member["hs300_member"] == 0.0)
        & (index_member["zz500_member"] == 0.0)
        & (index_member["zz1000_member"] == 0.0)
        & (index_member["zz2000_member"] == 0.0)
    ] = 0.5
    index_weight_list.append((date, index_weight))

index_weight_df = pd.DataFrame({date: weight for date, weight in index_weight_list}).T
index_weight_df.to_feather("/home/haris/raid0/shared/haris/mydata_20251231/index_weight.fea")

print("Done")
