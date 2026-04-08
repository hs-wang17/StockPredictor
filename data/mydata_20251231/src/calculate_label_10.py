import numpy as np
import pandas as pd

vwap = pd.read_feather("/home/haris/data/trade_support_data/vwap.fea")
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=vwap.index, columns=vwap.columns)
vwap = vwap * adjfactor

# 原始标签（10日）
idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
idx_ret_10 = idx_open.pct_change(10).shift(-11).dropna(how="all")["中证1000"].squeeze()
stk_ret_10 = vwap.pct_change(10).shift(-11).dropna(how="all")
label_10 = stk_ret_10.sub(idx_ret_10, axis=0).dropna(how="all")

high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label_10.index, columns=label_10.columns)
label_10 = label_10.mask(mask == 1).dropna(how="all")
# label_10.to_feather("/home/haris/project/backtester/data/label_10.fea")
label_10.to_feather("/home/haris/raid0/shared/haris/mydata_20251231/label_10.fea")
trade_date_df = pd.DataFrame({"trade_date": label_10.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/raid0/shared/haris/mydata_20251231/trade_date_10.fea")

print("Done")
