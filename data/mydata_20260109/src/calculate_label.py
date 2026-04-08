import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import neutralization as neu

vwap = pd.read_feather("/home/haris/project/backtester/data/vwap.fea")
adjfactor = pd.read_feather("/home/haris/data/data_frames/stk_adjfactor.feather")
adjfactor = adjfactor.reindex(index=vwap.index, columns=vwap.columns)
vwap = vwap * adjfactor

# 原始标签（20日）
# idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
# idx_ret_20 = idx_open.pct_change(20).shift(-21).dropna(how="all")["中证1000"].squeeze()
# stk_ret_20 = vwap.pct_change(20).shift(-21).dropna(how="all")
# label_20 = stk_ret_20.sub(idx_ret_20, axis=0).dropna(how="all")
# high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
# open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
# zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
# st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
# stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
# zt_st_stop_df = zt_df | st_df | stop_df
# mask = zt_st_stop_df.reindex(index=label_20.index, columns=label_20.columns)
# label_20 = label_20.mask(mask == 1).dropna(how="all")
# label_20.to_feather("/home/haris/project/backtester/data/label.fea")
# label_20.to_feather("/home/haris/mydata_20260109/label.fea")
# trade_date_df = pd.DataFrame({"trade_date": label_20.index})
# trade_date_df.reset_index(drop=True).to_feather("/home/haris/mydata_20260109/trade_date.fea")

idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
idx_ret_20 = idx_open.pct_change(20).shift(-21).dropna(how="all")["中证1000"].squeeze()
stk_ret_20 = vwap.pct_change(20).shift(-21).dropna(how="all")
label_20 = stk_ret_20.sub(idx_ret_20, axis=0).dropna(how="all")
high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label_20.index, columns=label_20.columns)
label_20 = label_20.mask(mask == 1).dropna(how="all")
label_20 = neu.neutralize_label_dummy(label_20)
label_20.to_feather("/home/haris/project/backtester/data/label_dummy.fea")
label_20.to_feather("/home/haris/mydata_20260109/label_dummy.fea")
trade_date_df = pd.DataFrame({"trade_date": label_20.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/mydata_20260109/trade_date_dummy.fea")

idx_open = pd.read_feather("/home/haris/data/data_frames/idx_open.feather")
idx_ret_20 = idx_open.pct_change(20).shift(-21).dropna(how="all")["中证1000"].squeeze()
stk_ret_20 = vwap.pct_change(20).shift(-21).dropna(how="all")
label_20 = stk_ret_20.sub(idx_ret_20, axis=0).dropna(how="all")
high_limit = pd.read_feather("/home/haris/data/data_frames/stk_ztprice.feather").replace(0, np.nan).ffill()
open = pd.read_feather("/home/haris/data/data_frames/stk_open.feather").replace(0, np.nan).ffill()
zt_df = (open == high_limit).shift(-1).fillna(False).astype(int)
st_df = pd.read_feather("/home/haris/data/data_frames/stk_is_st_stock.feather").shift(-1).fillna(False).astype(int)
stop_df = pd.read_feather("/home/haris/data/data_frames/stk_is_stop_stock.feather").shift(-1).fillna(False).astype(int)
zt_st_stop_df = zt_df | st_df | stop_df
mask = zt_st_stop_df.reindex(index=label_20.index, columns=label_20.columns)
label_20 = label_20.mask(mask == 1).dropna(how="all")
label_20 = neu.neutralize_label_beta(label_20)
label_20.to_feather("/home/haris/project/backtester/data/label_beta.fea")
label_20.to_feather("/home/haris/mydata_20260109/label_beta.fea")
trade_date_df = pd.DataFrame({"trade_date": label_20.index})
trade_date_df.reset_index(drop=True).to_feather("/home/haris/mydata_20260109/trade_date_beta.fea")

# 新标签1（3日、5日、10日、20日等权重计算均值）
# label_20 = stk_ret_20.sub(idx_ret_20, axis=0).dropna(how="all") / 20.0
# idx_ret_10 = idx_open.pct_change(10).shift(-11).dropna(how="all")["中证1000"].squeeze()
# stk_ret_10 = vwap.pct_change(10).shift(-11).dropna(how="all")
# label_10 = stk_ret_10.sub(idx_ret_10, axis=0).dropna(how="all") / 10.0
# idx_ret_5 = idx_open.pct_change(5).shift(-6).dropna(how="all")["中证1000"].squeeze()
# stk_ret_5 = vwap.pct_change(5).shift(-6).dropna(how="all")
# label_5 = stk_ret_5.sub(idx_ret_5, axis=0).dropna(how="all") / 5.0
# idx_ret_3 = idx_open.pct_change(3).shift(-4).dropna(how="all")["中证1000"].squeeze()
# stk_ret_3 = vwap.pct_change(3).shift(-4).dropna(how="all")
# label_3 = stk_ret_3.sub(idx_ret_3, axis=0).dropna(how="all") / 3.0
# label_mix = (label_20 + label_10 + label_5 + label_3) / 4.0
# mask_mix = zt_st_stop_df.reindex(index=label_mix.index, columns=label_mix.columns)
# label_mix = label_mix.mask(mask == 1).dropna(how="all")
# label_mix.to_feather("/home/haris/project/backtester/data/label_mix.fea")
# label_mix.to_feather("/home/haris/mydata_20260109/label_mix.fea")
# trade_date_df_mix = pd.DataFrame({"trade_date": label_mix.index})
# trade_date_df_mix.reset_index(drop=True).to_feather("/home/haris/mydata_20260109/trade_date_mix.fea")

# # 新标签2（3日、5日、10日、20日多标签向量）
# label_list = [label_3, label_5, label_10, label_20]
# common_index = label_list[0].index
# common_columns = label_list[0].columns
# for df in label_list[1:]:
#     common_index = common_index.intersection(df.index)
#     common_columns = common_columns.intersection(df.columns)
# label_list = [df.reindex(index=common_index, columns=common_columns) for df in label_list]
# mask_vector = zt_st_stop_df.reindex(index=common_index, columns=common_columns)
# label_list = [df.mask(mask_vector == 1).dropna(how="all") for df in label_list]
# label_stack = np.stack([df.values for df in label_list], axis=-1)
# label_vector = pd.DataFrame(index=common_index, columns=common_columns, dtype=object)
# for i in trange(label_stack.shape[0]):
#     for j in range(label_stack.shape[1]):
#         label_vector.iat[i, j] = label_stack[i, j, :]
# label_vector = label_vector.dropna(how="all", axis=0).dropna(how="all", axis=1)
# label_vector.to_feather("/home/haris/project/backtester/data/label_vector.fea")
# label_vector.to_feather("/home/haris/mydata_20260109/label_vector.fea")
# trade_date_df_vector = pd.DataFrame({"trade_date": label_vector.index})
# trade_date_df_vector.reset_index(drop=True).to_feather("/home/haris/mydata_20260109/trade_date_vector.fea")


# print("Done")
