import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")


# 读取数据
def get_eval_date(all_date, season, month=24, select_start=None):
    """
    获取指定季度的训练日期范围
    Args:
        all_date: 所有交易日列表
        season: 季度字符串，如"2022q4"
        month: 训练集回溯月数，默认24个月
        select_start: 可选的训练开始日期（覆盖month参数）
    Returns:
        (开始日期, 结束日期)
    Example:
        season="2022q4", month=24
        训练集：2020-10-01 至 2022-10-01（前6个交易日）
    """
    # 解析季度，计算季度开始日期
    year, q = season.split("q")
    month_num = int(q) * 3 - 2
    test_start = f"{year}{month_num:02d}01"
    # 计算训练开始日期
    if select_start:
        train_start = select_start
    else:
        start_date = datetime.strptime(test_start, "%Y%m%d")
        train_start = (start_date - relativedelta(months=month)).strftime("%Y%m%d")
    # 获取训练日期（排除最后6个交易日）
    train_date_list = [d for d in all_date if train_start <= d < test_start][:-6]
    # 排除特定日期（2024年2月）
    exclude_dates = [d for d in all_date if "20240201" <= d < "20240223"]
    train_date_list = sorted(set(train_date_list) - set(exclude_dates))
    return train_date_list[0], train_date_list[-1]


def adjust_sign(df):
    """
    根据IC的正负调整head和tail列。若IC为负，则交换head列和tail列的值
    """
    df = df.astype(float)
    # 获取需要交换的列名
    head_cols = [col for col in df.columns if "head" in col]
    tail_cols = [col for col in df.columns if "tail" in col]
    # 当IC为负时交换head和tail列的值
    mask = df["ic"] < 0
    df.loc[mask, head_cols + tail_cols] = df.loc[mask, tail_cols + head_cols].values
    return df


def clip_zscore(arr):
    """
    对数据进行裁剪和标准化
    :param arr: 原始数据
    :return: 裁剪和标准化后的数据
    """
    lower = np.nanquantile(arr, q=0.005, axis=0)
    upper = np.nanquantile(arr, q=0.995, axis=0)
    arr = np.clip(arr, lower, upper)
    arr = (arr - np.nanmean(arr, axis=0)) / np.nanstd(arr, axis=0)
    return arr


def get_fac_path(file_path):
    """
    从指定路径读取所有.feather文件，提取因子和收益率数据
    Args:
        file_path: 包含.feather文件的目录路径
        数据格式假设每个文件包含以下列：
            - "index": 股票代码
            - "ret_1d", "ret_5d", "ret_10d", "ret_20d": 收益率列
            - 其他列为因子数据
    Returns:
        tuple: (日期列表, 代码列表, 收益率数组列表, 因子数据列表)
    """
    # 获取所有.feather文件并按文件名排序
    files = sorted(f for f in os.listdir(file_path) if f.endswith(".fea"))
    all_date, all_code, all_ret, all_fac = [], [], [], []
    for file in tqdm(files, desc="加载因子文件"):
        date = file[:8]  # 从文件名提取日期
        df = pd.read_feather(os.path.join(file_path, file))
        # 提取收益率数据
        ret_columns = ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]
        all_ret.append(df[ret_columns].to_numpy(np.float32))
        # 提取股票代码
        all_code.append(df["index"].values)
        # 创建日期数组
        all_date.append([date] * len(df))
        # 提取因子数据（排除代码和收益率列）
        factor_data = df.drop(columns=["index"] + ret_columns, errors="ignore")
        all_fac.append(factor_data.values)
    return all_date, all_code, all_ret, all_fac


def get_ic(x, y):
    """
    计算数组x的每一列与y的皮尔逊相关系数
    Args:
        x: 2D数组 (n_samples, n_features)
        y: 1D数组 (n_samples,)
    Returns:
        1D数组: 每列的相关系数
    """
    # 去除均值
    x_centered = x - np.nanmean(x, axis=0, keepdims=True)
    y_centered = y - np.nanmean(y)
    # 计算协方差和标准差
    cov = np.nanmean(x_centered * y_centered[:, None], axis=0)
    x_std = np.sqrt(np.nanmean(x_centered**2, axis=0))
    y_std = np.sqrt(np.nanmean(y_centered**2))
    # 计算相关系数，避免除零
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = cov / (x_std * y_std)
        ic = np.nan_to_num(ic, nan=0.0, posinf=0.0, neginf=0.0)
    return ic


def get_facinfo_date(date):
    """
    获取指定日期的因子分析信息
    Args:
        date: 日期字符串，如"20220103"
    Returns:
        pd.DataFrame: 包含因子名称、IC、Rank IC、头部收益和尾部收益的DataFrame
    """
    # 数据准备
    returns = ret_data.loc[date].dropna()  # 收益率数据（5日）
    factors = fac_data.loc[date].set_index("Code")
    factor_names = factors.columns.to_numpy()
    # 数据对齐和筛选
    valid_codes = [c for c in returns.index.intersection(factors.index) if c in CODE_LIMIT]
    returns = returns.reindex(valid_codes).values.astype(float)
    factors = factors.reindex(valid_codes).values.astype(float)
    # 计算排名
    factors_rank = factors.argsort(axis=0).argsort(axis=0) / (len(returns) - 1)
    returns_rank = returns.argsort().argsort() / (len(returns) - 1)
    # 计算指标
    ic = get_ic(factors, returns)
    rank_ic = get_ic(factors_rank, returns_rank)
    # 计算头部尾部收益
    head_mask, tail_mask = factors_rank > 0.9, factors_rank < 0.1
    head_counts, tail_counts = head_mask.sum(axis=0), tail_mask.sum(axis=0)
    head10p = np.where(head_counts > 0, np.nansum(head_mask * returns[:, None], axis=0) / head_counts, np.nan)
    tail10p = np.where(tail_counts > 0, np.nansum(tail_mask * returns[:, None], axis=0) / tail_counts, np.nan)
    # 返回结果
    return pd.DataFrame({"date": date, "fac_name": factor_names, "ic": ic, "rank_ic": rank_ic, "head10p": head10p, "tail10p": tail10p})


def process_time_window(months, season):
    """处理单个时间窗口，返回因子集合"""
    eval_start, eval_end = get_eval_date(date_list, season, month=months)
    print(f"  回溯周期：{months}个月，评估时间段：{eval_start} - {eval_end}")
    fac_info = fac_info_all.loc[fac_info_all["date"].between(eval_start, eval_end)]
    fac_info = fac_info.sort_values(["fac_name", "date"]).groupby("fac_name", as_index=False)
    fac_info_mean = []
    for fac_name in fac_name_all:
        res = fac_info.get_group(fac_name)
        res_mean = res.loc[:, "ic":].mean()
        res_mean["fac_name"] = fac_name
        fac_info_mean.append(res_mean)
    fac_info_mean = pd.concat(fac_info_mean, axis=1).T.set_index("fac_name", drop=True)
    fac_info_mean = adjust_sign(fac_info_mean)
    rankic_abs = fac_info_mean["rank_ic"].abs()
    ic_abs = fac_info_mean["ic"].abs()
    rankic_threshold = rankic_abs.quantile(0.9)
    ic_threshold = ic_abs.quantile(0.9)
    sel_rankic = fac_info_mean[rankic_abs >= rankic_threshold].index.tolist()
    sel_ic = fac_info_mean[ic_abs >= ic_threshold].index.tolist()
    return set(sel_rankic + sel_ic)


def get_csstd_mvgroup_date(date, n_group=10):
    """
    分市值组标准化因子数据
    Args:
        date: 日期字符串，如"20220103"
        n_group: 市值分组数量，默认10
    Returns:
        pd.DataFrame: 标准化后的因子数据，包含"date"和"Code"列
    """
    c_data = fac_data.loc[date]
    mv = mv_df.loc[date].dropna()
    common_codes = c_data["Code"].intersection(mv.index).intersection(CODE_LIMIT)
    c_data = c_data.set_index("Code").loc[common_codes]
    mv = mv.loc[common_codes]
    mv_rank = mv.rank(pct=True)
    mv_group = (mv_rank * n_group).astype(int).clip(0, n_group - 1)
    factors = c_data.drop(columns=["date"], errors="ignore")
    result = factors.groupby(mv_group).apply(lambda x: pd.DataFrame(clip_zscore(x.values), index=x.index, columns=x.columns)).droplevel(0)
    result["date"] = date
    result["Code"] = result.index
    return result.reset_index(drop=True)


if __name__ == "__main__":
    BASE_PATH = r"/project/lz_remote"
    START_YEAR, END_YEAR = 2018, 2025  # 包含2025
    YEAR_LIST = [str(year) for year in range(START_YEAR, END_YEAR + 1)]
    c_date, c_code, c_ret, c_fac = [], [], [], []
    for y in YEAR_LIST:
        f_p = f"{BASE_PATH}/{y}"
        ydates, ycodes, yrets, yfacs = get_fac_path(f_p)
        c_date.extend(ydates)
        c_code.extend(ycodes)
        c_ret.extend(yrets)
        c_fac.extend(yfacs)

    all_date = np.concatenate(c_date, axis=0)
    all_code = np.concatenate(c_code, axis=0)
    all_ret = np.concatenate(c_ret, axis=0)
    all_fac = np.concatenate(c_fac, axis=0)
    fac_data = pd.DataFrame(all_fac, columns=[f"fac_{i+1}" for i in range(all_fac.shape[1])], index=all_date)
    fac_data["Code"] = all_code
    ret_df = pd.DataFrame(all_ret, columns=["ret_1d", "ret_5d", "ret_10d", "ret_20d"])
    ret_df["Code"] = all_code
    ret_df["date"] = all_date
    ret_1d = ret_df.pivot(index="date", columns="Code", values="ret_1d")
    ret_5d = ret_df.pivot(index="date", columns="Code", values="ret_5d")
    ret_10d = ret_df.pivot(index="date", columns="Code", values="ret_10d")
    ret_20d = ret_df.pivot(index="date", columns="Code", values="ret_20d")

    ret_data = ret_5d.copy()  # 使用5日收益率
    CODE_LIMIT = [c for c in ret_data.columns if not any(str(c).startswith(p) for p in "289")]  # 排除北交所、B股和其他特殊股票
    date_list = sorted([d for d in fac_data.index.unique() if d in ret_data.index])

    # 因子回测表现
    with Pool(processes=cpu_count() // 4) as pool:
        res = list(tqdm(pool.imap_unordered(get_facinfo_date, date_list), total=len(date_list)))
    fac_info_all = (pd.concat(res, axis=0, ignore_index=True)).sort_values("date")
    fac_name_all = sorted(list(fac_info_all["fac_name"].unique()))
    fac_info_all.reset_index(drop=True).to_feather(rf"/project/model_share/share_lz/lz_szsh_info_5d.fea")

    # fac_info_all的格式为：
    #        date    fac_name          ic      rank_ic    head10p    tail10p    ...
    # 0  20220103       fac_1      0.0123       0.0156     0.0023    -0.0012    ...
    # 1  20220103       fac_2      0.0123       0.0156     0.0023    -0.0012    ...
    # 2  20220103       fac_3      0.0123       0.0156     0.0023    -0.0012    ...
    # ...

    # 分季度因子筛选
    START_YEAR, END_YEAR = 2022, 2025
    SEASON_LIST = []
    for year in range(START_YEAR, END_YEAR + 1):
        for quarter in range(1, 5):  # 1-4季度
            SEASON_LIST.append(f"{year}q{quarter}")

    # 分长短回溯月数筛选得到高ic、rankic因子
    season_cluster_dict = dict()
    backward_months_list = [24, 36, 48]  # 长期回溯月数
    short_term_months_list = [3, 6]  # 短期回溯月数
    for season in SEASON_LIST:
        print(f"当前季度：{season}")
        # 长期处理：取三个时间窗口的交集
        long_term_sets = [process_time_window(m, season) for m in backward_months_list]
        long_term_intersection = set.intersection(*long_term_sets) if long_term_sets else set()
        print(f"  长期交集因子数：{len(long_term_intersection)}")
        # 短期处理：取两个时间窗口的并集（使用列表推导式，与长期风格统一）
        short_term_sets = [process_time_window(m, season) for m in short_term_months_list]
        short_term_union = set.union(*short_term_sets) if short_term_sets else set()
        print(f"  短期并集因子数：{len(short_term_union)}")
        # 合并长期交集和短期并集
        final_factors = long_term_intersection.union(short_term_union)
        season_cluster_dict[season] = list(final_factors)
        print(f"  {season} 最终因子数：{len(final_factors)}\n")

    # 市值分组标准化
    mv_df = pd.read_feather(rf"/project/tonglian_remote/ohlc_fea/MARKET_VALUE.fea")
    mv_df = mv_df.rename(columns={"TRADE_DATE": "date"})
    mv_df["date"] = mv_df["date"].apply(lambda x: str(x).replace("-", ""))
    mv_df = mv_df.set_index("date")

    with Pool(cpu_count() // 2) as pool:
        mv_std_results = list(tqdm(pool.imap(get_csstd_mvgroup_date, date_list), total=len(date_list)))
    fac_data = pd.concat(mv_std_results, axis=0, ignore_index=True)

    new_fac_name = rf"lz_icric_longshort0.9_mv10group_newcode_5dret间隔6"
    new_data_path = rf"/project/model_share/share_lz/factor_dly"
    os.makedirs(rf"{new_data_path}/{new_fac_name}", exist_ok=True)

    for season in tqdm(SEASON_LIST):
        sel_fac = season_cluster_dict[season]
        fea = fac_data[["Code", "date"] + sel_fac]
        fea.to_feather(rf"{new_data_path}/{new_fac_name}/{season}_num-{fea.shape[1]-2}.fea")
