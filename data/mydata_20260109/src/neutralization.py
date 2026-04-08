import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


def neutralize_label_dummy(label_df, factor_data_path="/home/haris/project/backtester/data/trade_support7"):
    """对标签进行行业和市值中性化（哑变量）"""
    neutralized_df = pd.DataFrame(index=label_df.index, columns=label_df.columns)
    for date in tqdm(label_df.index, desc="中性化处理（哑变量）"):
        date_str = str(date)
        # 加载该日期的因子数据
        factor_data = pd.read_feather(f"{factor_data_path}/{date_str}.fea")
        # 获取该日期的标签
        label_series = label_df.loc[date]
        valid_stocks = label_series.dropna().index
        # 获取因子数据中对应的股票
        factor_data_subset = factor_data.reindex(valid_stocks).dropna()
        common_stocks = label_series.reindex(factor_data_subset.index).dropna().index
        missing_stocks = valid_stocks.difference(factor_data_subset.index)
        # 获取行业和市值哑变量
        industry_cols = [col for col in factor_data_subset.columns if col.startswith("citic_r_")]
        market_cap_cols = [col for col in factor_data_subset.columns if col.startswith("cmvg_r_")]
        # 各自删除一列避免多重共线性
        industry_dummies = factor_data_subset[industry_cols].iloc[:, 1:]
        market_cap_dummies = factor_data_subset[market_cap_cols].iloc[:, 1:]
        # 合并哑变量
        all_dummies = pd.concat([industry_dummies, market_cap_dummies], axis=1)
        # 准备回归数据
        dummies_subset = all_dummies.reindex(common_stocks).fillna(0)
        y = label_series.reindex(common_stocks).values.reshape(-1, 1)
        X = dummies_subset.values
        # 进行线性回归
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y.flatten())
        y_pred = model.predict(X)
        residuals = y.flatten() - y_pred
        # 创建中性化后的结果
        result_series = label_series.copy()
        result_series.loc[common_stocks] = residuals
        result_series.loc[missing_stocks] = pd.NA
        neutralized_df.loc[date] = result_series

    return neutralized_df


def neutralize_label_beta(label_df, factor_data_path="/home/haris/project/backtester/data/trade_support7"):
    """对标签进行行业和市值中性化（贝塔）"""
    neutralized_df = pd.DataFrame(index=label_df.index, columns=label_df.columns)
    for date in tqdm(label_df.index, desc="中性化处理（贝塔）"):
        date_str = str(date)
        # 加载该日期的因子数据
        factor_data = pd.read_feather(f"{factor_data_path}/{date_str}.fea")
        # 获取该日期的标签
        label_series = label_df.loc[date]
        valid_stocks = label_series.dropna().index
        # 获取因子数据中对应的股票
        factor_data_subset = factor_data.reindex(valid_stocks).dropna()
        common_stocks = label_series.reindex(factor_data_subset.index).dropna().index
        missing_stocks = valid_stocks.difference(factor_data_subset.index)
        # 获取行业和市值贝塔
        industry_cols = [col for col in factor_data_subset.columns if col.startswith("citic_b_")]
        market_cap_cols = [col for col in factor_data_subset.columns if col.startswith("cmvg_b_")]
        industry_betas = factor_data_subset[industry_cols]
        market_cap_betas = factor_data_subset[market_cap_cols]
        # 合并贝塔
        all_betas = pd.concat([industry_betas, market_cap_betas], axis=1)
        # 准备回归数据
        betas_subset = all_betas.reindex(common_stocks).fillna(0)
        y = label_series.reindex(common_stocks).values.reshape(-1, 1)
        X = betas_subset.values
        # 进行线性回归
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y.flatten())
        y_pred = model.predict(X)
        residuals = y.flatten() - y_pred
        # 创建中性化后的结果
        result_series = label_series.copy()
        result_series.loc[common_stocks] = residuals
        result_series.loc[missing_stocks] = pd.NA
        neutralized_df.loc[date] = result_series

    return neutralized_df
