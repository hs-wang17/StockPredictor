import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings

def process_single_file_for_correlation(file_path):
    """
    处理单个文件的逻辑：读取 -> 计算因子相关系数矩阵
    """
    data = pd.read_feather(file_path)
    factors = data.iloc[:, :-1]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                               message="invalid value encountered in divide")
        correlation_matrix = np.corrcoef(factors, rowvar=False)
        # 将NaN填充为0（避免过度浪费因子）
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        # 对角线设置为1（因为常数列的对角线可能被填充为0）
        np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix

def calculate_period_correlation_matrix(suffix=""):
    """主函数：并行计算每个period的因子相关系数矩阵"""
    print(f"\n{'='*20} Processing suffix: {suffix} {'='*20}")
    # 路径配置
    base_path = "/home/haris/raid0/shared/haris/mydata_20251231"
    json_path = f"{base_path}/feature_selection/train_predict_period{suffix}.json"
    data_dir = f"{base_path}/concat_daily_factor_with_label{suffix}"
    output_file = f"{base_path}/feature_selection/period_correlation_matrix{suffix}.csv"
    rankic_file = f"{base_path}/feature_selection/period_rankic_mean{suffix}.csv"

    with open(json_path, "r") as f:
        meta_data = json.load(f)
        num_periods = meta_data["num_periods"]
        train_dates_list = meta_data["train_dates_list"]
    period_correlation_results = {}
    max_workers = multiprocessing.cpu_count() - 2

    for period_idx in range(num_periods):
        period_train_dates = train_dates_list[period_idx]
        file_paths = [os.path.join(data_dir, f"{date}.fea") for date in period_train_dates]
        n_factors = pd.read_csv(rankic_file).shape[0] 
        running_sum = np.zeros((n_factors, n_factors), dtype=np.float64)
        count = 0
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = {executor.submit(process_single_file_for_correlation, fp): fp for fp in file_paths}
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(futures), 
                               total=len(file_paths), 
                               desc=f"Period {period_idx + 1}/{num_periods}", 
                               colour='green'):
                result = future.result()
                if result is not None:
                    running_sum += result
                    count += 1
                    del result 
        if count > 0:
            period_correlation = running_sum / count
            period_correlation_results[f"period_{period_idx + 1}"] = period_correlation
        else:
            print(f"Warning: No valid data for period {period_idx + 1}")
    # 保存结果到CSV文件
    if period_correlation_results:
        # 创建多层DataFrame便于保存
        all_matrices = []
        for period_name, matrix in period_correlation_results.items():
            df = pd.DataFrame(matrix)
            df['period'] = period_name
            all_matrices.append(df)
        combined_df = pd.concat(all_matrices, ignore_index=False)
        combined_df.to_csv(output_file, index=False)
        print(f"Success! Results saved to: {output_file}")
    else:
        print("Warning: No results to save")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    suffixes = ["", "_10", "_mix"]
    for s in suffixes:
        calculate_period_correlation_matrix(suffix=s)
