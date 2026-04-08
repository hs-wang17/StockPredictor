import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings

def process_single_file(file_path):
    """
    处理单个文件的逻辑：读取 -> 排名 -> 计算相关系数
    """
    data = pd.read_feather(file_path)
    factors, labels = data.iloc[:, :-1], data.iloc[:, -1]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                message="An input array is constant.*")
        rank_ic = factors.corrwith(labels, method='spearman')
    # 处理异常值：如果因子没有波动导致 correlation 为 NaN，填充为 1.0 (即保留该因子)
    return rank_ic.fillna(1.0).values

def calculate_period_rankic_correlation_matrix(suffix=""):
    """主函数：并行计算 RankIC 并构建相关性矩阵"""
    print(f"\n{'='*20} Processing suffix: {suffix} {'='*20}")
    # 路径配置
    base_path = "/home/haris/raid0/shared/haris/mydata_20251231"
    json_path = f"{base_path}/feature_selection/train_predict_period{suffix}.json"
    data_dir = f"{base_path}/concat_daily_factor_with_label{suffix}"
    output_file = f"{base_path}/feature_selection/period_rankic_correlation_matrix{suffix}.csv"
        
    with open(json_path, "r") as f:
        meta_data = json.load(f)
        num_periods = meta_data["num_periods"]
        train_dates_list = meta_data["train_dates_list"]
    period_correlation_matrices = []
    max_workers = multiprocessing.cpu_count() - 2
    
    for period_idx in range(num_periods):
        period_train_dates = train_dates_list[period_idx]
        file_paths = [os.path.join(data_dir, f"{date}.fea") for date in period_train_dates]
        daily_rankic_list = []
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = {executor.submit(process_single_file, fp): fp for fp in file_paths}
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(futures), 
                               total=len(file_paths), 
                               desc=f"Period {period_idx + 1}/{num_periods}", 
                               colour='green'):
                result = future.result()
                if result is not None:
                    daily_rankic_list.append(result)
        # 构建 n×m 矩阵：行 = 因子，列 = 交易日
        # daily_rankic_list 的形状: [m, n]，需要转置为 [n, m]
        daily_rankic_matrix = np.array(daily_rankic_list).T
        # 计算因子间的相关系数矩阵 (n×n)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                   message="invalid value encountered in divide")
            correlation_matrix = np.corrcoef(daily_rankic_matrix)
        period_correlation_matrices.append(correlation_matrix)
    # 保存结果到CSV文件
    if period_correlation_matrices:
        # 获取因子数量
        n_factors = period_correlation_matrices[0].shape[0]
        # 创建多层DataFrame便于保存
        all_matrices = []
        for period_idx, matrix in enumerate(period_correlation_matrices):
            df = pd.DataFrame(matrix)
            df['period'] = f'period_{period_idx + 1}'
            all_matrices.append(df)
        combined_df = pd.concat(all_matrices, ignore_index=False)
        combined_df.to_csv(output_file, index=False)
        print(f"Success! Results saved to: {output_file}")
        print(f"Final 3D matrix shape: ({n_factors}, {n_factors}, {len(period_correlation_matrices)})")
        print(f"Number of factors: {n_factors}, Number of periods: {len(period_correlation_matrices)}")
    else:
        print("Warning: No valid data for any period")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    suffixes = ["", "_10", "_mix"]
    for s in suffixes:
        calculate_period_rankic_correlation_matrix(suffix=s)