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
    return rank_ic.fillna(1.0).values  # TODO: try other values

def calculate_period_rankic(suffix=""):
    """主函数：并行计算 RankIC"""
    print(f"\n{'='*20} Processing suffix: {suffix} {'='*20}")
    # 路径配置
    base_path = "/home/haris/raid0/shared/haris/mydata_20251231"
    json_path = f"{base_path}/feature_selection/train_predict_period{suffix}.json"
    data_dir = f"{base_path}/concat_daily_factor_with_label{suffix}"
    output_file = f"{base_path}/feature_selection/period_rankic_mean{suffix}.csv"
        
    with open(json_path, "r") as f:
        meta_data = json.load(f)
        num_periods = meta_data["num_periods"]
        train_dates_list = meta_data["train_dates_list"]
    period_rankic_results = {}
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
        # 计算该 period 的平均 RankIC
        if daily_rankic_list:
            # 转换为 numpy 矩阵一次性计算均值 (Axis 0 是日期维度)
            period_rankic = np.mean(daily_rankic_list, axis=0)
            period_rankic_results[f"period_{period_idx + 1}"] = period_rankic
        else:
            print(f"Warning: No valid data for period {period_idx + 1}")
    # 保存结果
    if period_rankic_results:
        df_result = pd.DataFrame(period_rankic_results)
        df_result.to_csv(output_file, index=True)
        print(f"Success! Results saved to: {output_file}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    suffixes = ["", "_10", "_mix"]
    for s in suffixes:
        calculate_period_rankic(suffix=s)