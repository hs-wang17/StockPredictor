import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class FactorSelectionFramework:
    def __init__(self, correlation_matrix_path: str, rankic_mean_path: str):
        self.correlation_matrix_path = correlation_matrix_path
        self.rankic_mean_path = rankic_mean_path
        self.correlation_matrix = None
        self.rankic_means = None
        self.factor_count = None
        self.period_count = None
    
    def load_and_validate_data(self) -> None:
        """加载并验证数据"""
        if not os.path.exists(self.correlation_matrix_path):
            raise FileNotFoundError(f"相关系数矩阵文件不存在: {self.correlation_matrix_path}")
        if not os.path.exists(self.rankic_mean_path):
            raise FileNotFoundError(f"RankIC均值文件不存在: {self.rankic_mean_path}")
        
        print("正在加载RankIC均值数据...")
        self.rankic_means = pd.read_csv(self.rankic_mean_path, index_col=0)
        self.factor_count, self.period_count = self.rankic_means.shape[0], self.rankic_means.shape[1]
        print(f"因子数量: {self.factor_count}, 周期数量: {self.period_count}")
        
        print("正在加载相关系数矩阵数据...")
        self.correlation_matrix = pd.read_csv(self.correlation_matrix_path, index_col=None)
        print(f"相关系数矩阵形状: {self.correlation_matrix.shape}")
        
        # 将相关系数矩阵转换为每个周期的矩阵字典，便于并行访问
        self.period_corr_matrices = {}
        for period_idx in range(self.period_count):
            period_data = self.correlation_matrix[self.correlation_matrix["period"] == f"period_{period_idx + 1}"]
            corr_matrix = period_data.iloc[:, :-1].values  # 去掉period列
            self.period_corr_matrices[period_idx] = corr_matrix
        
    def standardize_rankic(self) -> None:
        """标准化RankIC值"""
        for period_idx in range(self.period_count):
            rankic_values = self.rankic_means.iloc[:, period_idx].values
            min_val, max_val = np.min(rankic_values), np.max(rankic_values)
            if max_val - min_val > 0:
                self.rankic_means.iloc[:, period_idx] = (rankic_values - min_val) / (max_val - min_val)
            else:
                self.rankic_means.iloc[:, period_idx] = 0
    
    @staticmethod
    def _select_factors_single_period_static(period_idx: int, 
                                            rankic_values: np.ndarray,
                                            correlation_matrix_values: np.ndarray,
                                            correlation_threshold: float,
                                            target_count: int) -> Tuple[int, np.ndarray]:
        """
        静态方法：处理单个周期的因子筛选
        """
        selected = []
        factor_count = len(rankic_values)
        remaining = list(range(factor_count))
        remaining.sort(key=lambda x: -rankic_values[x], reverse=True)
        while len(selected) < target_count and remaining:
            best_score = -np.inf
            best_factor = None
            for factor in remaining:
                if selected:
                    corr_sum = np.sum(np.abs(correlation_matrix_values[factor, selected]))
                    avg_corr = corr_sum / len(selected)
                else:
                    avg_corr = 0
                score = rankic_values[factor] * (1 - avg_corr)
                if score > best_score:
                    best_score = score
                    best_factor = factor
            selected.append(best_factor)
            remaining.remove(best_factor)
            if len(selected) % 100 == 0:
                print(f"周期 {period_idx + 1} 已选择 {len(selected)} 个因子")
        return period_idx, np.array(selected, dtype=int)
    
    def select_factors_by_correlation_parallel(self, 
                                              correlation_threshold: float = 0.7, 
                                              target_factor_ratio: float = 0.5,
                                              max_workers: Optional[int] = None) -> np.ndarray:
        """
        并行处理所有周期的因子筛选
        """
        target_count = int(self.factor_count * target_factor_ratio)
        # 准备每个周期的输入数据
        period_data = []
        for period_idx in range(self.period_count):
            rankic_values = self.rankic_means.iloc[:, period_idx].values
            correlation_matrix_values = self.correlation_matrix[self.correlation_matrix["period"] == f"period_{period_idx + 1}"].values
            correlation_matrix_values = correlation_matrix_values[:, :-1]
            period_data.append((period_idx, rankic_values, correlation_matrix_values))
        
        # 设置并行进程数
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 2)
        
        # 存储结果
        results = [None] * self.period_count
        
        print(f"使用 {max_workers} 个进程并行处理 {self.period_count} 个周期...")
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {}
            for period_idx, rankic_values, correlation_matrix_values in period_data:
                future = executor.submit(
                    self._select_factors_single_period_static,
                    period_idx,
                    rankic_values,
                    correlation_matrix_values,
                    correlation_threshold,
                    target_count
                )
                futures[future] = period_idx
            
            # 收集结果，使用tqdm显示进度
            with tqdm(total=self.period_count, desc="并行筛选进度", colour='green') as pbar:
                for future in as_completed(futures):
                    period_idx, selected_factors = future.result()
                    results[period_idx] = selected_factors
                    pbar.update(1)
        
        # 将结果组装成矩阵
        selected_factor_matrix = np.column_stack(results)
        
        return selected_factor_matrix
    
    def select_factors_by_correlation(self, 
                                     correlation_threshold: float = 0.7, 
                                     target_factor_ratio: float = 0.5,
                                     parallel: bool = True,
                                     max_workers: Optional[int] = None) -> np.ndarray:
        """
        因子筛选主函数，可选择串行或并行模式
        """
        if parallel:
            return self.select_factors_by_correlation_parallel(
                correlation_threshold, target_factor_ratio, max_workers
            )
        else:
            # 保留原有的串行处理逻辑
            target_count = int(self.factor_count * target_factor_ratio)
            selected_factor_index = np.zeros((target_count, self.period_count), dtype=int)
            
            for period_idx in range(self.period_count):
                print(f"正在处理周期 {period_idx + 1}/{self.period_count}")
                rankic_values = self.rankic_means.iloc[:, period_idx].values
                corr_matrix = self.period_corr_matrices[period_idx]
                selected_factors = self._select_factors_single_period_static(
                    period_idx, rankic_values, corr_matrix, 
                    correlation_threshold, target_count
                )[1]
                selected_factor_index[:, period_idx] = selected_factors
            
            return selected_factor_index
    
    def evaluate_selection(self, selected_factors_matrix: np.ndarray) -> Dict[str, float]:
        """
        评估筛选结果
        """
        avg_correlation = []
        avg_rankic = []
        
        for period_idx in range(self.period_count):
            rankic_values = self.rankic_means.iloc[:, period_idx].values
            selected_factors = selected_factors_matrix[:, period_idx]
            corr_matrix = self.period_corr_matrices[period_idx]
            
            # 获取选中因子的相关系数子矩阵
            selected_corr = corr_matrix[np.ix_(selected_factors, selected_factors)]
            selected_rankic = rankic_values[selected_factors]
            
            # 计算平均相关系数（上三角部分）
            if selected_corr.shape[0] > 1:
                avg_corr = np.mean(np.abs(selected_corr[np.triu_indices(selected_corr.shape[0], k=1)]))
            else:
                avg_corr = 0
            
            avg_correlation.append(avg_corr)
            avg_rankic.append(np.mean(selected_rankic))
        
        return {
            'average_correlation': np.mean(avg_correlation),
            'average_rankic': np.mean(avg_rankic),
            'period_correlations': avg_correlation,
            'period_rankics': avg_rankic
        }
        
    def save_selected_factors(self, selected_factor_index: np.ndarray, output_path: str) -> None:
        """
        保存筛选结果
        """
        periods = [f'period_{i+1}' for i in range(self.period_count)]
        df = pd.DataFrame(selected_factor_index, columns=periods)
        df.index.name = 'factor_index'
        df.to_csv(output_path)
    
    def run_full_selection(self, 
                          correlation_threshold: float = 0.7, 
                          target_factor_ratio: float = 0.5,
                          standardize: bool = True,
                          parallel: bool = True,
                          max_workers: Optional[int] = None,
                          output_path: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        运行完整的因子筛选流程
        """
        self.load_and_validate_data()
        if standardize:
            self.standardize_rankic()
        selected_factor_index = self.select_factors_by_correlation(
            correlation_threshold, target_factor_ratio, parallel, max_workers
        )
        evaluation = self.evaluate_selection(selected_factor_index)
        if output_path:
            self.save_selected_factors(selected_factor_index, output_path)
        return selected_factor_index, evaluation


def process_single_suffix(suffix: str, correlation_threshold: float = 0.7, 
                         target_factor_ratio: float = 0.5, 
                         standardize: bool = True,
                         parallel: bool = True,
                         max_workers: Optional[int] = None) -> None:
    """
    处理单个后缀的因子筛选
    """
    correlation_path = f"/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/period_rankic_correlation_matrix{suffix}.csv"
    rankic_path = f"/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/period_rankic_mean{suffix}.csv"
    output_path = f"/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/selected_factor_index_rankic_correlation_matrix{suffix}.csv"
    
    print(f"\n{'='*50}")
    print(f"Processing suffix: '{suffix}'")
    print(f"{'='*50}")
    
    try:
        framework = FactorSelectionFramework(correlation_path, rankic_path)
        selected_matrix, evaluation = framework.run_full_selection(
            correlation_threshold=correlation_threshold,
            target_factor_ratio=target_factor_ratio,
            standardize=standardize,
            parallel=parallel,
            max_workers=max_workers,
            output_path=output_path
        )
        
        print(f"\n筛选完成！")
        print(f"筛选后因子数量: {selected_matrix.shape[0]}")
        print(f"周期数量: {selected_matrix.shape[1]}")
        print(f"平均相关系数: {evaluation['average_correlation']:.4f}")
        print(f"平均rankIC: {evaluation['average_rankic']:.4f}")
        print(f"结果已保存至: {output_path}")
        
    except Exception as e:
        print(f"处理后缀 '{suffix}' 时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数：并行处理多个后缀
    """
    suffixes = ["", "_10", "_mix"]
    
    # 设置并行模式
    # parallel=True: 每个周期的因子筛选并行执行
    # parallel=False: 串行执行
    parallel = True
    
    # 设置每个后缀处理时使用的最大进程数
    # None表示自动检测（CPU核心数-2）
    max_workers_per_suffix = 20
    
    print(f"并行模式: {'开启' if parallel else '关闭'}")
    if parallel:
        print(f"每个后缀使用的最大进程数: {max_workers_per_suffix or '自动'}")
    
    # 可以选择串行处理多个后缀，或者也并行处理多个后缀
    # 这里使用串行处理后缀（因为每个后缀已经使用了多进程）
    for suffix in suffixes:
        process_single_suffix(
            suffix=suffix,
            correlation_threshold=0.7,
            target_factor_ratio=0.5,
            standardize=True,
            parallel=parallel,
            max_workers=max_workers_per_suffix
        )


if __name__ == "__main__":
    # 设置启动方法为spawn（适用于多进程）
    multiprocessing.set_start_method('spawn', force=True)
    main()