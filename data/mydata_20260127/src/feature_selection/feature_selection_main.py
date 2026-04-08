import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, Tuple, Optional

class FactorSelectionFramework:
    def __init__(self, correlation_matrix_path: str, rankic_mean_path: str):
        self.correlation_matrix_path = correlation_matrix_path
        self.rankic_mean_path = rankic_mean_path
        self.correlation_matrix = None
        self.rankic_means = None
        self.factor_count = None
        self.period_count = None
    
    def load_and_validate_data(self) -> None:
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
        
    def standardize_rankic(self) -> None:
        for period_idx in range(self.period_count):
            rankic_values = self.rankic_means.iloc[:, period_idx].values
            min_val, max_val = np.min(rankic_values), np.max(rankic_values)
            if max_val - min_val > 0:
                self.rankic_means.iloc[:, period_idx] = (rankic_values - min_val) / (max_val - min_val)
            else:
                self.rankic_means.iloc[:, period_idx] = 0
                
    def select_factors_by_correlation(self, correlation_threshold: float = 0.7, target_factor_ratio: float = 0.5) -> np.ndarray:
        target_count = int(self.factor_count * target_factor_ratio)
        selected_factor_index = np.zeros((target_count, self.period_count), dtype=int)
        
        for period_idx in range(self.period_count):
            print(f"正在处理周期 {period_idx + 1}/{self.period_count}")
            rankic_values = self.rankic_means.iloc[:, period_idx].values
            correlation_matrix_values = self.correlation_matrix[self.correlation_matrix["period"] == f"period_{period_idx + 1}"].values
            correlation_matrix_values = correlation_matrix_values[:, :-1]
            selected_factors = self._select_factors_single_period(correlation_matrix_values, rankic_values, 
                                                                correlation_threshold, target_count)
            selected_factor_index[:, period_idx] = selected_factors
        return selected_factor_index
    
    def _select_factors_single_period(self, correlation_matrix_values: np.ndarray, rankic_values: np.ndarray,
                                    correlation_threshold: float, target_count: int) -> np.ndarray:
        pbar = tqdm(total=target_count, desc="因子筛选进度", leave=True, colour='green')
        selected = []
        remaining = list(range(self.factor_count))
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
            pbar.update(1)
        pbar.close()
        return np.array(selected, dtype=int)
    
    def evaluate_selection(self, selected_factors_matrix: np.ndarray) -> Dict[str, float]:
        avg_correlation = []
        avg_rankic = []
        for period_idx in range(self.period_count):
            print(f"正在评估周期 {period_idx + 1}/{self.period_count}")
            rankic_values = self.rankic_means.iloc[:, period_idx].values
            selected_factors = selected_factors_matrix[:, period_idx]
            selected_corr = self.correlation_matrix[np.ix_(selected_factors, selected_factors)]
            selected_rankic = rankic_values[selected_factors]
            avg_corr = np.mean(np.abs(selected_corr[np.triu_indices(selected_corr.shape[0], k=1)]))
            avg_correlation.append(avg_corr)
            avg_rankic.append(np.mean(selected_rankic))
        return {
            'average_correlation': np.mean(avg_correlation),
            'average_rankic': np.mean(avg_rankic)
        }
        
    def save_selected_factors(self, selected_factor_index: np.ndarray, output_path: str) -> None:
        periods = [f'period_{i+1}' for i in range(self.period_count)]
        df = pd.DataFrame(selected_factor_index, columns=periods)
        df.index.name = 'factor_index'
        df.to_csv(output_path)
    
    def run_full_selection(self, correlation_threshold: float = 0.7, 
                         target_factor_ratio: float = 0.55,
                         standardize: bool = True,
                         output_path: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        self.load_and_validate_data()
        if standardize:
            self.standardize_rankic()
        selected_factor_index = self.select_factors_by_correlation(correlation_threshold, target_factor_ratio)
        if output_path:
            self.save_selected_factors(selected_factor_index, output_path)
        return selected_factor_index

def main():
    suffixes = ["", "_10", "_mix"]
    for s in suffixes:
        correlation_path = f"/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/period_correlation_matrix{s}.csv"
        rankic_path = f"/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/period_rankic_mean{s}.csv"
        output_path = f"/home/haris/raid0/shared/haris/mydata_20260127/feature_selection/selected_factor_index_correlation_matrix{s}.csv"
        framework = FactorSelectionFramework(correlation_path, rankic_path)
        selected_matrix = framework.run_full_selection(
            correlation_threshold=0.7,
            target_factor_ratio=0.5,
            standardize=True,
            output_path=output_path
        )
        print("因子筛选完成！")
        print(f"筛选后因子数量: {selected_matrix.shape[0]}")
        print(f"周期数量: {selected_matrix.shape[1]}")
        print(f"平均相关系数: {evaluation['average_correlation']:.4f}")
        print(f"平均rankIC: {evaluation['average_rankic']:.4f}")
        print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()


