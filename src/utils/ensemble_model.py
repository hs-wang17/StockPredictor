from typing import Any, Dict, Optional
import lightgbm as lgb
import joblib
import os


def create_lgbm_model(
    n_estimators: int = 1000,
    objective: str = "regression",
    boosting_type: str = "gbdt",
    random_state: int = 42,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_data_in_leaf: int = 20,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 1,
) -> lgb.LGBMRegressor:
    """
    创建 LGBM 回归器 (sklearn API wrapper)。
    params: lightgbm 参数字典（会覆盖默认参数）。
    """
    params = {
        "objective": objective,
        "boosting_type": boosting_type,
        "random_state": random_state,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "device_type": "cuda",
    }
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)
    return model


def save_lgbm_model(model: lgb.LGBMRegressor, file_path: str):
    """
    保存整个 sklearn-wrapped LGBM 模型（使用 joblib，方便 reload 继续 predict）。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)


def load_lgbm_model(file_path: str) -> lgb.LGBMRegressor:
    """
    从文件加载模型。
    """
    model = joblib.load(file_path)
    return model
