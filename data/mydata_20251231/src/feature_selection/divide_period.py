import os
import pandas as pd
import json

def generate_train_predict_dates(
    date_list, train_period_days=720, predict_period_days=60, slide_period_days=60, gap_days=10, from_start=False, remove_abnormal=True
):
    """
    Generate training and prediction period date lists based on the provided trading dates.
    Period lengths and slide steps are measured in trading days (not calendar days).

    Parameters:
        date_list (list): List of trading dates in ascending order (format: 'YYYYMMDD')
        train_period_days (int): Number of trading days in each training period
        predict_period_days (int): Number of trading days in each prediction period
        slide_period_days (int): Number of trading days to slide the window forward each iteration
        gap_days (int): Number of trading days between the end of the training period and the start of the prediction period

    Returns:
        tuple: (len(train_dates_list), train_dates_list, predict_dates_list)
            - len(train_dates_list): Number of training/prediction periods
            - train_dates_list: List of all training periods, each as a sublist of dates
            - predict_dates_list: List of all prediction periods, each as a sublist of dates
    """
    train_dates_list = []
    predict_dates_list = []

    n = len(date_list)
    start_idx = 0

    while True:
        # Define index ranges
        if from_start:
            train_start_idx = 0
        else:
            train_start_idx = start_idx
        train_end_idx = start_idx + train_period_days
        predict_start_idx = train_end_idx + gap_days
        predict_end_idx = min(predict_start_idx + predict_period_days, n)

        # Check boundary conditions
        if train_end_idx >= n - gap_days:
            break

        # Slice trading days based on indices
        train_dates = date_list[train_start_idx:train_end_idx]
        if remove_abnormal:
            train_dates = [date for date in train_dates if (date < "20240201" or date > "20240223")]
        predict_dates = date_list[predict_start_idx:predict_end_idx]

        if not predict_dates:
            break

        train_dates_list.append(train_dates)
        predict_dates_list.append(predict_dates)

        # Slide window forward
        start_idx += slide_period_days

    return len(train_dates_list), train_dates_list, predict_dates_list

os.makedirs("/home/haris/raid0/shared/haris/mydata_20251231/feature_selection", exist_ok=True)

trade_date = pd.read_feather("/home/haris/raid0/shared/haris/mydata_20251231/trade_date.fea")["trade_date"].to_list()
trade_date = [date for date in trade_date if date >= "20180401"]
num_periods, train_dates_list, predict_dates_list = generate_train_predict_dates(trade_date, gap_days=20)
with open("/home/haris/raid0/shared/haris/mydata_20251231/feature_selection/train_predict_period.json", "w") as f:
    json.dump({"num_periods": num_periods, "train_dates_list": train_dates_list, "predict_dates_list": predict_dates_list}, f, indent=4)

trade_date = pd.read_feather("/home/haris/raid0/shared/haris/mydata_20251231/trade_date_10.fea")["trade_date"].to_list()
trade_date = [date for date in trade_date if date >= "20180401"]
num_periods, train_dates_list, predict_dates_list = generate_train_predict_dates(trade_date)
with open("/home/haris/raid0/shared/haris/mydata_20251231/feature_selection/train_predict_period_10.json", "w") as f:
    json.dump({"num_periods": num_periods, "train_dates_list": train_dates_list, "predict_dates_list": predict_dates_list}, f, indent=4)

trade_date = pd.read_feather("/home/haris/raid0/shared/haris/mydata_20251231/trade_date_mix.fea")["trade_date"].to_list()
trade_date = [date for date in trade_date if date >= "20180401"]
num_periods, train_dates_list, predict_dates_list = generate_train_predict_dates(trade_date)
with open("/home/haris/raid0/shared/haris/mydata_20251231/feature_selection/train_predict_period_mix.json", "w") as f:
    json.dump({"num_periods": num_periods, "train_dates_list": train_dates_list, "predict_dates_list": predict_dates_list}, f, indent=4)

