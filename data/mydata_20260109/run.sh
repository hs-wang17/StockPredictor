#!/bin/bash

/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260109/src/concat_day2day_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260109/src/concat_daily_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260109/src/calculate_label.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260109/src/calculate_label_dummy.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260109/src/calculate_label_beta.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260109/src/concat_daily_factor_with_label.py