#!/bin/bash

/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata/src/concat_day2day_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata/src/concat_daily_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata/src/calculate_label.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata/src/concat_daily_factor_with_label.py