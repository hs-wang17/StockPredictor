#!/bin/bash

/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/concat_daily_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/calculate_label.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/concat_daily_factor_with_label.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/calculate_label_10.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/concat_daily_factor_with_label_10.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/calculate_label_mix.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20260127/src/concat_daily_factor_with_label_mix.py