#!/bin/bash

/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/concat_day2day_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/concat_daily_factor.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/calculate_label.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/concat_daily_factor_with_label.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/calculate_label_10.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/concat_daily_factor_with_label_10.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/calculate_label_mix.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/concat_daily_factor_with_label_mix.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/calculate_index_weight.py
/home/haris/miniconda3/envs/myenv/bin/python /home/haris/mydata_20251231/src/concat_daily_factor_with_label_10_index.py