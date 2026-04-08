import pandas as pd

trade_date = pd.read_pickle("/home/haris/data/trade_support_data/trade_days_dict.pkl")
my_trade_date = pd.read_feather("/home/haris/raid0/shared/haris/mydata_20260127/trade_date.fea")

# 将my_trade_date中的字符串日期转换为datetime.date类型
my_trade_date['trade_date'] = pd.to_datetime(my_trade_date['trade_date'], format='%Y%m%d').dt.date
# 获取trade_date中的交易日列表
trade_days = trade_date['trade_days']
# 检查my_trade_date中的所有日期是否都在trade_days中
is_all_in_trade_date = my_trade_date['trade_date'].isin(trade_days)
# 打印结果
print(f"所有日期都在trade_date中: {is_all_in_trade_date.all()}")
print(f"不在trade_date中的日期数量: {len(my_trade_date[~is_all_in_trade_date])}")
if not is_all_in_trade_date.all():
    print("不在trade_date中的日期:")
    print(my_trade_date[~is_all_in_trade_date])

# 反过来，检查在my_trade_date范围内的trade_date日期是否都在my_trade_date中
my_trade_date_min = my_trade_date['trade_date'].min()
my_trade_date_max = my_trade_date['trade_date'].max()
# 找出trade_days中在my_trade_date范围内的日期
trade_days_in_range = [date for date in trade_days if my_trade_date_min <= date <= my_trade_date_max]
# 将trade_days_in_range转换为集合以便快速查找
trade_days_set = set(trade_days_in_range)
# 检查这些日期是否都在my_trade_date中
my_trade_date_set = set(my_trade_date['trade_date'])
missing_dates = trade_days_set - my_trade_date_set
# 打印反向检查结果
print(f"\nmy_trade_date范围: {my_trade_date_min} 到 {my_trade_date_max}")
print(f"trade_date在此范围内的日期数量: {len(trade_days_in_range)}")
print(f"在此范围内不在my_trade_date中的日期数量: {len(missing_dates)}")
if missing_dates:
    print("不在my_trade_date中的日期:")
    print(sorted(missing_dates))
