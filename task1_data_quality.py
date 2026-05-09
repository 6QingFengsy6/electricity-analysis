import pandas as pd
import numpy as np

# 读取清洗后的数据（修正：赋值给df）
df = pd.read_csv("data_cleaned.csv", parse_dates=['datetime'])

# 计算总点数（理论值：66户 × 约5856个点）
total_points_per_user = df.groupby('户号').size()
date_range = df['datetime'].dt.date.unique()
expected_per_user = len(date_range) * 96
print(f"数据覆盖天数: {len(date_range)} 天")
print(f"每户应有点数: {len(date_range) * 96}")

# ========== 1. 缺失率体检 ==========
missing_count = df.groupby('户号')['电能量'].apply(lambda x: x.isna().sum())
missing_rate = missing_count / expected_per_user * 100

print("\n=== 缺失率排名（TOP10）===")
missing_report = pd.DataFrame({
    '缺失点数': missing_count,
    '缺失率(%)': missing_rate
}).sort_values('缺失率(%)', ascending=False)

print(missing_report.head(10))

# 预警规则
high_risk = missing_report[missing_report['缺失率(%)'] > 30]
medium_risk = missing_report[(missing_report['缺失率(%)'] > 15) & (missing_report['缺失率(%)'] <= 30)]

print(f"\n⚠️ 严重故障电表（缺失率>30%）：{len(high_risk)} 户 -> {high_risk.index.tolist()}")
print(f"⚠️ 注意观察电表（缺失率15%-30%）：{len(medium_risk)} 户 -> {medium_risk.index.tolist()}")

# ========== 2. 异常值扫描 ==========
# 计算每个用户相邻时刻的电能量差值
df = df.sort_values(['户号', 'datetime'])
df['diff'] = df.groupby('户号')['电能量'].diff()

# 2.1 电表示数倒退（差值 < -0.001，考虑浮点误差）
backward = df[df['diff'] < -0.001]
backward_count = backward.groupby('户号').size()
print(f"\n=== 电表示数倒退情况 ===")
print(f"总倒退次数: {len(backward)}")
print(backward_count[backward_count > 0].sort_values(ascending=False).head(10))

# 2.2 突增检测（差值超过5倍滑动中位数）
# 简化版：取每户差值的99分位数作为阈值
threshold_99 = df.groupby('户号')['diff'].quantile(0.99)
spike = []
for user in df['户号'].unique():
    user_data = df[df['户号'] == user]
    thresh = threshold_99[user]
    spikes = user_data[user_data['diff'] > thresh * 3]  # 3倍99分位数
    spike.append(len(spikes))
print(f"\n突增异常统计（每户）: 最大值 {max(spike)} 次, 平均 {np.mean(spike):.1f} 次")

# 2.3 连续恒定值检测（连续6小时/24个点数值不变）
df['value_round'] = df['电能量'].round(4)  # 四舍五入到小数点后4位
constant_streak = df.groupby('户号')['value_round'].apply(
    lambda x: (x == x.shift(1)).rolling(24).sum().max()
)
print(f"\n=== 连续恒定值检测 ===")
print(constant_streak[constant_streak >= 20].head(10))

print("\n✅ 数据质量体检完成！")

# 画缺失率柱状图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

missing_rate_sorted = missing_rate.sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 5))
plt.bar(missing_rate_sorted.index.astype(str), missing_rate_sorted.values)
plt.xlabel('户号')
plt.ylabel('缺失率 (%)')
plt.title('各户电表缺失率TOP15')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("missing_rate.png")
plt.show()
