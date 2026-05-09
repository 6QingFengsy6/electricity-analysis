import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("data_cleaned.csv", parse_dates=['datetime'])

# 计算功率
df = df.sort_values(['户号', 'datetime'])
df['power_w'] = df.groupby('户号')['电能量'].diff() * 4000
df['power_w'] = df['power_w'].clip(lower=0).fillna(0)

# 时间特征
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date
df['is_night'] = (df['hour'] >= 23) | (df['hour'] < 5)

results = []
leak_users_list = []  # 记录漏电用户

for user_id in df['户号'].unique():
    user_data = df[df['户号'] == user_id].copy()
    user_data = user_data.sort_values('datetime')
    
    # 夜间基础功率
    night_data = user_data[user_data['is_night']]
    base_power = night_data['power_w'].quantile(0.25) if len(night_data) > 0 else 0
    
    # 漏电判断
    night_data['is_stable'] = (night_data['power_w'] >= 10) & (night_data['power_w'] <= 50)
    night_data['power_change'] = night_data['power_w'].diff().abs()
    low_variance = night_data['power_change'].median() < 5 if len(night_data) > 1 else False
    
    # 找连续稳定时段
    night_data['stable_streak'] = (night_data['is_stable'] != night_data['is_stable'].shift(1)).cumsum()
    streak_lengths = night_data[night_data['is_stable']].groupby('stable_streak').size()
    
    has_leak = (streak_lengths >= 12).any() and low_variance
    max_streak = streak_lengths.max() / 4 if len(streak_lengths) > 0 else 0
    
    if has_leak:
        leak_users_list.append(user_id)
    
    results.append({
        '户号': user_id,
        '疑似漏电': '是' if has_leak else '否',
        '最长连续稳定功率(小时)': round(max_streak, 1),
        '夜间基础功率(W)': round(base_power, 1),
        '漏电风险': '高' if max_streak > 6 else ('中' if max_streak > 3 else '低') if has_leak else '无'
    })

result_df = pd.DataFrame(results)

print("=== 漏电检测结果 ===")
print(result_df['疑似漏电'].value_counts())
print("\n漏电风险分布:")
print(result_df['漏电风险'].value_counts())

# ========== 画图部分 ==========
if len(leak_users_list) > 0:
    # 选第一个漏电用户画图
    leak_user = leak_users_list[0]
    print(f"\n正在为户号 {leak_user} 绘制漏电检测图...")
    
    user_data = df[df['户号'] == leak_user].copy()
    user_data = user_data.sort_values('datetime')
    
    # 选一个有夜间稳定功率的日期
    user_data['night_stable'] = user_data['is_night'] & (user_data['power_w'] >= 10) & (user_data['power_w'] <= 50)
    sample_dates = user_data[user_data['night_stable']]['datetime'].dt.date.unique()
    
    if len(sample_dates) > 0:
        sample_date = sample_dates[0]
        one_day = user_data[user_data['datetime'].dt.date == sample_date]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(one_day['datetime'], one_day['power_w'], 'b-', linewidth=1, label='功率')
        
        # 标记漏电区域
        leak_periods = one_day[one_day['is_night'] & (one_day['power_w'] >= 10) & (one_day['power_w'] <= 50)]
        if len(leak_periods) > 0:
            ax.scatter(leak_periods['datetime'], leak_periods['power_w'], 
                      c='red', s=30, label='疑似漏电时段', zorder=5)
        
        # 标记夜间时段
        night_start = pd.Timestamp(sample_date).replace(hour=23)
        night_end = pd.Timestamp(sample_date).replace(hour=5) + pd.Timedelta(days=1)
        ax.axvspan(night_start, night_end, alpha=0.2, color='gray', label='夜间时段(23-5点)')
        
        # 漏电阈值线
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='漏电上限(50W)')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='漏电下限(10W)')
        
        ax.set_xlabel('时间')
        ax.set_ylabel('功率 (W)')
        ax.set_title(f'户号{leak_user} 夜间功率曲线（漏电检测示例）')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("leakage_detection.png", dpi=150)
        print("✅ 漏电检测图已保存: leakage_detection.png")
        plt.show()
    else:
        print("未找到合适的日期绘制漏电图")
else:
    print("没有疑似漏电用户，无法绘制示例图")

# 保存结果
result_df.to_csv("leakage_result.csv", index=False)
print("\n✅ 漏电检测结果已保存: leakage_result.csv")
