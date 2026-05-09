import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("data_cleaned.csv", parse_dates=['datetime'])

# 提取时间特征
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date

# 计算每15分钟增量
df = df.sort_values(['户号', 'datetime'])
df['increment'] = df.groupby('户号')['电能量'].diff()
df['increment'] = df['increment'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)

# 定义时段
def get_time_period(hour):
    if 0 <= hour < 6:
        return '深夜(0-6点)'
    elif 6 <= hour < 9:
        return '清晨(6-9点)'
    elif 9 <= hour < 12:
        return '上午(9-12点)'
    elif 12 <= hour < 14:
        return '午间(12-14点)'
    elif 14 <= hour < 18:
        return '下午(14-18点)'
    elif 18 <= hour < 21:
        return '傍晚(18-21点)'
    else:
        return '夜间(21-24点)'

df['时段'] = df['hour'].apply(get_time_period)

# 计算每户每天各时段用电占比
daily_period = df.groupby(['户号', 'date', '时段'])['increment'].sum().reset_index()
# 计算每天总用电
daily_total = df.groupby(['户号', 'date'])['increment'].sum().reset_index()
daily_total.columns = ['户号', 'date', '日总用电']
# 合并
daily_period = daily_period.merge(daily_total, on=['户号', 'date'])
daily_period['占比'] = daily_period['increment'] / daily_period['日总用电']
daily_period['占比'] = daily_period['占比'].fillna(0)

# 计算每户的平均时段用电模式
user_pattern = daily_period.groupby(['户号', '时段'])['占比'].mean().reset_index()

# 计算每户的作息规律性（每天用电模式的相似度）
def calc_routine_score(user_id):
    user_daily = daily_period[daily_period['户号'] == user_id]
    # 计算每天与平均模式的余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 获取该用户的平均模式
    avg_pattern = user_pattern[user_pattern['户号'] == user_id].set_index('时段')['占比']
    periods = avg_pattern.index.tolist()
    
    similarities = []
    for date in user_daily['date'].unique():
        day_data = user_daily[user_daily['date'] == date].set_index('时段')['占比']
        # 补齐缺失时段
        for p in periods:
            if p not in day_data.index:
                day_data[p] = 0
        day_data = day_data.reindex(periods)
        # 计算相似度
        sim = cosine_similarity([avg_pattern.values], [day_data.values])[0][0]
        similarities.append(sim)
    
    return np.mean(similarities)

# 判断作息类型
def get_routine_type_v2(user_id, pattern_df, routine_score):
    # 找出该用户用电最多的时段
    user_data = pattern_df[pattern_df['户号'] == user_id]
    if len(user_data) == 0:
        return "数据不足"
    
    peak_period = user_data.loc[user_data['占比'].idxmax(), '时段']
    
    # 根据规律性得分和高峰时段判断
    if routine_score < 0.5:
        return "作息不规律"
    elif peak_period in ['深夜(0-6点)', '夜间(21-24点)']:
        return "晚睡型（夜间活跃）"
    elif peak_period in ['清晨(6-9点)']:
        return "早睡早起型"
    elif peak_period in ['上午(9-12点)', '下午(14-18点)']:
        return "日间活跃型"
    else:
        return "规律作息"

# 分析所有用户
print("=== 用户作息规律分析 ===")
print("户号\t规律性得分\t作息类型")
print("-" * 50)

routine_results = []
for user_id in df['户号'].unique():
    score = calc_routine_score(user_id)
    rtype = get_routine_type_v2(user_id, user_pattern, score)
    routine_results.append({'户号': user_id, '规律性得分': score, '作息类型': rtype})
    print(f"{user_id}\t{score:.3f}\t{rtype}")

# 统计
result_df = pd.DataFrame(routine_results)
print("\n=== 统计结果 ===")
print(result_df['作息类型'].value_counts())
print(f"\n平均规律性得分: {result_df['规律性得分'].mean():.3f}")
print(f"规律性得分最低的5户（最不规律）:")
print(result_df.nsmallest(5, '规律性得分')[['户号', '规律性得分']])

# 保存结果
result_df.to_csv("routine_result.csv", index=False)
print("\n✅ 结果已保存: routine_result.csv")
