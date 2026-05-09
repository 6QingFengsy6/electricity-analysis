import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("data_cleaned.csv", parse_dates=['datetime'])

# 提取日期
df['date'] = df['datetime'].dt.date

# 按户号和时间排序
df = df.sort_values(['户号', 'datetime'])

# 计算每15分钟的电量增量
df['increment'] = df.groupby('户号')['电能量'].diff()
df['increment'] = df['increment'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)

# 按户和日期汇总
daily_consumption = df.groupby(['户号', 'date'])['increment'].sum().reset_index()
daily_consumption.columns = ['户号', 'date', '日用电量']

daily_volatility = df.groupby(['户号', 'date'])['increment'].std().reset_index()
daily_volatility.columns = ['户号', 'date', '波动率']

daily = daily_consumption.merge(daily_volatility, on=['户号', 'date'])


# 先计算每户的平均日用电量（用于个性化阈值）
user_avg = daily.groupby('户号')['日用电量'].mean().to_dict()

def risk_level_v2(row):
    consumption = row['日用电量']
    volatility = row['波动率'] if pd.notna(row['波动率']) else 0
    user = row['户号']
    avg_consump = user_avg.get(user, 0.5)
    
    # 如果该用户平时用电就很低（平均值<0.3），调整判断标准
    if avg_consump < 0.3:
        if consumption < 0.05 and volatility < 0.01:
            return '高风险（全天无用电）'
        elif consumption < avg_consump * 0.5:
            return '中风险（用电异常偏低）'
        else:
            return '正常（低用电用户）'
    else:
        # 正常用电用户
        if consumption < 0.1 and volatility < 0.02:
            return '高风险（全天无用电）'
        elif consumption < 0.3:
            return '中风险（用电偏少）'
        else:
            return '正常'

daily['风险等级'] = daily.apply(risk_level_v2, axis=1)

# 输出结果
high_risk_days = daily[daily['风险等级'] == '高风险（全天无用电）']
print(f"=== 急症风险检测结果 ===")
print(f"总检测天数: {daily['date'].nunique()} 天")
print(f"高风险天数: {len(high_risk_days)} 天")
print(f"涉及户数: {high_risk_days['户号'].nunique()} 户")

print("\n=== 高风险用户详情 ===")
print(high_risk_days[['户号', 'date', '日用电量', '波动率', '风险等级']].head(20))

risk_summary = high_risk_days.groupby('户号').size().sort_values(ascending=False)
print("\n=== 高风险天数排名 ===")
print(risk_summary.head(10))

# 统计各类风险分布
print("\n=== 风险等级分布 ===")
print(daily['风险等级'].value_counts())

# 保存结果（改：相对路径）
daily.to_csv("risk_result.csv", index=False)
print("\n✅ 结果已保存")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

risk_counts = daily[daily['风险等级'] == '高风险（全天无用电）']['户号'].value_counts().head(10)
plt.figure(figsize=(10, 6))
plt.barh(risk_counts.index.astype(str), risk_counts.values)
plt.xlabel('高风险天数')
plt.ylabel('户号')
plt.title('急症风险排名TOP10')
plt.tight_layout()
plt.savefig("risk_top10.png")
plt.show()
