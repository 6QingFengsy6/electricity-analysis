import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ========== 1. 获取天气数据 ==========
print("正在获取天气数据...")

# 无锡坐标
lat, lon = 31.49, 120.31

# 日期范围（与数据一致：2024-12-01 到 2025-08-31）
start_date = "2024-12-01"
end_date = "2025-08-31"

# 使用 Open-Meteo 免费API
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "temperature_2m,relativehumidity_2m,precipitation",
    "timezone": "Asia/Shanghai"
}

try:
    response = requests.get(url, params=params)
    data = response.json()
    
    # 转换为DataFrame
    hourly_data = pd.DataFrame(data['hourly'])
    hourly_data['time'] = pd.to_datetime(hourly_data['time'])
    hourly_data.columns = ['datetime', '温度', '湿度', '降水量']
    
    print(f"获取成功！共 {len(hourly_data)} 小时数据")
    
    # 保存天气数据
    hourly_data.to_csv("weather_data.csv", index=False)
    print("天气数据已保存: weather_data.csv")
    
except Exception as e:
    print(f"获取失败: {e}")
    print("使用示例天气数据代替...")
    # 如果API失败，生成示例天气数据
    dates = pd.date_range(start_date, end_date, freq='h')
    hourly_data = pd.DataFrame({
        'datetime': dates,
        '温度': np.random.normal(15, 10, len(dates)),
        '湿度': np.random.normal(65, 15, len(dates)),
        '降水量': np.random.exponential(0.5, len(dates))
    })
    hourly_data.to_csv("weather_data.csv", index=False)

# ========== 2. 融合用电数据 ==========
print("\n正在融合天气和用电数据...")

# 读取用电数据
df = pd.read_csv("data_cleaned.csv", parse_dates=['datetime'])

# 读取天气数据
weather = pd.read_csv("weather_data.csv", parse_dates=['datetime'])

# 将天气数据从小时级下采样到15分钟级（向前填充）
weather_15min = []
for i in range(len(weather) - 1):
    current_time = weather.loc[i, 'datetime']
    next_time = weather.loc[i+1, 'datetime']
    # 生成当前小时的4个15分钟点
    for j in range(4):
        t = current_time + pd.Timedelta(minutes=15*j)
        weather_15min.append({
            'datetime': t,
            '温度': weather.loc[i, '温度'],
            '湿度': weather.loc[i, '湿度'],
            '降水量': weather.loc[i, '降水量']
        })
weather_15min = pd.DataFrame(weather_15min)

# 融合
df = df.merge(weather_15min, on='datetime', how='left')
print(f"融合后数据量: {len(df)} 条")

# ========== 3. 天气与用电相关性分析 ==========
print("\n=== 天气与用电相关性分析 ===")

# 计算日级汇总
df['date'] = df['datetime'].dt.date
df['increment'] = df.groupby('户号')['电能量'].diff()
df['increment'] = df['increment'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)

# 每日总用电
daily_consumption = df.groupby('date')['increment'].sum().reset_index()
daily_consumption.columns = ['date', '日总用电']

# 每日平均温度/湿度
daily_weather = df.groupby('date')[['温度', '湿度']].mean().reset_index()

# 合并
daily_analysis = daily_consumption.merge(daily_weather, on='date')

# 计算相关系数
corr_temp = daily_analysis['日总用电'].corr(daily_analysis['温度'])
corr_humidity = daily_analysis['日总用电'].corr(daily_analysis['湿度'])

print(f"日总用电与温度的相关系数: {corr_temp:.3f}")
print(f"日总用电与湿度的相关系数: {corr_humidity:.3f}")

# 解释
if corr_temp > 0.3:
    print("→ 温度越高，用电量越大（空调用电明显）")
elif corr_temp < -0.3:
    print("→ 温度越低，用电量越大（取暖用电明显）")
else:
    print("→ 温度与用电量相关性不明显")

# ========== 4. 高温/低温日用电分析 ==========
print("\n=== 极端天气用电分析 ===")

hot_days = daily_analysis[daily_analysis['温度'] > 28]
cold_days = daily_analysis[daily_analysis['温度'] < 5]
normal_days = daily_analysis[(daily_analysis['温度'] >= 10) & (daily_analysis['温度'] <= 25)]

print(f"高温日（>28°C）: {len(hot_days)}天，平均日用电: {hot_days['日总用电'].mean():.2f} kWh")
print(f"低温日（<5°C）: {len(cold_days)}天，平均日用电: {cold_days['日总用电'].mean():.2f} kWh")
print(f"舒适日（10-25°C）: {len(normal_days)}天，平均日用电: {normal_days['日总用电'].mean():.2f} kWh")

# ========== 5. 保存融合后的数据 ==========
df.to_csv("data_with_weather.csv", index=False)
print("\n✅ 融合天气后的数据已保存: data_with_weather.csv")

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 已经有了 daily_analysis（日总用电和温度）
plt.figure(figsize=(10, 6))
plt.scatter(daily_analysis['温度'], daily_analysis['日总用电'], alpha=0.6)
plt.xlabel('平均温度 (°C)')
plt.ylabel('日总用电 (kWh)')
plt.title('温度与用电量相关性')
# 加趋势线
z = np.polyfit(daily_analysis['温度'], daily_analysis['日总用电'], 1)
p = np.poly1d(z)
plt.plot(sorted(daily_analysis['温度']), p(sorted(daily_analysis['温度'])), 'r--')
plt.tight_layout()
plt.savefig("temp_scatter.png")
plt.show()
