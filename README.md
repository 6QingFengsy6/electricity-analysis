# 空巢老人用电数据分析

## 项目简介
针对无锡地区66户空巢老人的用电数据（15分钟采样间隔），进行多维度的数据分析。

## 任务列表
- 数据质量体检（缺失率、异常值扫描）
- 急症风险检测（全天无用电迹象）
- 作息规律分析（聚类+规律性得分）
- 天气数据融合（温度-用电相关性）
- 漏电疑似检测（夜间持续微小功率）

## 运行方法
python task1_data_quality.py
python task2_risk_detection.py
python task3_routine_v2.py
python task4_weather.py
python task5_leakage_detection.py


## 依赖库
pip install pandas numpy matplotlib scikit-learn requests


## 主要结果
- 户号54：急症风险最高 + 作息不规律 + 漏电高风险
- 高温日用电量是舒适日的8.5倍
- 35户存在高风险漏电疑似

## 数据格式要求
将数据文件命名为 data_cleaned.csv，与代码放在同一目录，需包含列：户号、电能量、datetime

## 作者
GitHub: 6QingFengsy6

## 许可证
MIT License
