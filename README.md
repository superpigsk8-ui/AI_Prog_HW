## 安装依赖:

	import sys
	import os
	import pandas
	import numpy
	import matplotlib.pyplot 
	import seaborn 
	import os
	import geopandas 
	from matplotlib.patches import Patch
	import torch
	from torch.utils.data import DataLoader, TensorDataset
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler, LabelEncoder
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import mean_absolute_error, mean_squared_error
	import random
	import time
	import re
	from datetime import datetime
	python3.9+

## 操作说明:

	main文件的DATA_PATH修改为yellow_tripdata_2026-01.parquet所在位置
	main文件的SHP_DIR修改为taxi_zones.shp所在位置
	taxi_zones.shp,taxi_zones.shx,taxi_zones.prj,taxi_zones.dbf,taxi_zones.cpg应处于同一目录
	运行时直接运行main文件

# 城市出租车出行数据分析与智能问答系统

> 纽约市黄色出租车行程数据分析（2026年1月，约372万条记录）

## 项目简介

本项目构建了一个完整的出租车出行数据分析与智能问答系统，包含四个核心功能模块：

- **M1 数据处理**：数据加载、清洗、时间特征提取、衍生特征创建
- **M2 分析可视化**：出行需求时间规律、区域热度分析、车费影响因素分析、自选分析
- **M3 预测模型**：PyTorch神经网络 + 随机森林对比，预测出行需求量
- **M4 问答接口**：命令行交互式问答系统，支持5种问题类型

## 项目结构
final_project/
├── data/ # 数据文件目录
│ ├── yellow_tripdata_2026-01.parquet
│ └── taxi_zones.shp
├── outputs/ # 输出文件目录
│ ├── data_quality_report.csv # 数据质量报告
│ ├── m2_1_hourly_demand.png # M2-1 出行需求时间规律
│ ├── m2_2_region_analysis.png # M2-2 区域热度分析
│ ├── m2_3_fare_analysis.png # M2-3 车费影响因素分析
│ ├── m2_4_payment_analysis.png # M2-4 付款方式分析（自选）
│ ├── m3_neural_network_loss.png # M3 神经网络训练损失曲线
│ └── m3_model_metrics.csv # M3 模型评估指标
├── src/ # 源代码目录
│ ├── m1_data_processing.py # 数据加载、清洗、特征工程
│ ├── m2_visualization.py # 可视化分析
│ ├── m3_modeling.py # 神经网络 + 随机森林建模
│ └── m4_qa_system.py # 命令行问答系统
├── main.py # 主入口函数
├── requirements.txt # 第三方依赖
└── README.md # 项目说明

text

## 安装与运行

### 环境要求

- Python 3.9+
- 建议使用虚拟环境

### 安装依赖

```bash
pip install -r requirements.txt
数据准备
从 TLC Trip Record Data 下载 2026年1月的黄色出租车行程数据

下载 taxi_zones.shp 地理数据文件

将两个文件放置于 data/ 目录下

运行项目
bash
python main.py
程序将自动执行：

数据加载与清洗

生成数据质量报告

生成所有可视化图表

训练神经网络和随机森林模型

启动命令行问答系统

问答系统使用示例
text
🔍 请输入您的问题: 早上8点订单量多少？
📊 结论: 8点订单量: 15,234 单, 平均车费: $12.50
💡 说明: 在8点时段，共有15,234笔订单，平均车费为$12.50。
📁 相关文件: outputs/m2_1_hourly_demand.png

🔍 请输入您的问题: 订单量最高的前5个区域是哪些？
📊 结论: 订单量TOP5区域: 区域1(45,678单), 区域2(32,123单), ...
💡 说明: 订单量最高的5个区域如上所示，其中区域1最为繁忙。
📁 相关文件: outputs/m2_2_region_analysis.png

🔍 请输入您的问题: 预测下午3点的订单需求
📊 结论: 预测15点订单需求: 预计小费 $3.45, 过路费 $1.23
💡 说明: 基于神经网络模型预测，15点时段的平均小费约为$3.45，过路费约为$1.23。
📁 相关文件: outputs/m3_neural_network_loss.png, outputs/m3_model_metrics.csv
模块功能说明
M1 数据处理 (m1_data_processing.py)
加载 Parquet 数据文件

13步数据清洗策略（删除重复、处理异常值、编码分类变量等）

时间特征提取：小时、星期、是否周末、是否高峰、高峰类型

3个衍生特征：行程时长(分钟)、平均速度(mph)、每英里费用($/mile)

输出：outputs/data_quality_report.csv

M2 分析可视化 (m2_visualization.py)
子任务	内容	输出文件
M2-1	工作日/周末分小时订单量对比折线图	m2_1_hourly_demand.png
M2-2	TOP10区域柱状图 + 小时热力图 + 分级设色地图	m2_2_region_analysis.png
M2-3	距离-车费散点图 + 时段中位车费 + 乘客人数箱线图	m2_3_fare_analysis.png
M2-4	付款方式分析（自选）	m2_4_payment_analysis.png
M3 预测模型 (m3_modeling.py)
特征设计：15个输入特征（时间3个 + 行程6个 + 费用6个）

数据划分：8:2训练/测试集，随机种子42保证可复现

神经网络：PyTorch多输出回归网络 (15→128→64→2)

对比模型：随机森林 (n_estimators=100, max_depth=20)

评估指标：MAE、RMSE、R²

输出：m3_neural_network_loss.png、m3_model_metrics.csv

M4 问答接口 (m4_qa_system.py)
支持5种问题类型：

时段查询：查询某时段订单量/平均费用

区域排名：查询订单量TOP N区域

需求预测：使用神经网络模型预测小费和过路费

费用估算：估算行程费用

人数估算：查询平均乘客人数

第三方依赖
详见 requirements.txt：

pandas>=1.5.0

numpy>=1.23.0

matplotlib>=3.6.0

seaborn>=0.12.0

geopandas>=0.12.0

torch>=2.0.0

scikit-learn>=1.2.0

pyarrow>=10.0.0