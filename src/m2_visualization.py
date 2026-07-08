
"""
m2_visualization.py - 可视化分析   
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from matplotlib.patches import Patch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_all_visualizations(df, shp_dir=None):
    """

    参数:
        df: 清洗后的DataFrame
        shp_dir: taxi_zones.shp 所在目录
    """
    print("\n" + "="*70)
    print("m2_visualization: 运行所有可视化")
    print("="*70)
    
    os.makedirs('outputs', exist_ok=True)
    
    # 出行需求时间规律
    plot_hourly_demand(df)
    
    # 区域热度分析
    if shp_dir and os.path.exists(os.path.join(shp_dir, 'taxi_zones.shp')):
        plot_region_analysis(df, shp_dir)
    else:
        print(" 未找到taxi_zones.shp，跳过区域热度分析")
    
    # 车费影响因素分析
    plot_fare_analysis(df)
    
    print("="*70)


def plot_hourly_demand(df):
    """工作日/周末分小时订单量对比折线图"""
    print("\n  出行需求时间规律...")
    
    hourly_counts = df.groupby(['pickup_hour', 'pickup_is_weekend']).size().reset_index(name='trip_count')
    
    weekday_data = hourly_counts[hourly_counts['pickup_is_weekend'] == False].copy()
    weekend_data = hourly_counts[hourly_counts['pickup_is_weekend'] == True].copy()
    
    all_hours = pd.DataFrame({'pickup_hour': range(24)})
    weekday_data = all_hours.merge(weekday_data, on='pickup_hour', how='left').fillna(0)
    weekend_data = all_hours.merge(weekend_data, on='pickup_hour', how='left').fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(weekday_data['pickup_hour'], weekday_data['trip_count'], 
            marker='o', linewidth=2.5, markersize=6, 
            color='#2E86AB', label='工作日 (周一至周五)')
    ax.plot(weekend_data['pickup_hour'], weekend_data['trip_count'], 
            marker='s', linewidth=2.5, markersize=6, 
            color='#A23B72', label='周末 (周六至周日)')
    
    ax.set_title('工作日 vs 周末 分小时订单量对比', fontsize=16, fontweight='bold')
    ax.set_xlabel('小时 (0-23)', fontsize=13)
    ax.set_ylabel('订单量', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24, 1))
    ax.set_xticklabels(range(0, 24, 1))
    ax.axvspan(7, 9, alpha=0.15, color='blue', label='早高峰')
    ax.axvspan(16, 18, alpha=0.15, color='orange', label='晚高峰')
    
    plt.tight_layout()
    plt.savefig('outputs/m2_1_hourly_demand.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: outputs/m2_1_hourly_demand.png")


def plot_region_analysis(df, shp_dir):
    """区域热度分析"""
    print("\n  区域热度分析...")
    
    shp_path = os.path.join(shp_dir, 'taxi_zones.shp')
    zones = gpd.read_file(shp_path)
    
    # 图表1: TOP 10 上下客区域柱状图
    pu_counts = df['PULocationID'].value_counts().head(10).reset_index()
    pu_counts.columns = ['LocationID', 'pickup_count']
    do_counts = df['DOLocationID'].value_counts().head(10).reset_index()
    do_counts.columns = ['LocationID', 'dropoff_count']
    
    zone_names = zones[['LocationID', 'zone']].drop_duplicates()
    pu_counts = pu_counts.merge(zone_names, on='LocationID', how='left')
    do_counts = do_counts.merge(zone_names, on='LocationID', how='left')
    
    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1 = axes[0]
    bars1 = ax1.barh(pu_counts['zone'], pu_counts['pickup_count'], color='#2E86AB')
    ax1.set_xlabel('上客订单量', fontsize=12)
    ax1.set_title('上客量最高的 TOP 10 区域', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars1, pu_counts['pickup_count']):
        ax1.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f}', va='center', fontsize=9)
    
    ax2 = axes[1]
    bars2 = ax2.barh(do_counts['zone'], do_counts['dropoff_count'], color='#A23B72')
    ax2.set_xlabel('下客订单量', fontsize=12)
    ax2.set_title('下客量最高的 TOP 10 区域', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars2, do_counts['dropoff_count']):
        ax2.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/m2_2_top10_regions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" 已保存: outputs/m2_2_top10_regions.png")
    
    # 图表2: 热门上车区域的小时订单量热力图
    top15_pu = df['PULocationID'].value_counts().head(15).index.tolist()
    df_top15 = df[df['PULocationID'].isin(top15_pu)]
    heatmap_data = df_top15.groupby(['PULocationID', 'pickup_hour']).size().reset_index(name='trip_count')
    pivot = heatmap_data.pivot(index='PULocationID', columns='pickup_hour', values='trip_count').fillna(0)
    
    zone_names_dict = zones.set_index('LocationID')['zone'].to_dict()
    pivot.index = pivot.index.map(lambda x: zone_names_dict.get(x, str(x))[:20])
    
    fig2, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', 
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': '订单量'})
    ax.set_xlabel('小时 (0-23)', fontsize=13)
    ax.set_ylabel('区域', fontsize=13)
    ax.set_title('TOP 15 热门上车区域 × 小时 订单量热力图', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/m2_2_hourly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" 已保存: outputs/m2_2_hourly_heatmap.png")
    
    # 图表3: 区域分级设色地图
    pu_total = df['PULocationID'].value_counts().reset_index()
    pu_total.columns = ['LocationID', 'pickup_count']
    zones_map = zones.merge(pu_total, on='LocationID', how='left')
    zones_map['pickup_count'] = zones_map['pickup_count'].fillna(0)
    zones_map['quantile_rank'] = pd.qcut(zones_map['pickup_count'], 
                                         q=5, labels=['最低', '较低', '中等', '较高', '最高'])
    
    fig3, ax = plt.subplots(figsize=(14, 12))
    zones_map.plot(column='quantile_rank', cmap='OrRd', edgecolor='white',
                   linewidth=0.3, legend=True,
                   legend_kwds={'title': '上客量等级', 'loc': 'lower right'},
                   ax=ax)
    ax.set_title('纽约市出租车区域分级设色地图\n(基于上客量)', fontsize=16, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig('outputs/m2_2_choropleth_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: outputs/m2_2_choropleth_map.png")


def plot_fare_analysis(df):
    """车费影响因素分析"""
    print("\n  车费影响因素分析...")
    
    plot_df = df[(df['trip_distance'] > 0) & (df['trip_distance'] <= 50)]
    plot_df = plot_df[(plot_df['fare_amount'] > 0) & (plot_df['fare_amount'] <= 150)]
    
    peak_labels = {'morning_peak': '早高峰 (7-9点)', 'evening_peak': '晚高峰 (16-18点)', 'off_peak': '非高峰'}
    plot_df['peak_label'] = plot_df['pickup_peak_type'].map(peak_labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0]
    axes[1, 1].axis('off')
    
    # 子图1: 散点图
    passenger_groups = sorted([p for p in plot_df['passenger_count'].dropna().unique() if p > 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(passenger_groups)))
    for i, p_count in enumerate(passenger_groups):
        subset = plot_df[plot_df['passenger_count'] == p_count]
        if len(subset) > 8000:
            subset = subset.sample(n=8000, random_state=42)
        ax1.scatter(subset['trip_distance'], subset['fare_amount'], 
                   alpha=0.4, s=6, color=colors[i], 
                   label=f'{int(p_count)}人' if p_count == int(p_count) else f'{p_count}人')
    ax1.set_xlabel('行程距离 (英里)', fontsize=12)
    ax1.set_ylabel('车费 (美元)', fontsize=12)
    ax1.set_title('行程距离与车费关系散点图', fontsize=13, fontweight='bold')
    ax1.legend(title='乘客人数', fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 中位车费柱状图
    peak_fare = plot_df.groupby('peak_label')['fare_amount'].median().reset_index()
    order = ['早高峰 (7-9点)', '晚高峰 (16-18点)', '非高峰']
    peak_fare['peak_label'] = pd.Categorical(peak_fare['peak_label'], categories=order, ordered=True)
    peak_fare = peak_fare.sort_values('peak_label')
    colors_bar = ['#E63946', '#F4A261', '#2A9D8F']
    bars = ax2.bar(peak_fare['peak_label'], peak_fare['fare_amount'], 
                  color=colors_bar, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, peak_fare['fare_amount']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'${val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('中位车费 (美元)', fontsize=12)
    ax2.set_title('不同时段的中位车费对比', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 子图3: 箱线图
    box_df = plot_df[plot_df['passenger_count'].between(1, 6)]
    sns.boxplot(data=box_df, x='passenger_count', y='fare_amount', 
                palette='viridis', ax=ax3, showfliers=False)
    mean_fare = box_df.groupby('passenger_count')['fare_amount'].mean().reset_index()
    ax3.scatter(mean_fare['passenger_count'], mean_fare['fare_amount'], 
               color='red', s=70, zorder=5, marker='D', label='均值')
    ax3.set_xlabel('乘客人数', fontsize=12)
    ax3.set_ylabel('车费 (美元)', fontsize=12)
    ax3.set_title('不同乘客人数对应的车费分布', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    fig.suptitle('车费影响因素分析', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('outputs/m2_3_fare_analysis_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" 已保存: outputs/m2_3_fare_analysis_combined.png")

'''
  分析单位行程距离的车费（每英里费用），从时段、区域和行程距离三个维度揭示费用结构的深层规律。
  在可视化设计上，采用2×2子图布局（共3个子图，空一格）将三项分析整合为一张图片
  m2_4_fare_per_mile_analysis.png：子图1为不同时段（早高峰/晚高峰/非高峰）的每英里费用柱状图，用于检验拥堵
  程度对单位距离费用的影响——高峰时段因低速行驶导致计价器时间费用累积，理论上每英里费用应高于非高峰；子图2为
  每英里费用最高的TOP10区域横向柱状图，用于识别费用异常偏高的热点区域（如机场、CBD等），并标注各区域样本量以确
  保统计可靠性；子图3为行程距离与每英里费用的散点图叠加距离分组的均值趋势线，用于揭示短途行程因包含固定起步价
  而导致单位距离费用虚高、长途行程则因固定成本被摊薄而趋于稳定的规律。 

'''