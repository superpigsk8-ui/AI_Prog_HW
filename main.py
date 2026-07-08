
import sys
import os
import pandas as pd

# 将src目录添加到路
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from m1_data_processing import load_and_process_data
from m2_visualization import run_all_visualizations
from m3_modeling import run_modeling_pipeline
from m4_qa_system import run_qa_system


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" 纽约出租车数据分析系统")
    print("="*70)
    
    # ---- 配置 ----
    # 请修改为您的实际文件路径
    DATA_PATH = 'data//yellow_tripdata_2026-01.parquet'  
    SHP_DIR = 'data'  # taxi_zones.shp 所在目录
    
    # 检查文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"\n错误: 数据文件不存在: {DATA_PATH}")
        print("   请修改 main.py 中的 DATA_PATH 变量")
        return
    
    print(f"\n  数据文件: {DATA_PATH}")
    print(f"  Shapefile目录: {SHP_DIR}")
    
    # ---- 1. 数据处理 ----
    df = load_and_process_data(DATA_PATH, save_report=True)
    
    # ---- 2. 可视化 ----
    run_all_visualizations(df, SHP_DIR)
    
    # ---- 3. 建模 ----
    model_results = run_modeling_pipeline(df)
    
    # ---- 4. 问答系统 ----
    run_qa_system(df, model_results)
    
 


if __name__ == "__main__":
    main()