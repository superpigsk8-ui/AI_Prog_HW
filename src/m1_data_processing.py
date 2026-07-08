
"""
m1_data_processing.py - 数据加载、清洗、特征工程

功能：
    1. 加载parquet数据
    2. 数据清洗（13步策略，每步注释理由）
    3. 时间特征提取（小时、星期、周末、高峰）
    4. 衍生特征创建（行程时长、平均速度、每英里费用）
    5. 数据质量报告生成
"""

import pandas as pd
import numpy as np
import os


def load_and_process_data(file_path, save_report=True):

    print("\n" + "="*70)
    print("m1_data_processing: 数据加载与处理")
    print("="*70)
    
    # ---- 1. 加载数据 ----
    print(f"\n  加载数据: {file_path}")
    df_raw = pd.read_parquet(file_path)
    print(f"  原始数据: {len(df_raw):,} 行, {len(df_raw.columns)} 列")
    
    # ---- 2. 数据清洗 ----
    print("\n  开始数据清洗...")
    df_clean = _clean_tlc_trip_data(df_raw)
    
    # ---- 3. 时间特征提取 ----
    print("\n  提取时间特征...")
    df_time = _extract_time_features(df_clean)
    
    # ---- 4. 衍生特征创建 ----
    print("\n  创建衍生特征...")
    df_final = _create_derived_features(df_time)
    
    # ---- 5. 数据质量报告 ----
    if save_report:
        print("\n  生成数据质量报告...")
        _generate_quality_report(df_final, 'outputs/data_quality_report.csv')
    
    print(f"\n数据处理完成！最终数据: {len(df_final):,} 行, {len(df_final.columns)} 列")
    print("="*70)
    
    return df_final


def _clean_tlc_trip_data(df):
    """数据清洗 - 13步策略"""
    df_clean = df.copy()
    
    # 步骤1: 删除完全重复的行
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dup_count = initial_count - len(df_clean)
    print(f"    步骤1: 删除完全重复行 {dup_count:,} 条")
    
    # 步骤2: 处理日期时间字段
    for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # 步骤3: 删除时间逻辑异常的记录
    valid_time_mask = df_clean['tpep_dropoff_datetime'] > df_clean['tpep_pickup_datetime']
    time_invalid_count = (~valid_time_mask).sum()
    df_clean = df_clean[valid_time_mask]
    print(f"    步骤3: 删除时间逻辑异常记录 {time_invalid_count:,} 条")
    
    # 步骤4: 删除关键字段缺失的记录
    critical_cols = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 
                     'PULocationID', 'DOLocationID']
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_cols)
    null_critical_count = before_drop - len(df_clean)
    print(f"    步骤4: 删除关键字段缺失记录 {null_critical_count:,} 条")
    
    # 步骤5: 处理passenger_count异常值
    passenger_mask = (df_clean['passenger_count'] <= 0) | (df_clean['passenger_count'] > 6)
    passenger_outlier_count = passenger_mask.sum()
    df_clean.loc[passenger_mask, 'passenger_count'] = np.nan
    print(f"    步骤5: 将passenger_count异常值({passenger_outlier_count:,}条)替换为NaN")
    
    # 步骤6: 处理trip_distance异常值
    dist_mask = (df_clean['trip_distance'] <= 0) | (df_clean['trip_distance'] > 100)
    dist_outlier_count = dist_mask.sum()
    df_clean.loc[dist_mask, 'trip_distance'] = np.nan
    print(f"    步骤6: 将trip_distance异常值({dist_outlier_count:,}条)替换为NaN")
    
    # 步骤7: 处理fare_amount异常值
    fare_mask = (df_clean['fare_amount'] < 0) | (df_clean['fare_amount'] > 500)
    fare_outlier_count = fare_mask.sum()
    df_clean.loc[fare_mask, 'fare_amount'] = np.nan
    print(f"    步骤7: 将fare_amount异常值({fare_outlier_count:,}条)替换为NaN")
    
    # 步骤8: 处理tip_amount异常值
    tip_mask = df_clean['tip_amount'] < 0
    tip_outlier_count = tip_mask.sum()
    df_clean.loc[tip_mask, 'tip_amount'] = np.nan
    print(f"    步骤8: 将tip_amount异常值({tip_outlier_count:,}条)替换为NaN")
    
    # 步骤9: 处理其他费用字段的负值
    fee_cols = ['extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge',
                'congestion_surcharge', 'airport_fee', 'cbd_congestion_fee']
    for col in fee_cols:
        if col in df_clean.columns:
            neg_mask = df_clean[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                df_clean.loc[neg_mask, col] = 0
                print(f"    步骤9: 将{col}的负值({neg_count:,}条)替换为0")
    
    # 步骤10: 处理分类变量的非法值
    vendor_mask = ~df_clean['VendorID'].isin([1, 2, 6, 7])
    vendor_invalid_count = vendor_mask.sum()
    df_clean.loc[vendor_mask, 'VendorID'] = np.nan
    print(f"    步骤10: 将VendorID非法值({vendor_invalid_count:,}条)替换为NaN")
    
    rate_mask = ~df_clean['RatecodeID'].isin([1, 2, 3, 4, 5, 6, 99])
    rate_invalid_count = rate_mask.sum()
    df_clean.loc[rate_mask, 'RatecodeID'] = np.nan
    print(f"    步骤10: 将RatecodeID非法值({rate_invalid_count:,}条)替换为NaN")
    
    payment_mask = ~df_clean['payment_type'].isin([0, 1, 2, 3, 4, 5, 6])
    payment_invalid_count = payment_mask.sum()
    df_clean.loc[payment_mask, 'payment_type'] = np.nan
    print(f"    步骤10: 将payment_type非法值({payment_invalid_count:,}条)替换为NaN")
    
    flag_mask = ~df_clean['store_and_fwd_flag'].isin(['Y', 'N'])
    flag_invalid_count = flag_mask.sum()
    df_clean.loc[flag_mask, 'store_and_fwd_flag'] = np.nan
    print(f"    步骤10: 将store_and_fwd_flag非法值({flag_invalid_count:,}条)替换为NaN")
    
    # 步骤11: 处理LocationID异常值
    loc_mask = (df_clean['PULocationID'] < 1) | (df_clean['PULocationID'] > 263)
    loc_invalid_count = loc_mask.sum()
    df_clean.loc[loc_mask, 'PULocationID'] = np.nan
    print(f"    步骤11: 将PULocationID非法值({loc_invalid_count:,}条)替换为NaN")
    
    loc_mask2 = (df_clean['DOLocationID'] < 1) | (df_clean['DOLocationID'] > 263)
    loc_invalid_count2 = loc_mask2.sum()
    df_clean.loc[loc_mask2, 'DOLocationID'] = np.nan
    print(f"    步骤11: 将DOLocationID非法值({loc_invalid_count2:,}条)替换为NaN")
    
    # 步骤12: 重新计算total_amount
    total_cols = ['fare_amount', 'extra', 'mta_tax', 'tolls_amount', 
                  'improvement_surcharge', 'congestion_surcharge', 'airport_fee']
    existing_total_cols = [c for c in total_cols if c in df_clean.columns]
    df_clean['total_amount_recalc'] = df_clean[existing_total_cols].sum(axis=1, skipna=False)
    print(f"    步骤12: 新增total_amount_recalc列（重新计算的总额）")
    
    # 步骤13: 删除清洗后仍存在关键字段缺失的记录
    critical_cols_after = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                           'PULocationID', 'DOLocationID']
    before_final = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_cols_after)
    final_drop_count = before_final - len(df_clean)
    print(f"    步骤13: 删除清洗后关键字段仍缺失的记录 {final_drop_count:,} 条")
    
    return df_clean


def _extract_time_features(df):
    """从 tpep_pickup_datetime 提取时间特征"""
    df_time = df.copy()
    df_time['tpep_pickup_datetime'] = pd.to_datetime(df_time['tpep_pickup_datetime'], errors='coerce')
    
    df_time['pickup_hour'] = df_time['tpep_pickup_datetime'].dt.hour
    df_time['pickup_weekday'] = df_time['tpep_pickup_datetime'].dt.weekday
    
    weekday_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 
                   4: '周五', 5: '周六', 6: '周日'}
    df_time['pickup_weekday_name'] = df_time['pickup_weekday'].map(weekday_map)
    
    df_time['pickup_is_weekend'] = df_time['pickup_weekday'].isin([5, 6])
    
    morning_peak = (df_time['pickup_hour'] >= 7) & (df_time['pickup_hour'] <= 9)
    evening_peak = (df_time['pickup_hour'] >= 16) & (df_time['pickup_hour'] <= 18)
    df_time['pickup_is_peak'] = morning_peak | evening_peak
    
    df_time['pickup_peak_type'] = 'off_peak'
    df_time.loc[morning_peak, 'pickup_peak_type'] = 'morning_peak'
    df_time.loc[evening_peak, 'pickup_peak_type'] = 'evening_peak'
    
    print(f"    新增时间特征: pickup_hour, pickup_weekday, pickup_weekday_name, pickup_is_weekend, pickup_is_peak, pickup_peak_type")
    return df_time


def _create_derived_features(df):
    """创建3个衍生特征"""
    df_feat = df.copy()
    df_feat['tpep_pickup_datetime'] = pd.to_datetime(df_feat['tpep_pickup_datetime'], errors='coerce')
    df_feat['tpep_dropoff_datetime'] = pd.to_datetime(df_feat['tpep_dropoff_datetime'], errors='coerce')
    
    # 衍生特征1: 行程时长（分钟）
    df_feat['trip_duration_minutes'] = (df_feat['tpep_dropoff_datetime'] - df_feat['tpep_pickup_datetime']).dt.total_seconds() / 60
    df_feat.loc[df_feat['trip_duration_minutes'] <= 0, 'trip_duration_minutes'] = np.nan
    
    # 衍生特征2: 平均速度（英里/小时）
    df_feat['avg_speed_mph'] = df_feat['trip_distance'] / (df_feat['trip_duration_minutes'] / 60)
    df_feat.loc[(df_feat['avg_speed_mph'] <= 0) | (df_feat['avg_speed_mph'] > 200), 'avg_speed_mph'] = np.nan
    
    # 衍生特征3: 每英里费用（美元/英里）
    df_feat['fare_per_mile'] = df_feat['fare_amount'] / df_feat['trip_distance']
    df_feat.loc[(df_feat['fare_per_mile'] <= 0) | (df_feat['fare_per_mile'] > 100), 'fare_per_mile'] = np.nan
    
    print(f"    新增衍生特征: trip_duration_minutes, avg_speed_mph, fare_per_mile")
    return df_feat


def _generate_quality_report(df, filename):
    """生成数据质量报告"""
    exclude_cols = ['trip_duration_minutes', 'avg_speed_mph', 'fare_per_mile']
    report_data = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        # 获取列的数据类型
        dtype = df[col].dtype
        
        # 基础信息
        col_info = {
            'column_name': col,
            'data_type': str(dtype),
            'total_count': len(df),
            'non_null_count': df[col].count(),
            'null_count': df[col].isna().sum(),
            'null_rate': f"{df[col].isna().sum() / len(df) * 100:.2f}%",
            'unique_count': df[col].nunique(),
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'q25': None,
            'q50': None,
            'q75': None,
            'outlier_count': 0,
            'outlier_rate': "0.00%",
            'time_invalid_count': None,
            'time_invalid_rate': None
        }
        
        # 安全处理：只对数值类型计算统计量
        try:
            if pd.api.types.is_numeric_dtype(dtype):
                # 过滤掉无穷大值
                numeric_data = df[col].replace([np.inf, -np.inf], np.nan)
                
                if numeric_data.count() > 0:
                    col_info['min'] = numeric_data.min()
                    col_info['max'] = numeric_data.max()
                    col_info['mean'] = numeric_data.mean()
                    col_info['std'] = numeric_data.std()
                    col_info['q25'] = numeric_data.quantile(0.25)
                    col_info['q50'] = numeric_data.quantile(0.50)
                    col_info['q75'] = numeric_data.quantile(0.75)
        except Exception as e:
            # 如果计算失败，保持默认值
            pass
        
        #安全处理：时间字段检查
        try:
            if col == 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' in df.columns:
                pickup = df['tpep_pickup_datetime']
                dropoff = df['tpep_dropoff_datetime']
                
                # 确保是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(pickup):
                    pickup = pd.to_datetime(pickup, errors='coerce')
                if not pd.api.types.is_datetime64_any_dtype(dropoff):
                    dropoff = pd.to_datetime(dropoff, errors='coerce')
                
                valid_mask = pickup.notna() & dropoff.notna()
                if valid_mask.sum() > 0:
                    invalid_time = (pickup[valid_mask] > dropoff[valid_mask]).sum()
                    col_info['time_invalid_count'] = int(invalid_time)
                    col_info['time_invalid_rate'] = f"{invalid_time / valid_mask.sum() * 100:.2f}%"
        except Exception as e:
            # 如果时间检查失败，保持默认值
            pass
        
        report_df = pd.DataFrame(report_data)
    
    # 修复：确保目录存在，并正确处理文件名
    os.makedirs('outputs', exist_ok=True)
    
    # 如果 filename 已经包含 'outputs/'，直接使用；否则拼接
    if filename.startswith('outputs/') or filename.startswith('outputs\\'):
        report_path = filename
    else:
        report_path = os.path.join('outputs', filename)
    
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"    质量报告已保存: {report_path}")
    
    return report_df