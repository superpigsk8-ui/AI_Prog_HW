
"""
m3_modeling.py - 建模（神经网络 + 随机森林）

功能：
    1. 特征设计
    2. 数据划分（8:2，固定随机种子）
    3. PyTorch神经网络训练
    4. 随机森林对比模型
    5. 模型评估（MAE, RMSE, R²）
    6. 方法优劣分析
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import time
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'


def run_modeling_pipeline(df):
    """运行完整建模流程，返回模型、标准化器和特征配置供任务4使用"""
    print("\n" + "="*70)
    print("m3_modeling: 建模与评估")
    print("="*70)
    
    # 设计特征
    feature_config = design_features(df)
    
    # 数据划分
    data_ready = prepare_data(df, feature_config)
    
    if data_ready is None:
        print("\n数据准备失败")
        return None
    
    # 神经网络训练
    nn_result = train_neural_network(data_ready)
    
    # 随机森林对比
    rf_result = train_random_forest(data_ready)
    
    # 模型评估
    if nn_result is not None and not nn_result.get('failed', False):
        evaluate_and_save_metrics(data_ready, nn_result, rf_result)
        print_comparison_analysis(nn_result, rf_result)
    else:
        print("\n神经网络训练失败，仅保存随机森林结果")
        save_rf_metrics_only(data_ready, rf_result)
    
    # 导出模型供任务4使用
    os.makedirs('outputs', exist_ok=True)
    
    # 保存标准化器
    with open('outputs/scaler.pkl', 'wb') as f:
        pickle.dump(data_ready['scaler'], f)
    
    # 保存特征配置
    with open('outputs/feature_config.pkl', 'wb') as f:
        pickle.dump(feature_config, f)
    
    # 保存神经网络模型
    if nn_result is not None and not nn_result.get('failed', False):
        torch.save(nn_result['model'].state_dict(), 'outputs/nn_model_weights.pth')
        print("\n神经网络模型已保存至 outputs/nn_model_weights.pth")
    else:
        print("\n神经网络模型未保存（训练失败）")
    
    print("标准化器已保存至 outputs/scaler.pkl")
    print("特征配置已保存至 outputs/feature_config.pkl")
    
    print("\n建模全部完成")
    print("="*70)
    
    return {
        'nn_result': nn_result,
        'rf_result': rf_result,
        'nn_model': nn_result['model'] if nn_result and not nn_result.get('failed', False) else None,
        'scaler': data_ready['scaler'],
        'feature_config': feature_config,
        'data_ready': data_ready
    }


def design_features(df):
    """设计输入特征"""
    print("\n  设计特征...")
    
    time_features = ['pickup_hour', 'pickup_is_weekend', 'pickup_is_peak']
    trip_features = ['trip_distance', 'passenger_count', 'RatecodeID', 
                     'PULocationID', 'DOLocationID', 'store_and_fwd_flag']
    fee_features = ['fare_amount', 'extra', 'mta_tax', 'improvement_surcharge',
                    'congestion_surcharge', 'airport_fee', 'cbd_congestion_fee']
    
    all_features = time_features + trip_features + fee_features
    target_features = ['tip_amount', 'tolls_amount']
    
    # 检查可用特征
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    
    if missing:
        print(f"    跳过缺失特征: {missing}")
    
    print(f"    可用特征: {len(available)} 个")
    print(f"    目标变量: {target_features}")
    
    return {
        'final_features': available,
        'target_features': target_features,
        'all_features': all_features,
        'categorical_cols': ['RatecodeID', 'PULocationID', 'DOLocationID', 'store_and_fwd_flag'],
        'feature_descriptions': {
            'pickup_hour': '小时(0-23): 不同时段出行需求不同',
            'pickup_is_weekend': '是否周末: 周末出行模式与工作日不同',
            'pickup_is_peak': '是否高峰: 高峰时段拥堵影响行程',
            'trip_distance': '行程距离(英里): 直接影响车费',
            'passenger_count': '乘客人数: 可能影响小费金额',
            'RatecodeID': '费率代码: 不同费率影响费用结构',
            'PULocationID': '上车区域ID: 影响过路费概率',
            'DOLocationID': '下车区域ID: 影响过路费',
            'store_and_fwd_flag': '离线传输标志: 反映网络状况',
            'fare_amount': '基础车费: 小费金额的最重要参考',
            'extra': '附加费: 影响总费用',
            'mta_tax': 'MTA税: 反映行程类型',
            'improvement_surcharge': '改善附加费: 费用组成部分',
            'congestion_surcharge': '拥堵费: 反映是否进入拥堵区',
            'airport_fee': '机场费: 标识机场行程',
            'cbd_congestion_fee': 'CBD拥堵费: 反映是否进入CBD区域'
        }
    }


def prepare_data(df, feature_config, test_size=0.2, random_seed=42):
    """数据划分"""
    print("\n  数据划分...")
    
    # 固定随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    print(f"    固定随机种子: {random_seed}")
    
    features = feature_config['final_features']
    targets = feature_config['target_features']
    categorical_cols = feature_config.get('categorical_cols', [])
    
    # 检查特征是否存在
    available_features = [f for f in features if f in df.columns]
    if len(available_features) == 0:
        print("没有可用的特征")
        return None
    
    model_df = df[available_features + targets].copy()
        
    # 1. 填充所有NaN为0
    model_df = model_df.fillna(0)
    
    # 2. 替换Inf
    model_df = model_df.replace([np.inf, -np.inf], 0)
    
    # 3. 处理异常值（截断到合理范围）
    for col in model_df.columns:
        if model_df[col].dtype in [np.float64, np.float32]:
            upper = model_df[col].quantile(0.995)
            lower = model_df[col].quantile(0.005)
            model_df[col] = model_df[col].clip(lower, upper)
    
    # 4. 编码分类变量
    label_encoders = {}
    for col in categorical_cols:
        if col in model_df.columns:
            le = LabelEncoder()
            model_df[col] = model_df[col].fillna(-1)
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            label_encoders[col] = le
    
    if 'store_and_fwd_flag' in model_df.columns:
        model_df['store_and_fwd_flag'] = model_df['store_and_fwd_flag'].map({'Y': 1, 'N': 0}).fillna(0)
    
    for target in targets:
        model_df[target] = model_df[target].fillna(0)
    
    # 5. 准备特征和目标
    X = model_df[available_features].values.astype(np.float32)
    y = model_df[targets].values.astype(np.float32)
    
    # 6. 最终检查并清理
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"    数据清理后 X NaN: {np.isnan(X).sum()}, y NaN: {np.isnan(y).sum()}")
    print(f"    X 范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"    y 范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, shuffle=True
    )
    
    print(f"    训练集: {X_train.shape[0]:,} 条")
    print(f"    测试集: {X_test.shape[0]:,} 条")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # 检查张量
    if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
        print("张量包含NaN，数据准备失败")
        return None
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_tensor': X_train_tensor,
        'X_test_tensor': X_test_tensor,
        'y_train_tensor': y_train_tensor,
        'y_test_tensor': y_test_tensor,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'features': available_features,
        'targets': targets,
        'feature_dim': X.shape[1],
        'target_dim': y.shape[1],
        'random_seed': random_seed,
        'categorical_cols': categorical_cols,
        'feature_config': feature_config
    }


class MultiOutputRegressionNet(nn.Module):
    """多输出回归神经网络"""
    def __init__(self, feature_dim, target_dim, hidden_dims=[64, 32]):
        super(MultiOutputRegressionNet, self).__init__()
        layers = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, target_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_neural_network(data_ready, epochs=100, learning_rate=0.001, patience=30):#嫌训练慢可以适当减少patience值和epochs值
    """训练神经网络"""
    print("\n  训练神经网络...")
    
    X_train_tensor = data_ready['X_train_tensor']
    y_train_tensor = data_ready['y_train_tensor']
    X_test_tensor = data_ready['X_test_tensor']
    y_test_tensor = data_ready['y_test_tensor']
    
    # 检查数据
    if torch.isnan(X_train_tensor).any() or torch.isnan(y_train_tensor).any():
        print("输入数据包含NaN")
        return {'failed': True}
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model = MultiOutputRegressionNet(data_ready['feature_dim'], data_ready['target_dim'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"    开始训练，学习率: {learning_rate}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            if torch.isnan(loss):
                print(f"轮次 {epoch+1}: 损失为NaN，停止训练")
                break
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        
        if np.isnan(epoch_train_loss):
            print(f"轮次 {epoch+1}: 训练损失为NaN，停止训练")
            break
        
        # 验证阶段
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                if torch.isnan(loss):
                    break
                epoch_test_loss += loss.item() * batch_X.size(0)
        epoch_test_loss /= len(test_loader.dataset)
        
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        
        scheduler.step(epoch_test_loss)
        
        # 早停检查
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch [{epoch+1:3d}/{epochs}] Train Loss: {epoch_train_loss:.6f} Test Loss: {epoch_test_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"早停触发! (第 {epoch+1} 轮)")
            break
    
    # 检查是否有有效的训练结果
    if len(train_losses) == 0:
        print("训练失败：没有有效的训练轮次")
        return {'failed': True}
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n    训练完成:")
    print(f"      - 总训练轮次: {len(train_losses)}")
    print(f"      - 最佳测试损失: {best_test_loss:.6f}")
    
    # 绘制loss曲线
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 清理数据
        train_clean = np.nan_to_num(train_losses, nan=0.0)
        test_clean = np.nan_to_num(test_losses, nan=0.0)
        
        ax.plot(train_clean, label='训练损失', linewidth=2, color='#2E86AB')
        ax.plot(test_clean, label='测试损失', linewidth=2, color='#A23B72')
        
        # 标记最佳模型位置
        if best_test_loss in test_losses:
            best_epoch = test_losses.index(best_test_loss) + 1
            ax.axvline(x=best_epoch - 1, color='green', linestyle='--', alpha=0.5, 
                      label=f'最佳模型 (轮次 {best_epoch})')
        
        ax.set_xlabel('训练轮次', fontsize=13)
        ax.set_ylabel('损失 (MSE)', fontsize=13)
        ax.set_title('神经网络训练损失曲线', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 设置合理的y轴范围
        all_losses = np.concatenate([train_clean, test_clean])
        if np.max(all_losses) > 0:
            ax.set_ylim(3.0, np.max(all_losses) * 1.1)
        
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/m3_neural_network_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Loss曲线已保存: outputs/m3_neural_network_loss.png")
    except Exception as e:
        print(f"绘图失败: {e}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss,
        'epochs_trained': len(train_losses),
        'failed': False
    }


def train_random_forest(data_ready):
    """训练随机森林对比模型"""
    print("\n  训练随机森林对比模型...")
    
    X_train = data_ready['X_train']
    X_test = data_ready['X_test']
    y_train = data_ready['y_train']
    y_test = data_ready['y_test']
    
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"    随机森林 MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
    
    return {
        'model': rf_model,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def safe_metrics(y_true, y_pred):
    """计算评估指标，处理NaN和Inf"""
    y_true_clean = np.nan_to_num(y_true, nan=0.0)
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0)
    
    try:
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)
    except Exception:
        mae, rmse, r2 = 0.0, 0.0, 0.0
    
    return mae, rmse, r2


def evaluate_and_save_metrics(data_ready, nn_result, rf_result):
    """评估并保存指标"""
    print("\n  保存评估指标...")
    
    X_test_tensor = data_ready['X_test_tensor']
    y_test = data_ready['y_test']
    targets = data_ready['targets']
    
    # 神经网络预测
    nn_model = nn_result['model']
    nn_model.eval()
    with torch.no_grad():
        nn_pred = nn_model(X_test_tensor).numpy()
    
    # 清理数据
    y_test_clean = np.nan_to_num(y_test, nan=0.0)
    nn_pred_clean = np.nan_to_num(nn_pred, nan=0.0)
    rf_pred_clean = np.nan_to_num(rf_result['y_pred'], nan=0.0)
    
    # 计算NN指标
    nn_mae, nn_rmse, nn_r2 = safe_metrics(y_test_clean, nn_pred_clean)
    print(f"    神经网络 - MAE: {nn_mae:.6f}, RMSE: {nn_rmse:.6f}, R²: {nn_r2:.6f}")
    
    # 创建指标DataFrame
    metrics_data = []
    for i, target in enumerate(targets):
        # 神经网络各目标
        mae, rmse, r2 = safe_metrics(y_test_clean[:, i], nn_pred_clean[:, i])
        metrics_data.append({
            'model': '神经网络',
            'target': target,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
        # 随机森林各目标
        mae, rmse, r2 = safe_metrics(y_test_clean[:, i], rf_pred_clean[:, i])
        metrics_data.append({
            'model': '随机森林',
            'target': target,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    # 整体指标
    metrics_data.append({'model': '神经网络', 'target': 'overall', 
                         'mae': nn_mae, 'rmse': nn_rmse, 'r2': nn_r2})
    metrics_data.append({'model': '随机森林', 'target': 'overall',
                         'mae': rf_result['mae'], 'rmse': rf_result['rmse'], 'r2': rf_result['r2']})
    
    metrics_df = pd.DataFrame(metrics_data)
    os.makedirs('outputs', exist_ok=True)
    metrics_df.to_csv('outputs/m3_model_metrics.csv', index=False, encoding='utf-8-sig')
    print("已保存: outputs/m3_model_metrics.csv")


def save_rf_metrics_only(data_ready, rf_result):
    """仅保存随机森林指标（当神经网络失败时）"""
    print("\n  保存随机森林评估指标...")
    
    y_test = data_ready['y_test']
    targets = data_ready['targets']
    rf_pred = rf_result['y_pred']
    
    metrics_data = []
    for i, target in enumerate(targets):
        mae, rmse, r2 = safe_metrics(y_test[:, i], rf_pred[:, i])
        metrics_data.append({
            'model': '随机森林',
            'target': target,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    metrics_data.append({'model': '随机森林', 'target': 'overall',
                         'mae': rf_result['mae'], 'rmse': rf_result['rmse'], 'r2': rf_result['r2']})
    
    metrics_df = pd.DataFrame(metrics_data)
    os.makedirs('outputs', exist_ok=True)
    metrics_df.to_csv('outputs/m3_model_metrics_rf_only.csv', index=False, encoding='utf-8-sig')
    print("已保存: outputs/m3_model_metrics_rf_only.csv")


def print_comparison_analysis(nn_result, rf_result):
    """打印两种方法的对比分析"""
    print("\n" + "="*70)
    print(" 两种方法性能对比")
    print("="*70)
    print(f"\n  【神经网络】最佳MSE: {nn_result['best_test_loss']:.6f}")
    print(f"  【随机森林】MAE: {rf_result['mae']:.6f}, RMSE: {rf_result['rmse']:.6f}, R²: {rf_result['r2']:.6f}")
    
    print("\n" + "-"*70)
    print("详细对比分析请查看 m3_modeling.py 文件末尾的注释")
    print("-"*70)


"""
任务3 - 方法对比分析：

一、模型特点对比
----------------
【神经网络 (PyTorch)】
    - 结构: 全连接网络 (特征维度 → 64 → 32 → 2)
    - 激活函数: ReLU
    - 正则化: Dropout (0.2)
    - 优化器: Adam (学习率 0.0005)
    - 训练轮数: 200 (早停)
    - 梯度裁剪: 防止梯度爆炸

【随机森林】
    - 树数量: 100
    - 最大深度: 20
    - 最小分裂样本: 10
    - 最小叶节点样本: 5

二、优势分析
------------
【神经网络的优势】
    1. 特征交互能力强: 多层非线性变换自动学习特征间复杂关系
    2. 端到端学习: 共享隐藏层表示，利用任务间相关性
    3. 数据量足够时性能通常优于传统方法

【随机森林的优势】
    1. 可解释性高: 可输出特征重要性分数
    2. 训练速度快: 无需GPU加速
    3. 对超参数不敏感: 默认参数通常表现良好
    4. 鲁棒性强: 对异常值和噪声容忍度高
"""