
"""
m4_qa_system.py - 命令行问答系统



支持的问题类型：
    - 时段查询：查询某时段的订单量/平均费用
    - 区域排名：查询订单量最高/最低的区域
    - 需求预测：使用神经网络预测某区域某时段的出行需求量
    - 费用估算：估算某行程的费用
    - 人数估算：估算某时段/区域的平均乘客人数
"""

import pandas as pd
import numpy as np
import re
import torch
import pickle
import os


class TaxiQASystem:
    """出租车数据问答系统"""
    
    def __init__(self, df, model_results=None):
        """
        初始化问答系统
        
        参数:
            df: 清洗后的DataFrame（任务1输出）
            model_results: 建模结果（任务3输出，包含神经网络模型和标准化器）
        """
        self.df = df
        self.model_results = model_results
        
        # 从model_results中提取模型和标准化器
        self.nn_model = None
        self.scaler = None
        self.feature_config = None
        self.feature_dim = None
        self.target_dim = None
        
        if model_results:
            self.nn_model = model_results.get('nn_model')
            self.scaler = model_results.get('scaler')
            self.feature_config = model_results.get('feature_config')
            self.feature_dim = model_results.get('feature_dim')
            self.target_dim = model_results.get('target_dim')
            print("已加载神经网络模型和标准化器")
        
        # 定义关键词模式
        self.patterns = {
            '时段查询': {
                'keywords': ['几点', '小时', '时段', '早高峰', '晚高峰', '凌晨', '上午', '下午', '晚上', '周末', '工作日'],
                'question_words': ['多少', '订单', '客流量']
            },
            '区域排名': {
                'keywords': ['区域', '地区', '地点', 'top', '最高', '最低', '排名', '前', '后'],
                'question_words': ['订单', '客流量', '热闹', '繁忙']
            },
            '需求预测': {
                'keywords': ['预测', '预计', '估', '未来'],
                'question_words': ['订单', '需求', '客流量']
            },
            '费用估算': {
                'keywords': ['费用', '车费', '多少钱', '价格', '花费', '小费', '过路费'],
                'question_words': ['多少', '大概']
            },
            '人数估算': {
                'keywords': ['人数', '乘客', '人', '载客'],
                'question_words': ['平均', '多少', '通常']
            }
        }
        
        print("\n问答系统初始化完成")
        print("   支持的问题类型: 时段查询, 区域排名, 需求预测, 费用估算, 人数估算")
        print("   输入 'quit' 或 'exit' 退出")
    
    def parse_question(self, question):
        """
        解析用户问题，识别意图
        
        返回:
            intent: 问题类型
            entities: 提取的实体（如时间、区域等）
        """
        question = question.strip()
        entities = {}
        
        # 1. 识别问题类型
        intent_scores = {}
        for intent, pattern in self.patterns.items():
            score = 0
            for kw in pattern['keywords']:
                if kw in question:
                    score += 2
            for qw in pattern['question_words']:
                if qw in question:
                    score += 1
            intent_scores[intent] = score
        
        # 选择得分最高的意图
        intent = max(intent_scores, key=intent_scores.get) if max(intent_scores.values()) > 0 else '未知'
        
        # 2. 提取实体
        # 提取小时（如"8点"、"下午3点"）
        hour_patterns = [
            r'(\d{1,2})\s*点',
            r'(\d{1,2})\s*时',
            r'(凌晨|早上|上午|中午|下午|晚上|深夜)\s*(\d{1,2})?\s*点?'
        ]
        for pattern in hour_patterns:
            match = re.search(pattern, question)
            if match:
                if len(match.groups()) == 2:
                    period = match.group(1)
                    hour = match.group(2)
                    hour = int(hour) if hour else self._period_to_hour(period)
                    entities['hour'] = hour
                else:
                    hour = int(match.group(1))
                    entities['hour'] = hour
                break
        
        # 提取区域
        zone_patterns = [
            r'区域\s*(\d+)',
            r'(曼哈顿|布鲁克林|皇后|布朗克斯|史坦顿)',
        ]
        for pattern in zone_patterns:
            match = re.search(pattern, question)
            if match:
                entities['zone'] = match.group(1)
                break
        
        # 提取时段类型
        if '早高峰' in question:
            entities['period'] = 'morning_peak'
        elif '晚高峰' in question:
            entities['period'] = 'evening_peak'
        elif '周末' in question:
            entities['is_weekend'] = True
        elif '工作日' in question:
            entities['is_weekend'] = False
        
        return intent, entities
    
    def _period_to_hour(self, period):
        """将时段转换为近似小时"""
        period_map = {
            '凌晨': 2, '早上': 8, '上午': 10,
            '中午': 12, '下午': 15, '晚上': 20, '深夜': 23
        }
        return period_map.get(period, 12)
    
    def _extract_features_for_prediction(self, entities):
        """
        从实体和DataFrame中提取特征向量，用于神经网络预测
        
        返回:
            feature_vector: shape (1, feature_dim) 的numpy数组
        """
        if not self.feature_config:
            return None
        
        features = self.feature_config['final_features']
        
        # 构建特征字典，默认值
        feature_dict = {}
        
        # 1. 时间特征
        hour = entities.get('hour', 12)  # 默认12点
        feature_dict['pickup_hour'] = hour % 24
        feature_dict['pickup_is_weekend'] = 1 if entities.get('is_weekend', False) else 0
        is_peak = 1 if entities.get('period') in ['morning_peak', 'evening_peak'] else 0
        feature_dict['pickup_is_peak'] = is_peak
        
        # 2. 行程特征（使用平均值填充）
        feature_dict['trip_distance'] = self.df['trip_distance'].mean()
        feature_dict['passenger_count'] = self.df['passenger_count'].mean()
        feature_dict['RatecodeID'] = 1  # 默认标准费率
        feature_dict['PULocationID'] = 1  # 默认区域1
        feature_dict['DOLocationID'] = 1  # 默认区域1
        feature_dict['store_and_fwd_flag'] = 0  # 默认N
        
        # 3. 费用特征（使用平均值）
        feature_dict['fare_amount'] = self.df['fare_amount'].mean()
        feature_dict['extra'] = self.df['extra'].mean()
        feature_dict['mta_tax'] = self.df['mta_tax'].mean()
        feature_dict['improvement_surcharge'] = self.df['improvement_surcharge'].mean()
        feature_dict['congestion_surcharge'] = self.df['congestion_surcharge'].mean()
        feature_dict['airport_fee'] = self.df['airport_fee'].mean()
        feature_dict['cbd_congestion_fee'] = self.df['cbd_congestion_fee'].mean()
        
        # 按特征顺序构建向量
        feature_vector = []
        for f in features:
            if f in feature_dict:
                feature_vector.append(feature_dict[f])
            elif f in self.df.columns:
                # 如果特征在DataFrame中但未在字典中，使用中位数
                feature_vector.append(self.df[f].median())
            else:
                feature_vector.append(0)
        
        return np.array(feature_vector).reshape(1, -1).astype(np.float32)
    
    def answer(self, question):
        """
        回答用户问题
        
        返回:
            dict: {
                'conclusion': 数字结论,
                'explanation': 文本解释,
                'file_paths': [相关文件路径]
            }
        """
        intent, entities = self.parse_question(question)
        
        if intent == '时段查询':
            return self._answer_time_query(question, entities)
        elif intent == '区域排名':
            return self._answer_region_ranking(question, entities)
        elif intent == '需求预测':
            return self._answer_demand_prediction(question, entities)
        elif intent == '费用估算':
            return self._answer_fare_estimation(question, entities)
        elif intent == '人数估算':
            return self._answer_passenger_estimation(question, entities)
        else:
            return {
                'conclusion': None,
                'explanation': '抱歉，我不理解您的问题。请尝试询问：时段查询、区域排名、需求预测、费用估算或人数估算相关问题。',
                'file_paths': []
            }
    
    def _answer_time_query(self, question, entities):
        """处理时段查询 - 动态查询DataFrame"""
        # 检查是否问具体小时
        if 'hour' in entities:
            hour = entities['hour']
            # 动态查询DataFrame
            subset = self.df[self.df['pickup_hour'] == hour]
            if len(subset) > 0:
                count = len(subset)
                avg_fare = subset['fare_amount'].mean()
                return {
                    'conclusion': f'{hour}点订单量: {count:,} 单, 平均车费: ${avg_fare:.2f}',
                    'explanation': f'在{hour}点时段，共有{count:,}笔订单，平均车费为${avg_fare:.2f}。',
                    'file_paths': ['outputs/m2_1_hourly_demand.png']
                }
        
        # 检查是否问高峰时段
        if '早高峰' in question:
            subset = self.df[self.df['pickup_peak_type'] == 'morning_peak']
            if len(subset) > 0:
                count = len(subset)
                avg_fare = subset['fare_amount'].mean()
                return {
                    'conclusion': f'早高峰订单量: {count:,} 单, 平均车费: ${avg_fare:.2f}',
                    'explanation': f'早高峰时段(7-9点)共有{count:,}笔订单，平均车费为${avg_fare:.2f}。',
                    'file_paths': ['outputs/m2_1_hourly_demand.png']
                }
        elif '晚高峰' in question:
            subset = self.df[self.df['pickup_peak_type'] == 'evening_peak']
            if len(subset) > 0:
                count = len(subset)
                avg_fare = subset['fare_amount'].mean()
                return {
                    'conclusion': f'晚高峰订单量: {count:,} 单, 平均车费: ${avg_fare:.2f}',
                    'explanation': f'晚高峰时段(16-18点)共有{count:,}笔订单，平均车费为${avg_fare:.2f}。',
                    'file_paths': ['outputs/m2_1_hourly_demand.png']
                }
        
        # 检查是否问周末/工作日
        if '周末' in question:
            subset = self.df[self.df['pickup_is_weekend'] == True]
            count = len(subset)
            avg_fare = subset['fare_amount'].mean()
            return {
                'conclusion': f'周末订单量: {count:,} 单, 平均车费: ${avg_fare:.2f}',
                'explanation': f'周末共有{count:,}笔订单，平均车费为${avg_fare:.2f}。',
                'file_paths': ['outputs/m2_1_hourly_demand.png']
            }
        elif '工作日' in question:
            subset = self.df[self.df['pickup_is_weekend'] == False]
            count = len(subset)
            avg_fare = subset['fare_amount'].mean()
            return {
                'conclusion': f'工作日订单量: {count:,} 单, 平均车费: ${avg_fare:.2f}',
                'explanation': f'工作日共有{count:,}笔订单，平均车费为${avg_fare:.2f}。',
                'file_paths': ['outputs/m2_1_hourly_demand.png']
            }
        
        # 默认返回整体统计
        total_orders = len(self.df)
        avg_fare = self.df['fare_amount'].mean()
        return {
            'conclusion': f'总订单量: {total_orders:,}, 平均车费: ${avg_fare:.2f}',
            'explanation': f'数据集中共有{total_orders:,}笔订单，平均车费为${avg_fare:.2f}。',
            'file_paths': ['outputs/m2_1_hourly_demand.png']
        }
    
    def _answer_region_ranking(self, question, entities):
        """处理区域排名 - 动态查询DataFrame"""
        top_n = 5
        match = re.search(r'前\s*(\d+)', question)
        if match:
            top_n = int(match.group(1))
        
        # 动态统计各区域订单量
        region_counts = self.df['PULocationID'].value_counts()
        top_regions = region_counts.head(top_n)
        
        conclusion = f"订单量TOP{top_n}区域: "
        for i, (loc_id, count) in enumerate(top_regions.items()):
            conclusion += f"区域{loc_id}({count:,}单)"
            if i < top_n - 1:
                conclusion += ", "
        
        return {
            'conclusion': conclusion,
            'explanation': f'订单量最高的{top_n}个区域如上所示，其中区域{top_regions.index[0]}最为繁忙。',
            'file_paths': ['outputs/m2_2_top10_regions.png', 'outputs/m2_2_choropleth_map.png']
        }
    
    def _answer_demand_prediction(self, question, entities):
        """处理需求预测 - 使用神经网络模型预测"""
        # 如果模型可用，使用神经网络预测
        if self.nn_model is not None and self.scaler is not None:
            try:
                # 提取特征
                feature_vector = self._extract_features_for_prediction(entities)
                if feature_vector is not None:
                    # 标准化
                    feature_scaled = self.scaler.transform(feature_vector)
                    # 转换为张量并预测
                    self.nn_model.eval()
                    with torch.no_grad():
                        feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32)
                        prediction = self.nn_model(feature_tensor).numpy()
                    
                    # 解析预测结果
                    tip_pred = prediction[0][0]
                    toll_pred = prediction[0][1]
                    hour = entities.get('hour', 12)
                    
                    return {
                        'conclusion': f'预测{hour}点订单需求: 预计小费 ${tip_pred:.2f}, 过路费 ${toll_pred:.2f}',
                        'explanation': f'基于神经网络模型预测，{hour}点时段的平均小费约为${tip_pred:.2f}，过路费约为${toll_pred:.2f}。',
                        'file_paths': ['outputs/m3_neural_network_loss.png', 'outputs/m3_model_metrics.csv']
                    }
            except Exception as e:
                print(f"    模型预测出错: {e}")
                # 降级到历史平均
        
        # 降级方案：基于历史平均
        hour = entities.get('hour', 12)
        subset = self.df[self.df['pickup_hour'] == hour]
        if len(subset) > 0:
            avg_tip = subset['tip_amount'].mean()
            avg_toll = subset['tolls_amount'].mean()
            return {
                'conclusion': f'预测{hour}点订单需求: 平均小费 ${avg_tip:.2f}, 平均过路费 ${avg_toll:.2f}',
                'explanation': f'基于历史数据预测，{hour}点时段的平均小费约为${avg_tip:.2f}，过路费约为${avg_toll:.2f}。',
                'file_paths': ['outputs/m3_neural_network_loss.png', 'outputs/m3_model_metrics.csv']
            }
        else:
            return {
                'conclusion': f'预测{hour}点订单量: 约 {len(self.df) // 24:,} 单',
                'explanation': f'基于平均每小时订单量估算，{hour}点时段预计约有{len(self.df) // 24:,}笔订单。',
                'file_paths': ['outputs/m3_neural_network_loss.png', 'outputs/m3_model_metrics.csv']
            }
    
    def _answer_fare_estimation(self, question, entities):
        """处理费用估算 - 动态查询DataFrame"""
        # 提取距离
        dist_match = re.search(r'(\d+\.?\d*)\s*英里', question)
        if dist_match:
            distance = float(dist_match.group(1))
            # 基于距离估算车费（简化：基础费用+距离*单价）
            base_fare = 2.50
            per_mile = 2.50
            estimated = base_fare + distance * per_mile
            
            # 加上平均附加费
            avg_extra = self.df['extra'].mean()
            total_est = estimated + avg_extra
            
            return {
                'conclusion': f'预估车费: ${total_est:.2f} (距离{distance}英里)',
                'explanation': f'基于标准费率估算，{distance}英里的行程预计总费用为${total_est:.2f}（含平均附加费${avg_extra:.2f}）。',
                'file_paths': ['outputs/m2_3_fare_analysis_combined.png']
            }
        
        # 默认返回平均车费
        avg_fare = self.df['fare_amount'].mean()
        avg_total = self.df['total_amount'].mean()
        return {
            'conclusion': f'平均车费: ${avg_fare:.2f}, 平均总费用: ${avg_total:.2f}',
            'explanation': f'数据集中平均车费为${avg_fare:.2f}，平均总费用为${avg_total:.2f}。',
            'file_paths': ['outputs/m2_3_fare_analysis_combined.png']
        }
    
    def _answer_passenger_estimation(self, question, entities):
        """处理人数估算 - 动态查询DataFrame"""
        # 如果有小时信息，按小时统计
        if 'hour' in entities:
            hour = entities['hour']
            subset = self.df[self.df['pickup_hour'] == hour]
            if len(subset) > 0:
                avg = subset['passenger_count'].mean()
                return {
                    'conclusion': f'{hour}点平均乘客人数: {avg:.2f} 人',
                    'explanation': f'在{hour}点时段，平均每车乘客人数为{avg:.2f}人。',
                    'file_paths': ['outputs/m2_3_fare_analysis_combined.png']
                }
        
        # 检查是否问周末/工作日
        if '周末' in question:
            subset = self.df[self.df['pickup_is_weekend'] == True]
            avg = subset['passenger_count'].mean()
            return {
                'conclusion': f'周末平均乘客人数: {avg:.2f} 人',
                'explanation': f'周末平均每车乘客人数为{avg:.2f}人。',
                'file_paths': ['outputs/m2_3_fare_analysis_combined.png']
            }
        elif '工作日' in question:
            subset = self.df[self.df['pickup_is_weekend'] == False]
            avg = subset['passenger_count'].mean()
            return {
                'conclusion': f'工作日平均乘客人数: {avg:.2f} 人',
                'explanation': f'工作日平均每车乘客人数为{avg:.2f}人。',
                'file_paths': ['outputs/m2_3_fare_analysis_combined.png']
            }
        
        # 默认返回整体平均
        avg = self.df['passenger_count'].mean()
        return {
            'conclusion': f'平均乘客人数: {avg:.2f} 人',
            'explanation': f'数据集中平均每车乘客人数为{avg:.2f}人。',
            'file_paths': ['outputs/m2_3_fare_analysis_combined.png']
        }
    
    def run_cli(self):
        """运行命令行交互循环"""
        print("\n" + "="*60)
        print("出租车数据问答系统")
        print("="*60)
        print("\n支持的问题示例:")
        print("  - '早上8点订单量多少？'")
        print("  - '订单量最高的前5个区域是哪些？'")
        print("  - '预测下午3点的订单需求'")
        print("  - '估算5英里的车费'")
        print("  - '平均每车坐几个人？'")
        print("  - '周末平均乘客人数是多少？'")
        print("\n输入 'quit' 或 'exit' 退出")
        print("-"*60)
        
        while True:
            try:
                question = input("\n请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n 再见！")
                    break
                
                if not question:
                    continue
                
                # 回答问题
                result = self.answer(question)
                
                # 输出结果
                print("\n" + "-"*50)
                if result['conclusion']:
                    print(f"结论: {result['conclusion']}")
                print(f" 说明: {result['explanation']}")
                if result['file_paths']:
                    print(f"相关文件: {', '.join(result['file_paths'])}")
                else:
                    print("无相关文件")
                print("-"*50)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n 处理出错: {e}")


def run_qa_system(df, model_results=None):
    """运行问答系统"""
    print("\n" + "="*70)
    print("m4_qa_system: 启动问答系统")
    print("="*70)
    
    qa_system = TaxiQASystem(df, model_results)
    qa_system.run_cli()