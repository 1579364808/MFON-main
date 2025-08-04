"""
损失权重网格搜索脚本

用于寻找最优的 delta_va 和 delta_nce 组合
"""

import os
import sys
import json
import time
import datetime
from itertools import product
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config
from train.Adaptive_TVA_train import AdaptiveTVA_train_fusion
from utils import Metrics
from data_loader import MOSEIDataloader


class LossWeightGridSearch:
    """
    损失权重网格搜索类

    搜索最优的 delta_va (知识蒸馏权重) 和 delta_nce (对比学习权重) 组合

    总损失函数：
    total_loss = main_loss + delta_va * (loss_v + loss_a) + delta_nce * loss_nce

    其中：
    - main_loss: 主任务MSE损失 (情感回归)
    - loss_v, loss_a: 视觉和音频的KL散度知识蒸馏损失
    - loss_nce: InfoNCE对比学习损失 (模态间语义对齐)

    搜索策略：
    1. delta_va 控制预训练知识的保持程度，过大可能导致过拟合，过小可能丢失知识
    2. delta_nce 控制模态间对齐强度，过大可能干扰主任务，过小可能对齐不足
    3. 使用较少的epoch (8轮) 进行快速搜索，避免过度训练
    4. 启用早停机制，防止在搜索过程中的过拟合
    """
    
    def __init__(self, config, train_data, valid_data, metrics):
        self.config = config
        self.train_data = train_data
        self.valid_data = valid_data
        self.metrics = metrics
        
        # 搜索空间定义 - 基于损失函数的重要性设计
        self.search_space = {
            # 知识蒸馏权重：控制预训练知识的保持程度
            # 范围：0.1-1.0，当前默认0.3
            'delta_va': [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0],

            # 对比学习权重：控制模态间语义对齐的强度
            # 范围：0.001-0.2，当前默认0.001
            'delta_nce': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        }
        
        # 搜索结果存储
        self.results = []
        self.best_result = None
        self.best_score = float('inf')
        
        # 创建结果保存目录
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = f"loss_weight_search_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def run_single_trial(self, delta_va, delta_nce, trial_idx, total_trials):
        """
        运行单次试验
        
        Args:
            delta_va: 知识蒸馏权重
            delta_nce: 对比学习权重
            trial_idx: 试验索引
            total_trials: 总试验数
            
        Returns:
            result: 试验结果字典
        """
        print(f"\n{'='*60}")
        print(f"🔍 试验 {trial_idx}/{total_trials}")
        print(f"参数: delta_va={delta_va}, delta_nce={delta_nce}")
        print(f"{'='*60}")
        
        # 临时修改配置
        original_delta_va = self.config.MOSEI.downStream.adaptiveTVAtrain.delta_va
        original_delta_nce = self.config.MOSEI.downStream.adaptiveTVAtrain.delta_nce
        original_epoch = self.config.MOSEI.downStream.adaptiveTVAtrain.epoch
        
        try:
            # 设置搜索参数
            self.config.MOSEI.downStream.adaptiveTVAtrain.delta_va = delta_va
            self.config.MOSEI.downStream.adaptiveTVAtrain.delta_nce = delta_nce

            # 快速搜索配置：减少训练时间，保持搜索效率
            self.config.MOSEI.downStream.adaptiveTVAtrain.epoch = 8  # 8个epoch足够观察趋势
            self.config.MOSEI.downStream.adaptiveTVAtrain.early_stopping_patience = 3  # 更早停止
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行训练
            training_summary = AdaptiveTVA_train_fusion(
                self.config, self.metrics, self.config.seed, 
                self.train_data, self.valid_data
            )
            
            # 计算训练时间
            duration = time.time() - start_time
            
            # 提取结果
            result = {
                'trial_idx': trial_idx,
                'params': {
                    'delta_va': delta_va,
                    'delta_nce': delta_nce
                },
                'best_loss': training_summary['best_loss'],
                'total_epochs': training_summary['total_epochs'],
                'stopped_early': training_summary['stopped_early'],
                'duration': duration,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'success'
            }
            
            # 如果有引导权重统计，也记录下来
            if training_summary['guidance_stats_history']:
                final_stats = training_summary['guidance_stats_history'][-1]
                result['guidance_weights'] = {
                    'text_weight': final_stats['mean_text_weight'],
                    'vision_weight': final_stats['mean_vision_weight'],
                    'audio_weight': final_stats['mean_audio_weight']
                }
            
            print(f"✅ 试验完成 - 最佳损失: {result['best_loss']:.6f}")
            
        except Exception as e:
            # 处理训练失败的情况
            result = {
                'trial_idx': trial_idx,
                'params': {
                    'delta_va': delta_va,
                    'delta_nce': delta_nce
                },
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'failed'
            }
            print(f"❌ 试验失败: {e}")
            
        finally:
            # 恢复原始配置
            self.config.MOSEI.downStream.adaptiveTVAtrain.delta_va = original_delta_va
            self.config.MOSEI.downStream.adaptiveTVAtrain.delta_nce = original_delta_nce
            self.config.MOSEI.downStream.adaptiveTVAtrain.epoch = original_epoch
        
        return result
    
    def run_grid_search(self):
        """
        运行完整的网格搜索
        """
        print("🚀 开始损失权重网格搜索")
        print(f"搜索空间:")
        print(f"  delta_va: {self.search_space['delta_va']}")
        print(f"  delta_nce: {self.search_space['delta_nce']}")
        
        # 生成所有参数组合
        param_combinations = list(product(
            self.search_space['delta_va'],
            self.search_space['delta_nce']
        ))
        
        total_trials = len(param_combinations)
        print(f"总试验数: {total_trials}")
        
        # 记录搜索开始时间
        search_start_time = time.time()
        
        # 运行所有试验
        for trial_idx, (delta_va, delta_nce) in enumerate(param_combinations, 1):
            result = self.run_single_trial(delta_va, delta_nce, trial_idx, total_trials)
            self.results.append(result)
            
            # 更新最佳结果
            if (result['status'] == 'success' and 
                result['best_loss'] < self.best_score):
                self.best_score = result['best_loss']
                self.best_result = result
                print(f"🎉 发现新的最佳结果! 损失: {self.best_score:.6f}")
            
            # 保存中间结果
            self.save_intermediate_results()
        
        # 计算总搜索时间
        total_duration = time.time() - search_start_time
        
        # 保存最终结果
        self.save_final_results(total_duration)
        
        # 显示搜索总结
        self.print_search_summary()
    
    def save_intermediate_results(self):
        """保存中间结果"""
        results_file = os.path.join(self.save_dir, 'intermediate_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'best_result': self.best_result,
                'best_score': self.best_score
            }, f, indent=2)
    
    def save_final_results(self, total_duration):
        """保存最终结果"""
        # 统计成功和失败的试验
        successful_trials = [r for r in self.results if r['status'] == 'success']
        failed_trials = [r for r in self.results if r['status'] == 'failed']
        
        # 创建搜索报告
        search_report = {
            'search_config': {
                'search_space': self.search_space,
                'total_trials': len(self.results),
                'successful_trials': len(successful_trials),
                'failed_trials': len(failed_trials)
            },
            'search_stats': {
                'duration_hours': total_duration / 3600,
                'best_score': self.best_score,
                'best_params': self.best_result['params'] if self.best_result else None
            },
            'best_result': self.best_result,
            'all_results': self.results
        }
        
        # 保存搜索报告
        report_file = os.path.join(self.save_dir, 'search_report.json')
        with open(report_file, 'w') as f:
            json.dump(search_report, f, indent=2)
        
        # 保存最佳配置
        if self.best_result:
            best_config_file = os.path.join(self.save_dir, 'best_config.txt')
            with open(best_config_file, 'w') as f:
                f.write("# 最佳损失权重配置\n")
                f.write("# 复制以下参数到config.py的adaptiveTVAtrain类中:\n\n")
                f.write(f"delta_va = {self.best_result['params']['delta_va']}\n")
                f.write(f"delta_nce = {self.best_result['params']['delta_nce']}\n")
                f.write(f"\n# 最佳验证损失: {self.best_score:.6f}\n")
    
    def print_search_summary(self):
        """打印搜索总结"""
        print("\n" + "="*60)
        print("🎯 损失权重网格搜索完成!")
        print("="*60)
        
        if self.best_result:
            print(f"🏆 最佳结果:")
            print(f"   delta_va: {self.best_result['params']['delta_va']}")
            print(f"   delta_nce: {self.best_result['params']['delta_nce']}")
            print(f"   最佳损失: {self.best_score:.6f}")
            
            if 'guidance_weights' in self.best_result:
                gw = self.best_result['guidance_weights']
                print(f"   引导权重 - 文本: {gw['text_weight']:.3f}, "
                      f"视觉: {gw['vision_weight']:.3f}, "
                      f"音频: {gw['audio_weight']:.3f}")
        else:
            print("❌ 没有找到成功的试验结果")
        
        # 统计信息
        successful_trials = [r for r in self.results if r['status'] == 'success']
        print(f"\n📊 搜索统计:")
        print(f"   总试验数: {len(self.results)}")
        print(f"   成功试验: {len(successful_trials)}")
        print(f"   失败试验: {len(self.results) - len(successful_trials)}")
        
        print(f"\n📁 结果保存在: {self.save_dir}/")
        print("   - search_report.json: 完整搜索报告")
        print("   - best_config.txt: 最佳配置文件")


def main():
    """主函数"""
    print("🔍 损失权重网格搜索")
    print("="*60)
    
    # 数据加载
    batch_size = config.MOSEI.downStream.batch_size
    print('📊 加载数据...')
    
    train_data = MOSEIDataloader('train', config.MOSEI.path.raw_data_path, batch_size=batch_size)
    valid_data = MOSEIDataloader('valid', config.MOSEI.path.raw_data_path, batch_size=batch_size, shuffle=False)
    
    print('✅ 数据加载完成')
    
    # 评估指标
    metrics = Metrics()
    
    # 设置随机种子
    config.seed = 1111
    
    # 创建搜索器
    searcher = LossWeightGridSearch(config, train_data, valid_data, metrics)
    
    # 运行搜索
    searcher.run_grid_search()


if __name__ == '__main__':
    main()
