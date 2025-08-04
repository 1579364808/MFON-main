"""
快速损失权重搜索脚本

用于快速测试几个关键的损失权重组合
"""

import os
import sys
import json
import time
import datetime

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config
from train.Adaptive_TVA_train import AdaptiveTVA_train_fusion
from utils import Metrics
from data_loader import MOSEIDataloader


def quick_search():
    """
    快速搜索几个关键的损失权重组合
    
    基于理论分析和经验，测试以下组合：
    1. 当前默认值: delta_va=0.3, delta_nce=0.001
    2. 增强知识蒸馏: delta_va=0.5, delta_nce=0.001  
    3. 增强对比学习: delta_va=0.3, delta_nce=0.01
    4. 平衡组合: delta_va=0.4, delta_nce=0.005
    5. 保守组合: delta_va=0.2, delta_nce=0.001
    6. 激进组合: delta_va=0.8, delta_nce=0.05
    """
    
    print("🚀 快速损失权重搜索")
    print("="*60)
    
    # 定义测试组合
    test_combinations = [
        {'name': '当前默认', 'delta_va': 0.3, 'delta_nce': 0.001, 'description': '当前配置基线'},
        {'name': '增强蒸馏', 'delta_va': 0.5, 'delta_nce': 0.001, 'description': '增强知识蒸馏权重'},
        {'name': '增强对比', 'delta_va': 0.3, 'delta_nce': 0.01, 'description': '增强对比学习权重'},
        {'name': '平衡组合', 'delta_va': 0.4, 'delta_nce': 0.005, 'description': '平衡两种辅助损失'},
        {'name': '保守组合', 'delta_va': 0.2, 'delta_nce': 0.001, 'description': '降低辅助损失影响'},
        {'name': '激进组合', 'delta_va': 0.8, 'delta_nce': 0.05, 'description': '大幅增强辅助损失'}
    ]
    
    print(f"测试组合数: {len(test_combinations)}")
    for i, combo in enumerate(test_combinations, 1):
        print(f"  {i}. {combo['name']}: delta_va={combo['delta_va']}, delta_nce={combo['delta_nce']} - {combo['description']}")
    print()
    
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
    
    # 保存原始配置
    original_delta_va = config.MOSEI.downStream.adaptiveTVAtrain.delta_va
    original_delta_nce = config.MOSEI.downStream.adaptiveTVAtrain.delta_nce
    original_epoch = config.MOSEI.downStream.adaptiveTVAtrain.epoch
    
    # 存储结果
    results = []
    best_result = None
    best_score = float('inf')
    
    # 创建结果保存目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"quick_search_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 运行所有测试组合
        for i, combo in enumerate(test_combinations, 1):
            print(f"\n{'='*60}")
            print(f"🧪 测试 {i}/{len(test_combinations)}: {combo['name']}")
            print(f"参数: delta_va={combo['delta_va']}, delta_nce={combo['delta_nce']}")
            print(f"说明: {combo['description']}")
            print(f"{'='*60}")
            
            try:
                # 设置测试参数
                config.MOSEI.downStream.adaptiveTVAtrain.delta_va = combo['delta_va']
                config.MOSEI.downStream.adaptiveTVAtrain.delta_nce = combo['delta_nce']
                config.MOSEI.downStream.adaptiveTVAtrain.epoch = 8  # 快速测试
                config.MOSEI.downStream.adaptiveTVAtrain.early_stopping_patience = 3
                
                # 记录开始时间
                start_time = time.time()
                
                # 运行训练
                training_summary = AdaptiveTVA_train_fusion(
                    config, metrics, config.seed, train_data, valid_data
                )
                
                # 计算训练时间
                duration = time.time() - start_time
                
                # 记录结果
                result = {
                    'name': combo['name'],
                    'params': {
                        'delta_va': combo['delta_va'],
                        'delta_nce': combo['delta_nce']
                    },
                    'description': combo['description'],
                    'best_loss': training_summary['best_loss'],
                    'total_epochs': training_summary['total_epochs'],
                    'stopped_early': training_summary['stopped_early'],
                    'duration': duration,
                    'status': 'success'
                }
                
                # 添加引导权重信息
                if training_summary['guidance_stats_history']:
                    final_stats = training_summary['guidance_stats_history'][-1]
                    result['guidance_weights'] = {
                        'text_weight': final_stats['mean_text_weight'],
                        'vision_weight': final_stats['mean_vision_weight'],
                        'audio_weight': final_stats['mean_audio_weight']
                    }
                
                results.append(result)
                
                # 更新最佳结果
                if result['best_loss'] < best_score:
                    best_score = result['best_loss']
                    best_result = result
                    print(f"🎉 发现新的最佳结果! 损失: {best_score:.6f}")
                
                print(f"✅ 测试完成 - 最佳损失: {result['best_loss']:.6f}, 用时: {duration:.1f}s")
                
            except Exception as e:
                error_result = {
                    'name': combo['name'],
                    'params': {
                        'delta_va': combo['delta_va'],
                        'delta_nce': combo['delta_nce']
                    },
                    'description': combo['description'],
                    'error': str(e),
                    'status': 'failed'
                }
                results.append(error_result)
                print(f"❌ 测试失败: {e}")
        
        # 保存结果
        save_results(results, best_result, save_dir)
        
        # 显示总结
        print_summary(results, best_result)
        
    finally:
        # 恢复原始配置
        config.MOSEI.downStream.adaptiveTVAtrain.delta_va = original_delta_va
        config.MOSEI.downStream.adaptiveTVAtrain.delta_nce = original_delta_nce
        config.MOSEI.downStream.adaptiveTVAtrain.epoch = original_epoch


def save_results(results, best_result, save_dir):
    """保存搜索结果"""
    
    # 保存详细结果
    results_file = os.path.join(save_dir, 'quick_search_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'search_type': 'quick_search',
            'timestamp': datetime.datetime.now().isoformat(),
            'best_result': best_result,
            'all_results': results
        }, f, indent=2, ensure_ascii=False)
    
    # 保存最佳配置
    if best_result:
        config_file = os.path.join(save_dir, 'best_config.txt')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("# 快速搜索最佳损失权重配置\n")
            f.write("# 复制以下参数到config.py的adaptiveTVAtrain类中:\n\n")
            f.write(f"delta_va = {best_result['params']['delta_va']}\n")
            f.write(f"delta_nce = {best_result['params']['delta_nce']}\n")
            f.write(f"\n# 配置说明: {best_result['description']}\n")
            f.write(f"# 最佳验证损失: {best_result['best_loss']:.6f}\n")
            f.write(f"# 训练轮数: {best_result['total_epochs']}\n")
    
    print(f"\n📁 结果已保存到: {save_dir}/")


def print_summary(results, best_result):
    """打印搜索总结"""
    
    print("\n" + "="*60)
    print("🎯 快速搜索完成总结")
    print("="*60)
    
    # 成功的结果
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    print(f"总测试数: {len(results)}")
    print(f"成功测试: {len(successful_results)}")
    print(f"失败测试: {len(failed_results)}")
    
    if best_result:
        print(f"\n🏆 最佳结果:")
        print(f"   配置: {best_result['name']}")
        print(f"   参数: delta_va={best_result['params']['delta_va']}, delta_nce={best_result['params']['delta_nce']}")
        print(f"   说明: {best_result['description']}")
        print(f"   最佳损失: {best_result['best_loss']:.6f}")
        print(f"   训练轮数: {best_result['total_epochs']}")
        
        if 'guidance_weights' in best_result:
            gw = best_result['guidance_weights']
            print(f"   引导权重 - 文本: {gw['text_weight']:.3f}, "
                  f"视觉: {gw['vision_weight']:.3f}, "
                  f"音频: {gw['audio_weight']:.3f}")
    
    if successful_results:
        print(f"\n📊 所有成功结果 (按损失排序):")
        sorted_results = sorted(successful_results, key=lambda x: x['best_loss'])
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result['name']}: "
                  f"delta_va={result['params']['delta_va']}, "
                  f"delta_nce={result['params']['delta_nce']}, "
                  f"loss={result['best_loss']:.6f}")
    
    print("="*60)


if __name__ == '__main__':
    quick_search()
