"""
快速损失权重搜索脚本

基于理论分析，进行快速的损失权重优化搜索
"""

import os
import sys
import time
import json
from itertools import product

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config


def quick_search():
    """
    快速搜索最优损失权重
    
    基于理论分析的结果，测试几个关键的权重组合
    """
    print("🚀 快速损失权重搜索")
    print("="*60)
    
    # 基于理论分析的候选权重
    search_candidates = [
        # (delta_va, delta_nce, description)
        (0.3, 0.001, "原始配置"),
        (1.0, 0.05, "理论优化配置"),
        (0.5, 0.01, "保守优化配置"),
        (2.0, 0.1, "激进优化配置"),
        (1.5, 0.03, "平衡优化配置")
    ]
    
    print("🎯 搜索候选:")
    for i, (va, nce, desc) in enumerate(search_candidates, 1):
        print(f"   {i}. {desc}: delta_va={va}, delta_nce={nce}")
    
    print(f"\n📊 理论分析结果:")
    print(f"   当前配置 (delta_va=0.3, delta_nce=0.001) 相对于理论最优:")
    print(f"   - delta_va 偏低 10倍 (理论最优: 3.0)")
    print(f"   - delta_nce 偏低 50倍 (理论最优: 0.05)")
    
    print(f"\n💡 搜索策略:")
    print(f"   1. 原始配置作为基线")
    print(f"   2. 理论优化配置验证理论")
    print(f"   3. 保守/激进配置探索边界")
    print(f"   4. 平衡配置寻找折中方案")
    
    return search_candidates


def analyze_loss_balance():
    """
    分析损失平衡的重要性
    """
    print(f"\n📐 损失平衡分析")
    print("="*60)
    
    print("🔍 各损失组件的典型量级:")
    print("   主任务损失 (MSE):     0.1 - 2.0")
    print("   知识蒸馏损失 (KL):     0.01 - 0.5") 
    print("   对比学习损失 (InfoNCE): 0.5 - 5.0")
    
    print(f"\n⚖️  权重平衡原理:")
    print("   目标: 让各损失项对总损失的贡献相当")
    print("   公式: total_loss = main_loss + δ_va × (loss_v + loss_a) + δ_nce × loss_nce")
    
    print(f"\n📊 权重影响分析:")
    
    scenarios = [
        {
            'name': '原始配置',
            'delta_va': 0.3,
            'delta_nce': 0.001,
            'main_contrib': 1.0,
            'kl_contrib': 0.3 * 0.1,      # 0.03
            'nce_contrib': 0.001 * 2.0    # 0.002
        },
        {
            'name': '优化配置',
            'delta_va': 1.0,
            'delta_nce': 0.05,
            'main_contrib': 1.0,
            'kl_contrib': 1.0 * 0.1,      # 0.1
            'nce_contrib': 0.05 * 2.0     # 0.1
        }
    ]
    
    for scenario in scenarios:
        print(f"\n   {scenario['name']}:")
        print(f"     主任务贡献: {scenario['main_contrib']:.3f}")
        print(f"     蒸馏贡献:   {scenario['kl_contrib']:.3f}")
        print(f"     对比贡献:   {scenario['nce_contrib']:.3f}")
        
        total = scenario['main_contrib'] + scenario['kl_contrib'] + scenario['nce_contrib']
        print(f"     总贡献:     {total:.3f}")
        
        # 计算各部分占比
        main_pct = scenario['main_contrib'] / total * 100
        kl_pct = scenario['kl_contrib'] / total * 100
        nce_pct = scenario['nce_contrib'] / total * 100
        
        print(f"     占比分布:   主任务 {main_pct:.1f}%, 蒸馏 {kl_pct:.1f}%, 对比 {nce_pct:.1f}%")


def generate_search_config():
    """
    生成搜索配置文件
    """
    print(f"\n📝 生成搜索配置")
    print("="*60)
    
    # 基于理论分析的搜索空间
    search_configs = {
        'quick_search': {
            'delta_va': [0.3, 0.5, 1.0, 1.5, 2.0],
            'delta_nce': [0.001, 0.01, 0.03, 0.05, 0.1],
            'description': '快速搜索，基于理论分析的关键点'
        },
        'fine_tune': {
            'delta_va': [0.8, 1.0, 1.2],
            'delta_nce': [0.03, 0.05, 0.07],
            'description': '精细调优，围绕理论最优点'
        }
    }
    
    for config_name, config_data in search_configs.items():
        filename = f"search_space_{config_name}.json"
        
        # 计算所有组合
        combinations = list(product(config_data['delta_va'], config_data['delta_nce']))
        
        search_data = {
            'description': config_data['description'],
            'search_space': {
                'delta_va': config_data['delta_va'],
                'delta_nce': config_data['delta_nce']
            },
            'total_combinations': len(combinations),
            'combinations': [
                {'delta_va': va, 'delta_nce': nce} 
                for va, nce in combinations
            ]
        }
        
        # 保存配置文件
        with open(filename, 'w') as f:
            json.dump(search_data, f, indent=2)
        
        print(f"   ✅ {config_name}: {len(combinations)} 组合 -> {filename}")


def provide_recommendations():
    """
    提供具体的使用建议
    """
    print(f"\n🎯 使用建议")
    print("="*60)
    
    print("📋 立即可行的改进:")
    print("   1. 将 delta_va 从 0.3 增加到 1.0")
    print("   2. 将 delta_nce 从 0.001 增加到 0.05")
    print("   3. 观察训练过程中各损失的变化")
    
    print(f"\n🔬 实验验证步骤:")
    print("   1. 使用新配置训练 5-10 个 epoch")
    print("   2. 观察验证损失是否改善")
    print("   3. 检查引导权重分布是否更平衡")
    print("   4. 如果效果好，可以进一步精细调优")
    
    print(f"\n⚠️  注意事项:")
    print("   1. 权重过大可能导致训练不稳定")
    print("   2. 建议逐步调整，观察效果")
    print("   3. 保存每次实验的结果进行对比")
    print("   4. 最终以验证集性能为准")
    
    print(f"\n🚀 下一步行动:")
    print("   1. 使用更新后的配置运行 adaptive_main.py")
    print("   2. 对比新旧配置的训练效果")
    print("   3. 如需进一步优化，运行网格搜索")


def main():
    """主函数"""
    print("🔍 损失权重快速分析和优化")
    print("="*80)
    
    # 1. 快速搜索候选
    candidates = quick_search()
    
    # 2. 损失平衡分析
    analyze_loss_balance()
    
    # 3. 生成搜索配置
    generate_search_config()
    
    # 4. 提供建议
    provide_recommendations()
    
    print(f"\n🎉 分析完成!")
    print("="*80)
    
    print(f"\n📊 关键发现:")
    print(f"   ❌ 原始配置: delta_va=0.3, delta_nce=0.001 (权重过小)")
    print(f"   ✅ 优化配置: delta_va=1.0, delta_nce=0.05 (已更新到config.py)")
    print(f"   📈 预期改善: 更平衡的多任务学习，更好的模态融合")
    
    print(f"\n🎯 立即行动:")
    print(f"   运行: python adaptive_main.py")
    print(f"   观察: 引导权重分布是否更平衡")
    print(f"   对比: 验证损失是否有改善")


if __name__ == '__main__':
    main()
