"""
损失权重理论分析和推荐搜索脚本

基于损失函数的理论分析，提供更科学的搜索策略
"""

import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)


class LossWeightAnalyzer:
    """
    损失权重分析器
    
    提供理论分析和搜索建议
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_loss_components(self):
        """
        分析各个损失组件的特性
        """
        print("📊 损失组件分析")
        print("="*60)
        
        analysis = {
            'main_loss': {
                'type': 'MSE回归损失',
                'typical_range': [0.1, 2.0],
                'description': '主任务损失，情感回归的核心目标',
                'weight': 1.0,
                'priority': 'highest'
            },
            'distillation_loss': {
                'type': 'KL散度损失',
                'typical_range': [0.01, 0.5],
                'description': '知识蒸馏损失，保持预训练知识',
                'weight_range': [0.1, 1.0],
                'priority': 'high'
            },
            'contrastive_loss': {
                'type': 'InfoNCE损失',
                'typical_range': [0.5, 5.0],
                'description': '对比学习损失，增强模态对齐',
                'weight_range': [0.001, 0.1],
                'priority': 'medium'
            }
        }
        
        for loss_name, info in analysis.items():
            print(f"\n🔍 {loss_name}:")
            print(f"   类型: {info['type']}")
            print(f"   典型范围: {info['typical_range']}")
            print(f"   描述: {info['description']}")
            if 'weight_range' in info:
                print(f"   推荐权重范围: {info['weight_range']}")
            print(f"   优先级: {info['priority']}")
        
        self.analysis_results['loss_components'] = analysis
        return analysis
    
    def calculate_optimal_ratios(self):
        """
        基于理论分析计算最优权重比例
        """
        print(f"\n📐 最优权重比例计算")
        print("="*60)
        
        # 基于损失量级的理论分析
        main_loss_scale = 1.0      # MSE损失，基准
        kl_loss_scale = 0.1        # KL散度损失，通常较小
        nce_loss_scale = 2.0       # InfoNCE损失，通常较大
        
        # 计算平衡权重（使各损失项贡献相当）
        balanced_weights = {
            'main': 1.0,
            'delta_va': main_loss_scale / kl_loss_scale,    # ≈ 10.0
            'delta_nce': main_loss_scale / nce_loss_scale   # ≈ 0.5
        }
        
        # 考虑任务重要性的调整权重
        task_importance = {
            'main': 1.0,        # 主任务最重要
            'distillation': 0.3, # 知识蒸馏中等重要
            'contrastive': 0.1   # 对比学习辅助作用
        }
        
        adjusted_weights = {
            'main': 1.0,
            'delta_va': balanced_weights['delta_va'] * task_importance['distillation'],
            'delta_nce': balanced_weights['delta_nce'] * task_importance['contrastive']
        }
        
        print(f"理论平衡权重:")
        print(f"   主任务: {balanced_weights['main']}")
        print(f"   知识蒸馏: {balanced_weights['delta_va']:.3f}")
        print(f"   对比学习: {balanced_weights['delta_nce']:.3f}")
        
        print(f"\n任务重要性调整权重:")
        print(f"   主任务: {adjusted_weights['main']}")
        print(f"   知识蒸馏: {adjusted_weights['delta_va']:.3f}")
        print(f"   对比学习: {adjusted_weights['delta_nce']:.3f}")
        
        self.analysis_results['optimal_ratios'] = {
            'balanced': balanced_weights,
            'adjusted': adjusted_weights
        }
        
        return adjusted_weights
    
    def generate_search_recommendations(self):
        """
        生成搜索建议
        """
        print(f"\n🎯 搜索建议")
        print("="*60)
        
        # 基于理论分析的搜索空间
        recommendations = {
            'coarse_search': {
                'delta_va': [0.1, 0.3, 0.5, 1.0, 2.0],
                'delta_nce': [0.001, 0.01, 0.05, 0.1, 0.2],
                'description': '粗粒度搜索，覆盖广泛范围'
            },
            'fine_search': {
                'delta_va': [0.2, 0.25, 0.3, 0.35, 0.4],
                'delta_nce': [0.005, 0.01, 0.02, 0.03, 0.05],
                'description': '细粒度搜索，基于当前最佳结果'
            },
            'theory_based': {
                'delta_va': [2.0, 2.5, 3.0, 3.5, 4.0],
                'delta_nce': [0.03, 0.05, 0.07, 0.1, 0.15],
                'description': '基于理论分析的搜索空间'
            }
        }
        
        for search_type, config in recommendations.items():
            print(f"\n📋 {search_type}:")
            print(f"   描述: {config['description']}")
            print(f"   delta_va: {config['delta_va']}")
            print(f"   delta_nce: {config['delta_nce']}")
            print(f"   组合数: {len(config['delta_va']) * len(config['delta_nce'])}")
        
        self.analysis_results['search_recommendations'] = recommendations
        return recommendations
    
    def analyze_current_settings(self):
        """
        分析当前配置的合理性
        """
        print(f"\n🔍 当前配置分析")
        print("="*60)
        
        # 当前配置
        current_config = {
            'delta_va': 0.3,
            'delta_nce': 0.001
        }
        
        # 理论最优配置
        optimal_config = self.analysis_results.get('optimal_ratios', {}).get('adjusted', {})
        
        print(f"当前配置:")
        print(f"   delta_va: {current_config['delta_va']}")
        print(f"   delta_nce: {current_config['delta_nce']}")
        
        if optimal_config:
            print(f"\n理论最优配置:")
            print(f"   delta_va: {optimal_config['delta_va']:.3f}")
            print(f"   delta_nce: {optimal_config['delta_nce']:.3f}")
            
            # 计算差异
            va_ratio = current_config['delta_va'] / optimal_config['delta_va']
            nce_ratio = current_config['delta_nce'] / optimal_config['delta_nce']
            
            print(f"\n配置差异分析:")
            print(f"   delta_va 比例: {va_ratio:.2f} ({'偏低' if va_ratio < 0.8 else '偏高' if va_ratio > 1.2 else '合理'})")
            print(f"   delta_nce 比例: {nce_ratio:.2f} ({'偏低' if nce_ratio < 0.8 else '偏高' if nce_ratio > 1.2 else '合理'})")
        
        # 给出建议
        suggestions = []
        if current_config['delta_va'] < 0.5:
            suggestions.append("考虑增加 delta_va 以加强知识蒸馏")
        if current_config['delta_nce'] < 0.01:
            suggestions.append("考虑增加 delta_nce 以加强对比学习")
        
        if suggestions:
            print(f"\n💡 优化建议:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print(f"\n✅ 当前配置在合理范围内")
    
    def create_search_script(self, search_type='coarse_search'):
        """
        创建定制化的搜索脚本
        """
        recommendations = self.analysis_results.get('search_recommendations', {})
        if search_type not in recommendations:
            print(f"❌ 未知的搜索类型: {search_type}")
            return
        
        config = recommendations[search_type]
        
        script_content = f'''"""
定制化损失权重搜索脚本 - {search_type}

{config['description']}
"""

# 搜索空间配置
SEARCH_SPACE = {{
    'delta_va': {config['delta_va']},
    'delta_nce': {config['delta_nce']}
}}

# 使用方法:
# 1. 将此配置复制到 loss_weight_grid_search.py 中的 search_space
# 2. 运行: python loss_weight_grid_search.py

print("🔍 {search_type} 搜索配置:")
print(f"delta_va 候选值: {{SEARCH_SPACE['delta_va']}}")
print(f"delta_nce 候选值: {{SEARCH_SPACE['delta_nce']}}")
print(f"总组合数: {{len(SEARCH_SPACE['delta_va']) * len(SEARCH_SPACE['delta_nce'])}}")
'''
        
        filename = f"search_config_{search_type}.py"
        with open(filename, 'w') as f:
            f.write(script_content)
        
        print(f"\n📄 搜索配置已保存到: {filename}")
    
    def run_full_analysis(self):
        """
        运行完整分析
        """
        print("🧮 损失权重理论分析")
        print("="*80)
        
        # 1. 分析损失组件
        self.analyze_loss_components()
        
        # 2. 计算最优比例
        self.calculate_optimal_ratios()
        
        # 3. 生成搜索建议
        self.generate_search_recommendations()
        
        # 4. 分析当前配置
        self.analyze_current_settings()
        
        # 5. 创建搜索脚本
        self.create_search_script('coarse_search')
        self.create_search_script('theory_based')
        
        print(f"\n🎉 分析完成!")
        print(f"建议的搜索策略:")
        print(f"1. 先运行 coarse_search 找到大致范围")
        print(f"2. 基于结果运行 fine_search 精细调优")
        print(f"3. 可以尝试 theory_based 搜索验证理论")


def main():
    """主函数"""
    analyzer = LossWeightAnalyzer()
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
