"""
测试无损负载均衡功能

验证DeepSeek风格的负载均衡是否能有效解决模态权重不平衡问题
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config
from models.adaptive_tva_fusion import AdaptiveTVA_fusion
from data_loader import MOSEIDataloader


def test_load_balancing():
    """
    测试负载均衡功能的效果
    
    对比启用和未启用负载均衡时的权重分布差异
    """
    print("🧪 测试无损负载均衡功能")
    print("="*60)
    
    # 数据加载
    batch_size = 32
    train_data = MOSEIDataloader('train', config.MOSEI.path.raw_data_path, batch_size=batch_size)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试配置
    test_configs = [
        {
            'name': '未启用负载均衡',
            'enable_load_balancing': False,
            'color': 'red',
            'linestyle': '-'
        },
        {
            'name': '启用负载均衡 (update_rate=0.01)',
            'enable_load_balancing': True,
            'update_rate': 0.01,
            'target_balance_ratio': 0.15,
            'color': 'blue',
            'linestyle': '--'
        },
        {
            'name': '启用负载均衡 (update_rate=0.05)',
            'enable_load_balancing': True,
            'update_rate': 0.05,
            'target_balance_ratio': 0.2,
            'color': 'green',
            'linestyle': ':'
        }
    ]
    
    results = {}
    
    # 测试每种配置
    for test_config in test_configs:
        print(f"\n📊 测试配置: {test_config['name']}")
        print("-" * 40)
        
        # 创建模型
        model = AdaptiveTVA_fusion(
            config,
            enable_load_balancing=test_config['enable_load_balancing'],
            update_rate=test_config.get('update_rate', 0.01),
            target_balance_ratio=test_config.get('target_balance_ratio', 0.15)
        ).to(device)
        
        model.train()  # 确保在训练模式下测试负载均衡
        
        # 收集权重统计
        weight_history = []
        bias_history = []
        
        # 模拟训练过程
        num_batches = 50  # 测试50个batch
        
        for batch_idx, sample in enumerate(train_data):
            if batch_idx >= num_batches:
                break
                
            # 准备数据
            text = sample['raw_text']
            vision = sample['vision'].to(device).float()
            audio = sample['audio'].to(device).float()
            
            # 前向传播 (只关注AGM的权重输出)
            with torch.no_grad():
                # 获取编码器输出
                text_repr = model.text_encoder(text)  # [bs, 768]
                vision_repr = model.vision_encoder(vision)  # [bs, 768]
                audio_repr = model.audio_encoder(audio)  # [bs, 768]
                
                # 获取AGM权重
                guidance_weights = model.adaptive_guidance(text_repr, vision_repr, audio_repr)
                
                # 统计权重分布
                mean_weights = guidance_weights.mean(dim=0).cpu().numpy()
                weight_history.append(mean_weights)
                
                # 获取偏置信息 (如果启用负载均衡)
                if test_config['enable_load_balancing']:
                    load_stats = model.adaptive_guidance.get_load_statistics()
                    bias_history.append(load_stats['modality_bias'].copy())
            
            # 每10个batch打印一次统计
            if (batch_idx + 1) % 10 == 0:
                current_weights = weight_history[-1]
                print(f"  Batch {batch_idx+1:2d}: Text={current_weights[0]:.3f}, "
                      f"Vision={current_weights[1]:.3f}, Audio={current_weights[2]:.3f}")
                
                if test_config['enable_load_balancing']:
                    current_bias = bias_history[-1]
                    print(f"            Bias: Text={current_bias[0]:.4f}, "
                          f"Vision={current_bias[1]:.4f}, Audio={current_bias[2]:.4f}")
        
        # 保存结果
        results[test_config['name']] = {
            'weight_history': np.array(weight_history),
            'bias_history': np.array(bias_history) if bias_history else None,
            'config': test_config
        }
        
        # 打印最终统计
        final_weights = weight_history[-1]
        print(f"\n  最终权重分布:")
        print(f"    文本: {final_weights[0]:.3f} ({final_weights[0]*100:.1f}%)")
        print(f"    视觉: {final_weights[1]:.3f} ({final_weights[1]*100:.1f}%)")
        print(f"    音频: {final_weights[2]:.3f} ({final_weights[2]*100:.1f}%)")
        
        # 计算权重方差 (衡量平衡程度)
        weight_variance = np.var(final_weights)
        print(f"    权重方差: {weight_variance:.6f} (越小越平衡)")
    
    # 可视化结果
    plot_results(results)
    
    # 分析结果
    analyze_results(results)


def plot_results(results):
    """可视化权重变化过程"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('无损负载均衡效果对比', fontsize=16)
    
    # 子图1: 文本权重变化
    ax1 = axes[0, 0]
    for name, result in results.items():
        config = result['config']
        weights = result['weight_history']
        ax1.plot(weights[:, 0], label=name, color=config['color'], 
                linestyle=config['linestyle'], linewidth=2)
    ax1.set_title('文本模态权重变化')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('权重')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 视觉权重变化
    ax2 = axes[0, 1]
    for name, result in results.items():
        config = result['config']
        weights = result['weight_history']
        ax2.plot(weights[:, 1], label=name, color=config['color'], 
                linestyle=config['linestyle'], linewidth=2)
    ax2.set_title('视觉模态权重变化')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('权重')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 音频权重变化
    ax3 = axes[1, 0]
    for name, result in results.items():
        config = result['config']
        weights = result['weight_history']
        ax3.plot(weights[:, 2], label=name, color=config['color'], 
                linestyle=config['linestyle'], linewidth=2)
    ax3.set_title('音频模态权重变化')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('权重')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 权重方差变化 (平衡程度)
    ax4 = axes[1, 1]
    for name, result in results.items():
        config = result['config']
        weights = result['weight_history']
        # 计算每个batch的权重方差
        variances = np.var(weights, axis=1)
        ax4.plot(variances, label=name, color=config['color'], 
                linestyle=config['linestyle'], linewidth=2)
    ax4.set_title('权重方差变化 (越小越平衡)')
    ax4.set_xlabel('Batch')
    ax4.set_ylabel('方差')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('load_balancing_test_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 可视化结果已保存到: load_balancing_test_results.png")


def analyze_results(results):
    """分析测试结果"""
    
    print("\n" + "="*60)
    print("📈 负载均衡效果分析")
    print("="*60)
    
    for name, result in results.items():
        weights = result['weight_history']
        
        print(f"\n🔍 {name}:")
        
        # 初始和最终权重对比
        initial_weights = weights[0]
        final_weights = weights[-1]
        
        print(f"  初始权重: Text={initial_weights[0]:.3f}, Vision={initial_weights[1]:.3f}, Audio={initial_weights[2]:.3f}")
        print(f"  最终权重: Text={final_weights[0]:.3f}, Vision={final_weights[1]:.3f}, Audio={final_weights[2]:.3f}")
        
        # 权重变化量
        weight_changes = final_weights - initial_weights
        print(f"  权重变化: Text={weight_changes[0]:+.3f}, Vision={weight_changes[1]:+.3f}, Audio={weight_changes[2]:+.3f}")
        
        # 平衡程度评估
        initial_variance = np.var(initial_weights)
        final_variance = np.var(final_weights)
        variance_reduction = (initial_variance - final_variance) / initial_variance * 100
        
        print(f"  初始方差: {initial_variance:.6f}")
        print(f"  最终方差: {final_variance:.6f}")
        print(f"  方差减少: {variance_reduction:+.1f}%")
        
        # 平衡效果评级
        if final_variance < 0.01:
            balance_grade = "优秀"
        elif final_variance < 0.05:
            balance_grade = "良好"
        elif final_variance < 0.1:
            balance_grade = "一般"
        else:
            balance_grade = "较差"
        
        print(f"  平衡评级: {balance_grade}")


if __name__ == '__main__':
    test_load_balancing()
