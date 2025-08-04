import random
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config
from train.Adaptive_TVA_train import AdaptiveTVA_train_fusion, AdaptiveTVA_test_fusion
from utils import Metrics
from data_loader import MOSEIDataloader


def main():
    """
    自适应TVA融合模型的主函数
    
    核心创新：集成自适应引导模块(AGM)的完整训练和测试流程
    
    创新特点：
    1. 动态选举向导模态，而非固定使用文本
    2. 记录和分析引导权重的分布情况
    3. 支持引导权重的可视化和统计分析
    4. 保持与原始模型的完全兼容性
    """
    
    print("="*60)
    print("🚀 Adaptive TVA Fusion Model - MFON Enhanced")
    print("="*60)
    print("核心创新：自适应引导模块 (Adaptive Guidance Module)")
    print("从'固定向导'到'动态选举'的自适应引导机制")
    print("="*60)

    # ========== 配置验证和摘要 ==========
    config.MOSEI.downStream.adaptiveTVAtrain.print_config_summary()

    # 验证配置
    is_valid, warnings = config.MOSEI.downStream.adaptiveTVAtrain.validate_config()
    if not is_valid:
        print("❌ 配置验证失败，请检查配置文件")
        return

    if warnings:
        print("⚠️  发现配置警告，但可以继续运行")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    # ========== 数据加载 ==========
    batch_size = config.MOSEI.downStream.batch_size
    print('📊 加载数据...')
    
    train_data = MOSEIDataloader('train', config.MOSEI.path.raw_data_path, batch_size=batch_size)
    print('✅ 训练数据加载完成')
    
    valid_data = MOSEIDataloader('valid', config.MOSEI.path.raw_data_path, batch_size=batch_size, shuffle=False)
    print('✅ 验证数据加载完成')
    
    test_data = MOSEIDataloader('test', config.MOSEI.path.raw_data_path, batch_size=batch_size, shuffle=False)
    print('✅ 测试数据加载完成')
    
    # ========== 评估指标初始化 ==========
    metrics = Metrics()
    
    # ========== 设置随机种子 ==========
    config.seed = 1111
    print(f'🎲 随机种子设置为: {config.seed}')
    
    # ========== 模型训练 ==========
    print("\n" + "="*60)
    print("🔥 开始自适应TVA融合模型训练")
    print("="*60)
    
    # 训练自适应模型并获取训练总结
    training_summary = AdaptiveTVA_train_fusion(
        config, metrics, config.seed, train_data, valid_data
    )

    print("✅ 训练完成！")

    # ========== 早停信息显示 ==========
    if training_summary['stopped_early']:
        print(f"\n🛑 训练提前停止信息:")
        print(f"   停止轮数: {training_summary['early_stop_epoch']}")
        print(f"   最佳指标: {training_summary['best_monitor_score']:.6f}")
        print(f"   节省时间: {config.MOSEI.downStream.adaptiveTVAtrain.epoch - training_summary['total_epochs']} epochs")
    else:
        print(f"\n⏰ 完整训练信息:")
        print(f"   完成轮数: {training_summary['total_epochs']}")
        print(f"   最佳损失: {training_summary['best_loss']:.6f}")

    # ========== 引导权重分析 ==========
    guidance_stats_history = training_summary['guidance_stats_history']
    if guidance_stats_history:
        print("\n" + "="*60)
        print("📈 引导权重分布分析")
        print("="*60)
        
        # 分析最终epoch的权重分布
        final_stats = guidance_stats_history[-1]
        print(f"最终权重分布:")
        print(f"  📝 文本权重: {final_stats['mean_text_weight']:.3f}")
        print(f"  👁️  视觉权重: {final_stats['mean_vision_weight']:.3f}")
        print(f"  🔊 音频权重: {final_stats['mean_audio_weight']:.3f}")
        
        print(f"\n主导模态统计:")
        print(f"  📝 文本主导样本: {final_stats['text_dominant_count']}")
        print(f"  👁️  视觉主导样本: {final_stats['vision_dominant_count']}")
        print(f"  🔊 音频主导样本: {final_stats['audio_dominant_count']}")
        
        # 计算权重分布的多样性
        text_ratio = final_stats['text_dominant_count'] / final_stats['total_samples']
        vision_ratio = final_stats['vision_dominant_count'] / final_stats['total_samples']
        audio_ratio = final_stats['audio_dominant_count'] / final_stats['total_samples']
        
        print(f"\n主导模态比例:")
        print(f"  📝 文本主导: {text_ratio:.1%}")
        print(f"  👁️  视觉主导: {vision_ratio:.1%}")
        print(f"  🔊 音频主导: {audio_ratio:.1%}")
        
        # 分析自适应性
        if vision_ratio > 0.1 or audio_ratio > 0.1:
            print("\n🎉 检测到显著的自适应行为！")
            print("   模型成功学会了在不同情况下选择不同的向导模态")
        else:
            print("\n📋 模型主要依赖文本作为向导")
            print("   这可能表明当前数据集中文本信息最为重要")
    
    # ========== 模型测试 ==========
    print("\n" + "="*60)
    print("🧪 开始自适应TVA融合模型测试")
    print("="*60)
    
    AdaptiveTVA_test_fusion(config, metrics, test_data)
    
    print("✅ 测试完成！")
    
    # ========== 创新总结 ==========
    print("\n" + "="*60)
    print("🌟 创新总结")
    print("="*60)
    print("1. ✨ 自适应引导模块 (AGM):")
    print("   - 动态选举向导模态，突破固定文本向导的限制")
    print("   - 基于MoE和GRU门控机制的创新设计")
    
    print("\n2. 🎯 核心优势:")
    print("   - 处理极端情况：语气/表情夸张而文字平淡")
    print("   - 提高模型鲁棒性和适应性")
    print("   - 智能早停机制：防止过拟合，节省训练时间")
    print("   - 保持与原始架构的完全兼容")
    
    print("\n3. 📊 可观测性:")
    print("   - 实时监控引导权重分布")
    print("   - 支持模态选择行为的可视化分析")
    print("   - 提供详细的统计信息和日志")
    
    print("\n" + "="*60)
    print("🎊 自适应TVA融合模型运行完成！")
    print("="*60)


def analyze_guidance_behavior():
    """
    分析引导权重行为的辅助函数
    
    可以用于深入分析模型的自适应行为，
    包括不同类型样本的权重分布模式等。
    """
    print("🔍 引导行为分析功能")
    print("此功能可用于:")
    print("1. 分析不同情感强度样本的权重分布")
    print("2. 识别模型的自适应模式")
    print("3. 可视化权重变化趋势")
    print("4. 对比原始模型和自适应模型的行为差异")


def compare_with_original():
    """
    与原始TVA模型的对比分析
    
    可以实现原始模型和自适应模型的并行训练和对比，
    量化自适应机制带来的性能提升。
    """
    print("⚖️  模型对比分析功能")
    print("此功能可用于:")
    print("1. 并行训练原始模型和自适应模型")
    print("2. 对比两个模型在相同数据上的表现")
    print("3. 分析自适应机制的有效性")
    print("4. 识别自适应模型的优势场景")


if __name__ == '__main__':
    main()
    
    # 可选：运行分析功能
    print("\n" + "="*40)
    print("📋 可选分析功能")
    print("="*40)
    analyze_guidance_behavior()
    print()
    compare_with_original()
