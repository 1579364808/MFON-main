"""
早停机制测试脚本

用于验证早停机制的正确性和配置的有效性。
"""

import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config
from train.Adaptive_TVA_train import EarlyStopping
import torch
import torch.nn as nn


class DummyModel(nn.Module):
    """用于测试的虚拟模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_early_stopping_config():
    """测试早停配置的有效性"""
    print("🧪 测试早停配置...")
    
    # 验证配置
    is_valid, warnings = config.MOSEI.downStream.adaptiveTVAtrain.validate_config()
    
    print(f"配置有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
    
    if warnings:
        print("⚠️  配置警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # 显示早停相关配置
    adaptive_config = config.MOSEI.downStream.adaptiveTVAtrain
    print(f"\n📋 早停配置:")
    print(f"  启用早停: {adaptive_config.early_stopping}")
    print(f"  耐心值: {adaptive_config.early_stopping_patience}")
    print(f"  最小改善阈值: {adaptive_config.early_stopping_min_delta}")
    print(f"  监控指标: {adaptive_config.early_stopping_monitor}")
    print(f"  监控模式: {adaptive_config.early_stopping_mode}")
    print(f"  恢复最佳权重: {adaptive_config.early_stopping_restore_best}")


def test_early_stopping_mechanism():
    """测试早停机制的工作流程"""
    print("\n🔬 测试早停机制...")
    
    # 创建早停对象
    early_stopping = EarlyStopping(
        patience=2,
        min_delta=1e-6,
        monitor='val_loss',
        mode='min',
        restore_best_weights=True,
        verbose=True
    )
    
    # 创建虚拟模型
    model = DummyModel()
    
    # 模拟训练过程中的验证损失变化
    val_losses = [1.0, 0.8, 0.6, 0.65, 0.67, 0.68, 0.69]  # 前3个epoch改善，后4个epoch恶化
    
    print(f"模拟验证损失序列: {val_losses}")
    print(f"预期行为: 前3个epoch改善，第6个epoch触发早停 (耐心值=2)")
    print()
    
    for epoch, val_loss in enumerate(val_losses, 1):
        should_stop = early_stopping(val_loss, model, epoch)
        
        if should_stop:
            print(f"✅ 早停机制正常工作！在第 {epoch} epoch触发早停")
            break
    else:
        print("❌ 早停机制未按预期工作")
    
    # 测试权重恢复
    print(f"\n🔄 测试权重恢复...")
    early_stopping.restore_best_model(model)


def test_different_monitor_modes():
    """测试不同的监控模式"""
    print("\n📊 测试不同监控模式...")
    
    # 测试 'max' 模式 (准确率等指标)
    early_stopping_max = EarlyStopping(
        patience=2,
        min_delta=1e-6,
        monitor='val_acc',
        mode='max',
        restore_best_weights=False,
        verbose=True
    )
    
    model = DummyModel()
    
    # 模拟准确率变化 (越大越好)
    val_accs = [0.6, 0.7, 0.8, 0.75, 0.73, 0.72]  # 前3个epoch改善，后3个epoch恶化
    
    print(f"模拟验证准确率序列: {val_accs}")
    print(f"监控模式: max (越大越好)")
    print()
    
    for epoch, val_acc in enumerate(val_accs, 1):
        should_stop = early_stopping_max(val_acc, model, epoch)
        
        if should_stop:
            print(f"✅ 'max'模式早停机制正常工作！在第 {epoch} epoch触发早停")
            break
    else:
        print("❌ 'max'模式早停机制未按预期工作")


def test_edge_cases():
    """测试边界情况"""
    print("\n🎯 测试边界情况...")
    
    # 测试耐心值为1的情况
    early_stopping_impatient = EarlyStopping(
        patience=1,
        min_delta=0.0,
        monitor='val_loss',
        mode='min',
        restore_best_weights=False,
        verbose=True
    )
    
    model = DummyModel()
    val_losses = [1.0, 1.1, 0.9]  # 第2个epoch恶化，第3个epoch改善
    
    print(f"测试耐心值=1的情况:")
    print(f"验证损失序列: {val_losses}")
    print()
    
    for epoch, val_loss in enumerate(val_losses, 1):
        should_stop = early_stopping_impatient(val_loss, model, epoch)
        
        if should_stop:
            print(f"✅ 耐心值=1的早停在第 {epoch} epoch触发")
            break
    else:
        print("❌ 耐心值=1的早停未按预期工作")


def main():
    """主测试函数"""
    print("="*60)
    print("🧪 自适应TVA早停机制测试")
    print("="*60)
    
    # 测试配置
    test_early_stopping_config()
    
    # 测试机制
    test_early_stopping_mechanism()
    
    # 测试不同模式
    test_different_monitor_modes()
    
    # 测试边界情况
    test_edge_cases()
    
    print("\n" + "="*60)
    print("🎉 早停机制测试完成！")
    print("="*60)
    
    print("\n📝 使用建议:")
    print("1. 对于损失函数，使用 mode='min'")
    print("2. 对于准确率/F1等指标，使用 mode='max'")
    print("3. 耐心值建议设置为2-5，避免过早停止")
    print("4. 建议启用 restore_best_weights 以获得最佳性能")
    print("5. 可以根据数据集大小调整 min_delta 阈值")


if __name__ == '__main__':
    main()
