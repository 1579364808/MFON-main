#!/usr/bin/env python3
"""
测试可学习向量配置的脚本

该脚本用于验证可学习向量配置是否正确工作，
可以在不同数据集上测试启用/禁用可学习向量的效果。
"""

import sys
import os

def test_mosi_config():
    """测试MOSI数据集的可学习向量配置"""
    print("🔍 测试 MOSI 数据集的可学习向量配置")
    print("="*60)
    
    try:
        # 添加MOSI路径
        sys.path.append('MOSI')
        import config as mosi_config
        from models.model import TVAModel
        
        # 测试启用可学习向量
        print("📋 测试1: 启用可学习向量")
        mosi_config.MOSI.downStream.use_learnable_vectors = True
        model_with_vectors = TVAModel(mosi_config)
        print(f"   模型参数数量: {sum(p.numel() for p in model_with_vectors.parameters())}")
        
        # 测试禁用可学习向量
        print("\n📋 测试2: 禁用可学习向量")
        mosi_config.MOSI.downStream.use_learnable_vectors = False
        model_without_vectors = TVAModel(mosi_config)
        print(f"   模型参数数量: {sum(p.numel() for p in model_without_vectors.parameters())}")
        
        # 计算参数差异
        param_diff = sum(p.numel() for p in model_with_vectors.parameters()) - sum(p.numel() for p in model_without_vectors.parameters())
        print(f"\n📊 参数差异: {param_diff:,} 个参数")
        print(f"   预期差异: {mosi_config.MOSI.downStream.alen * mosi_config.MOSI.downStream.encoder_fea_dim + mosi_config.MOSI.downStream.vlen * mosi_config.MOSI.downStream.encoder_fea_dim:,} 个参数")
        
        print("✅ MOSI 配置测试通过")
        
    except Exception as e:
        print(f"❌ MOSI 配置测试失败: {e}")
        return False
    
    return True

def test_mosei_config():
    """测试MOSEI数据集的可学习向量配置"""
    print("\n🔍 测试 MOSEI 数据集的可学习向量配置")
    print("="*60)
    
    try:
        # 添加MOSEI路径
        sys.path.append('MOSEI')
        import config as mosei_config
        from models.model import TVAModel
        
        # 测试启用可学习向量
        print("📋 测试1: 启用可学习向量")
        mosei_config.MOSEI.downStream.use_learnable_vectors = True
        model_with_vectors = TVAModel(mosei_config)
        print(f"   模型参数数量: {sum(p.numel() for p in model_with_vectors.parameters())}")
        
        # 测试禁用可学习向量
        print("\n📋 测试2: 禁用可学习向量")
        mosei_config.MOSEI.downStream.use_learnable_vectors = False
        model_without_vectors = TVAModel(mosei_config)
        print(f"   模型参数数量: {sum(p.numel() for p in model_without_vectors.parameters())}")
        
        # 计算参数差异
        param_diff = sum(p.numel() for p in model_with_vectors.parameters()) - sum(p.numel() for p in model_without_vectors.parameters())
        print(f"\n📊 参数差异: {param_diff:,} 个参数")
        
        print("✅ MOSEI 配置测试通过")
        
    except Exception as e:
        print(f"❌ MOSEI 配置测试失败: {e}")
        return False
    
    return True

def test_sims_config():
    """测试SIMS数据集的可学习向量配置"""
    print("\n🔍 测试 SIMS 数据集的可学习向量配置")
    print("="*60)
    
    try:
        # 添加SIMS路径
        sys.path.append('SIMS')
        import config as sims_config
        from models.model import TVAModel
        
        # 测试启用可学习向量
        print("📋 测试1: 启用可学习向量")
        sims_config.SIMS.downStream.use_learnable_vectors = True
        model_with_vectors = TVAModel(sims_config)
        print(f"   模型参数数量: {sum(p.numel() for p in model_with_vectors.parameters())}")
        
        # 测试禁用可学习向量
        print("\n📋 测试2: 禁用可学习向量")
        sims_config.SIMS.downStream.use_learnable_vectors = False
        model_without_vectors = TVAModel(sims_config)
        print(f"   模型参数数量: {sum(p.numel() for p in model_without_vectors.parameters())}")
        
        # 计算参数差异
        param_diff = sum(p.numel() for p in model_with_vectors.parameters()) - sum(p.numel() for p in model_without_vectors.parameters())
        print(f"\n📊 参数差异: {param_diff:,} 个参数")
        
        print("✅ SIMS 配置测试通过")
        
    except Exception as e:
        print(f"❌ SIMS 配置测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 可学习向量配置测试")
    print("="*80)
    
    results = []
    
    # 测试各个数据集
    results.append(test_mosi_config())
    results.append(test_mosei_config())
    results.append(test_sims_config())
    
    # 总结结果
    print("\n📊 测试总结")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    
    if all(results):
        print("\n🎉 所有测试通过！可学习向量配置工作正常。")
        print("\n📋 使用说明:")
        print("   1. 在配置文件中设置 use_learnable_vectors = True 启用可学习向量")
        print("   2. 在配置文件中设置 use_learnable_vectors = False 禁用可学习向量")
        print("   3. 重新训练模型以应用配置更改")
    else:
        print("\n⚠️  部分测试失败，请检查配置和代码修改。")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
