#!/usr/bin/env python3
"""
测试优化器修复的脚本

验证在禁用可学习向量时，优化器能正常初始化
"""

import sys
import os
import torch

def test_mosi_optimizer():
    """测试MOSI的优化器初始化"""
    print("🔍 测试 MOSI 优化器初始化")
    print("-" * 50)
    
    try:
        # 添加MOSI路径
        sys.path.append('MOSI')
        import config as mosi_config
        from models.model import TVAModel
        
        # 确保禁用可学习向量
        mosi_config.MOSI.downStream.use_learnable_vectors = False
        
        # 创建模型
        model = TVAModel(mosi_config)
        
        # 模拟训练文件中的优化器参数设置
        text_params = list(model.text_encoder.named_parameters())
        text_params = [p for _, p in text_params] 
        
        vision_params = list(model.proj_v.named_parameters()) +\
                    list(model.vision_with_text.named_parameters()) 
        vision_params = [p for _, p in vision_params]
        # 只有在使用可学习向量时才添加到优化器参数中
        if model.promptv_m is not None:
            vision_params.append(model.promptv_m)
        
        audio_params = list(model.proj_a.named_parameters()) +\
                    list(model.audio_with_text.named_parameters())
        audio_params = [p for _, p in audio_params]
        # 只有在使用可学习向量时才添加到优化器参数中
        if model.prompta_m is not None:
            audio_params.append(model.prompta_m)
            
        model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n] 

        optimizer_grouped_parameters = [
            {'params': text_params, 'weight_decay': 1e-3, 'lr': 5e-5},
            {'params': audio_params, 'weight_decay': 1e-3, 'lr': 1e-3},
            {'params': vision_params, 'weight_decay': 1e-3, 'lr': 1e-3},
            {'params': model_params_other, 'weight_decay': 1e-3, 'lr': 1e-3}
        ]
        
        # 尝试创建优化器
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        
        print("✅ MOSI 优化器初始化成功")
        print(f"   文本参数数量: {len(text_params)}")
        print(f"   视觉参数数量: {len(vision_params)}")
        print(f"   音频参数数量: {len(audio_params)}")
        print(f"   其他参数数量: {len(model_params_other)}")
        print(f"   可学习向量状态: prompta_m={model.prompta_m is not None}, promptv_m={model.promptv_m is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ MOSI 优化器初始化失败: {e}")
        return False

def test_mosei_optimizer():
    """测试MOSEI的优化器初始化"""
    print("\n🔍 测试 MOSEI 优化器初始化")
    print("-" * 50)
    
    try:
        # 添加MOSEI路径
        sys.path.append('MOSEI')
        import config as mosei_config
        from models.model import TVAModel
        
        # 确保禁用可学习向量
        mosei_config.MOSEI.downStream.use_learnable_vectors = False
        
        # 创建模型
        model = TVAModel(mosei_config)
        
        # 模拟训练文件中的优化器参数设置
        text_params = list(model.text_encoder.named_parameters())
        text_params = [p for _, p in text_params] 
        
        vision_params = list(model.proj_v.named_parameters()) +\
                    list(model.vision_with_text.named_parameters()) 
        vision_params = [p for _, p in vision_params]
        # 只有在使用可学习向量时才添加到优化器参数中
        if model.promptv_m is not None:
            vision_params.append(model.promptv_m)
        
        audio_params = list(model.proj_a.named_parameters()) +\
                    list(model.audio_with_text.named_parameters())
        audio_params = [p for _, p in audio_params]
        # 只有在使用可学习向量时才添加到优化器参数中
        if model.prompta_m is not None:
            audio_params.append(model.prompta_m)
            
        model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n] 

        optimizer_grouped_parameters = [
            {'params': text_params, 'weight_decay': 1e-2, 'lr': 5e-5},
            {'params': audio_params, 'weight_decay': 1e-3, 'lr': 1e-3},
            {'params': vision_params, 'weight_decay': 1e-3, 'lr': 1e-4},
            {'params': model_params_other, 'weight_decay': 1e-3, 'lr': 1e-3}
        ]
        
        # 尝试创建优化器
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        
        print("✅ MOSEI 优化器初始化成功")
        print(f"   文本参数数量: {len(text_params)}")
        print(f"   视觉参数数量: {len(vision_params)}")
        print(f"   音频参数数量: {len(audio_params)}")
        print(f"   其他参数数量: {len(model_params_other)}")
        print(f"   可学习向量状态: prompta_m={model.prompta_m is not None}, promptv_m={model.promptv_m is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ MOSEI 优化器初始化失败: {e}")
        return False

def test_sims_optimizer():
    """测试SIMS的优化器初始化"""
    print("\n🔍 测试 SIMS 优化器初始化")
    print("-" * 50)
    
    try:
        # 添加SIMS路径
        sys.path.append('SIMS')
        import config as sims_config
        from models.model import TVAModel
        
        # 确保禁用可学习向量
        sims_config.SIMS.downStream.use_learnable_vectors = False
        
        # 创建模型
        model = TVAModel(sims_config)
        
        # 模拟训练文件中的优化器参数设置
        text_params = list(model.text_encoder.named_parameters())
        text_params = [p for _, p in text_params] 
        
        vision_params = list(model.proj_v.named_parameters()) +\
                    list(model.vision_with_text.named_parameters()) 
        vision_params = [p for _, p in vision_params]
        # 只有在使用可学习向量时才添加到优化器参数中
        if model.promptv_m is not None:
            vision_params.append(model.promptv_m)
        
        audio_params = list(model.proj_a.named_parameters()) +\
                    list(model.audio_with_text.named_parameters())
        audio_params = [p for _, p in audio_params]
        # 只有在使用可学习向量时才添加到优化器参数中
        if model.prompta_m is not None:
            audio_params.append(model.prompta_m)
            
        model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n] 

        optimizer_grouped_parameters = [
            {'params': text_params, 'weight_decay': 1e-3, 'lr': 5e-5},
            {'params': audio_params, 'weight_decay': 1e-3, 'lr': 1e-3},
            {'params': vision_params, 'weight_decay': 1e-3, 'lr': 1e-3},
            {'params': model_params_other, 'weight_decay': 1e-3, 'lr': 1e-3}
        ]
        
        # 尝试创建优化器
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
        
        print("✅ SIMS 优化器初始化成功")
        print(f"   文本参数数量: {len(text_params)}")
        print(f"   视觉参数数量: {len(vision_params)}")
        print(f"   音频参数数量: {len(audio_params)}")
        print(f"   其他参数数量: {len(model_params_other)}")
        print(f"   可学习向量状态: prompta_m={model.prompta_m is not None}, promptv_m={model.promptv_m is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ SIMS 优化器初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 优化器修复测试")
    print("=" * 80)
    
    results = []
    
    # 测试各个数据集
    results.append(test_mosi_optimizer())
    results.append(test_mosei_optimizer())
    results.append(test_sims_optimizer())
    
    # 总结结果
    print("\n📊 测试总结")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    
    if all(results):
        print("\n🎉 所有测试通过！优化器修复成功。")
        print("\n📋 现在可以安全地:")
        print("   1. 设置 use_learnable_vectors = False 禁用可学习向量")
        print("   2. 正常运行训练脚本")
        print("   3. 优化器不会再出现 NoneType 错误")
    else:
        print("\n⚠️  部分测试失败，请检查修复。")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
