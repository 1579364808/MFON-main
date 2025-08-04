#!/usr/bin/env python3
"""
可学习向量配置使用示例

该脚本展示如何在不同数据集上使用可学习向量配置
"""

def example_mosi_with_learnable_vectors():
    """MOSI数据集启用可学习向量的示例"""
    print("📋 示例1: MOSI数据集启用可学习向量")
    print("-" * 50)
    
    # 修改MOSI配置文件中的设置
    config_example = """
    # 在 MOSI/config.py 中设置
    class downStream:
        # ... 其他配置 ...
        
        # ========== 可学习向量配置 ==========
        use_learnable_vectors = True  # 启用可学习向量
    """
    
    print("🔧 配置文件修改:")
    print(config_example)
    
    print("📊 预期效果:")
    print("   ✅ 使用可学习向量增强跨模态交互")
    print("   ✅ 音频向量: [375, 768] = 288,000 参数")
    print("   ✅ 视觉向量: [500, 768] = 384,000 参数")
    print("   ✅ 总计增加: 672,000 个参数")
    print()

def example_mosi_without_learnable_vectors():
    """MOSI数据集禁用可学习向量的示例"""
    print("📋 示例2: MOSI数据集禁用可学习向量")
    print("-" * 50)
    
    # 修改MOSI配置文件中的设置
    config_example = """
    # 在 MOSI/config.py 中设置
    class downStream:
        # ... 其他配置 ...
        
        # ========== 可学习向量配置 ==========
        use_learnable_vectors = False  # 禁用可学习向量
    """
    
    print("🔧 配置文件修改:")
    print(config_example)
    
    print("📊 预期效果:")
    print("   ✅ 不使用可学习向量，减少参数数量")
    print("   ✅ 减少 672,000 个参数")
    print("   ✅ 降低过拟合风险")
    print("   ✅ 加快训练速度")
    print()

def example_training_comparison():
    """训练对比示例"""
    print("📋 示例3: 训练对比实验")
    print("-" * 50)
    
    training_script = """
    # 实验1: 启用可学习向量
    # 修改 MOSI/config.py: use_learnable_vectors = True
    cd MOSI
    python main.py  # 训练并记录结果
    
    # 实验2: 禁用可学习向量  
    # 修改 MOSI/config.py: use_learnable_vectors = False
    python main.py  # 训练并记录结果
    
    # 对比两次实验的结果
    """
    
    print("🚀 实验步骤:")
    print(training_script)
    
    print("📊 对比指标:")
    print("   1. 验证集准确率/F1分数")
    print("   2. 训练时间（每个epoch）")
    print("   3. GPU内存使用量")
    print("   4. 模型参数总数")
    print("   5. 收敛速度")
    print()

def example_all_datasets():
    """所有数据集配置示例"""
    print("📋 示例4: 所有数据集配置")
    print("-" * 50)
    
    configs = {
        "MOSI": {
            "file": "MOSI/config.py",
            "path": "MOSI.downStream.use_learnable_vectors",
            "audio_params": "375 × 768 = 288,000",
            "vision_params": "500 × 768 = 384,000",
            "total": "672,000"
        },
        "MOSEI": {
            "file": "MOSEI/config.py", 
            "path": "MOSEI.downStream.use_learnable_vectors",
            "audio_params": "500 × 768 = 384,000",
            "vision_params": "500 × 768 = 384,000", 
            "total": "768,000"
        },
        "SIMS": {
            "file": "SIMS/config.py",
            "path": "SIMS.downStream.use_learnable_vectors",
            "audio_params": "400 × 768 = 307,200",
            "vision_params": "55 × 768 = 42,240",
            "total": "349,440"
        }
    }
    
    for dataset, info in configs.items():
        print(f"🔧 {dataset} 数据集:")
        print(f"   配置文件: {info['file']}")
        print(f"   配置路径: {info['path']}")
        print(f"   音频参数: {info['audio_params']}")
        print(f"   视觉参数: {info['vision_params']}")
        print(f"   总计参数: {info['total']}")
        print()

def example_model_output():
    """模型输出示例"""
    print("📋 示例5: 模型初始化输出")
    print("-" * 50)
    
    print("🔧 启用可学习向量时的输出:")
    print("   ✅ 使用可学习向量: 音频向量 torch.Size([375, 768]), 视觉向量 torch.Size([500, 768])")
    print()
    
    print("🔧 禁用可学习向量时的输出:")
    print("   ❌ 不使用可学习向量")
    print()

def main():
    """主函数"""
    print("🚀 可学习向量配置使用示例")
    print("=" * 80)
    print()
    
    # 运行所有示例
    example_mosi_with_learnable_vectors()
    example_mosi_without_learnable_vectors()
    example_training_comparison()
    example_all_datasets()
    example_model_output()
    
    print("💡 使用建议:")
    print("   1. 首次实验建议启用可学习向量 (use_learnable_vectors = True)")
    print("   2. 如果出现过拟合，尝试禁用可学习向量 (use_learnable_vectors = False)")
    print("   3. 通过对比实验量化可学习向量的贡献")
    print("   4. 在资源受限的情况下，可以禁用可学习向量以节省内存")
    print()
    
    print("📚 更多信息请参考: LEARNABLE_VECTORS_CONFIG_README.md")
    print("🧪 运行测试脚本: python test_learnable_vectors_config.py")

if __name__ == '__main__':
    main()
