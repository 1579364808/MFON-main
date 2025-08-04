# 🔍 损失权重网格搜索指南

## 📋 概述

本指南介绍如何使用网格搜索来优化自适应TVA模型的损失权重参数 `delta_va` 和 `delta_nce`。

## 🎯 损失函数解析

### 总损失函数
```python
total_loss = main_loss + delta_va * (loss_v + loss_a) + delta_nce * loss_nce
```

### 各项损失说明
1. **主任务损失 (main_loss)**: MSE回归损失，用于情感预测
2. **知识蒸馏损失 (loss_v + loss_a)**: KL散度损失，保持预训练知识
3. **对比学习损失 (loss_nce)**: InfoNCE损失，增强模态间语义对齐

### 权重参数作用
- **delta_va**: 控制知识蒸馏的强度
  - 过大：可能导致过拟合到预训练知识
  - 过小：可能丢失预训练的有用信息
  
- **delta_nce**: 控制模态间对齐的强度
  - 过大：可能干扰主任务学习
  - 过小：模态间对齐不足，融合效果差

## 🚀 使用方法

### 1. 快速搜索 (推荐)
```bash
cd MOSEI
python quick_loss_search.py
```

**特点**:
- 测试6个精心选择的参数组合
- 每个组合训练8个epoch
- 约30-60分钟完成
- 适合快速验证和调优

**测试组合**:
1. **当前默认**: delta_va=0.3, delta_nce=0.001
2. **增强蒸馏**: delta_va=0.5, delta_nce=0.001
3. **增强对比**: delta_va=0.3, delta_nce=0.01
4. **平衡组合**: delta_va=0.4, delta_nce=0.005
5. **保守组合**: delta_va=0.2, delta_nce=0.001
6. **激进组合**: delta_va=0.8, delta_nce=0.05

### 2. 完整网格搜索
```bash
cd MOSEI
python loss_weight_grid_search.py
```

**特点**:
- 测试所有参数组合 (7×7=49个组合)
- 每个组合训练8个epoch
- 约4-8小时完成
- 适合全面优化

**搜索空间**:
- delta_va: [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
- delta_nce: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

## 📊 结果分析

### 输出文件
搜索完成后会生成以下文件：

1. **quick_search_YYYYMMDD_HHMMSS/**
   - `quick_search_results.json`: 详细结果
   - `best_config.txt`: 最佳配置参数

2. **loss_weight_search_YYYYMMDD_HHMMSS/**
   - `final_results.json`: 完整搜索结果
   - `intermediate_results.json`: 中间结果
   - `best_config.txt`: 最佳配置参数

### 关键指标
- **best_loss**: 验证集上的最佳损失值 (越小越好)
- **total_epochs**: 实际训练轮数
- **stopped_early**: 是否触发早停
- **guidance_weights**: AGM的权重分布

## 🔧 配置应用

### 1. 查看最佳结果
```bash
# 查看最佳配置
cat quick_search_*/best_config.txt
```

### 2. 更新配置文件
将最佳参数复制到 `config.py` 的 `adaptiveTVAtrain` 类中：

```python
class adaptiveTVAtrain:
    # ... 其他配置 ...
    
    # 使用搜索得到的最佳权重
    delta_va = 0.4      # 替换为搜索结果
    delta_nce = 0.005   # 替换为搜索结果
```

### 3. 验证效果
```bash
# 使用新配置训练完整模型
python adaptive_main.py
```

## 📈 搜索策略建议

### 初次搜索
1. 先运行 `quick_loss_search.py` 获得初步结果
2. 观察哪种策略效果最好
3. 根据结果调整搜索范围

### 精细搜索
1. 基于快速搜索结果，缩小搜索范围
2. 在最佳结果附近进行更密集的搜索
3. 考虑增加训练轮数以获得更稳定的结果

### 搜索空间调整
如果需要调整搜索空间，修改脚本中的 `search_space` 或 `test_combinations`：

```python
# 在 loss_weight_grid_search.py 中
self.search_space = {
    'delta_va': [0.2, 0.3, 0.4, 0.5],      # 缩小范围
    'delta_nce': [0.001, 0.005, 0.01]      # 缩小范围
}

# 在 quick_loss_search.py 中
test_combinations = [
    {'name': '自定义1', 'delta_va': 0.35, 'delta_nce': 0.003, 'description': '...'},
    # 添加更多自定义组合
]
```

## ⚠️ 注意事项

### 计算资源
- 快速搜索: 约需要2-4GB GPU内存，30-60分钟
- 完整搜索: 约需要2-4GB GPU内存，4-8小时

### 搜索设置
- 搜索时使用较少的epoch (8轮) 以提高效率
- 启用早停机制防止过拟合
- 使用相同的随机种子确保可重复性

### 结果解释
- 验证损失是主要评估指标
- 考虑训练稳定性 (是否频繁早停)
- 观察AGM权重分布的合理性

## 🎯 预期效果

### 成功的搜索应该显示
1. **明显的性能差异**: 不同参数组合有显著的损失差异
2. **稳定的训练**: 最佳组合训练过程稳定
3. **合理的权重分布**: AGM权重分布符合预期

### 可能的结果模式
- **知识蒸馏主导**: delta_va较大时效果好，说明预训练知识重要
- **对比学习主导**: delta_nce较大时效果好，说明模态对齐重要
- **平衡最优**: 中等权重组合效果最好，说明需要平衡各项损失

## 🔄 迭代优化

1. **第一轮**: 使用快速搜索确定大致范围
2. **第二轮**: 在最佳结果附近进行精细搜索
3. **第三轮**: 使用最佳参数进行完整训练验证
4. **最终**: 在测试集上评估最终性能

通过系统的网格搜索，您可以找到最适合您数据集的损失权重组合，从而最大化自适应TVA模型的性能！🎉
