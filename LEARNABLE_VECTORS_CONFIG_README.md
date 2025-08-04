# 可学习向量配置说明

## 📋 概述

本文档说明如何在MFON项目中配置和使用可学习向量（Learnable Prompt Vectors）。可学习向量是用于增强跨模态交互的可训练参数，您现在可以通过配置文件轻松启用或禁用它们。

## 🔧 配置选项

### 配置位置

在每个数据集的配置文件中，都添加了 `use_learnable_vectors` 配置项：

- **MOSI**: `MOSI/config.py` → `MOSI.downStream.use_learnable_vectors`
- **MOSEI**: `MOSEI/config.py` → `MOSEI.downStream.use_learnable_vectors`  
- **SIMS**: `SIMS/config.py` → `SIMS.downStream.use_learnable_vectors`

### 配置值

```python
# 启用可学习向量（默认）
use_learnable_vectors = True

# 禁用可学习向量
use_learnable_vectors = False
```

## 📊 技术细节

### 可学习向量的作用

可学习向量是添加到音频和视觉特征上的可训练参数：

```python
# 音频可学习向量: [audio_seq_len, encoder_fea_dim]
self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))

# 视觉可学习向量: [vision_seq_len, encoder_fea_dim]  
self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))
```

### 参数数量影响

**MOSI数据集**:
- 音频向量: 375 × 768 = 288,000 参数
- 视觉向量: 500 × 768 = 384,000 参数
- **总计**: 672,000 个额外参数

**MOSEI数据集**:
- 音频向量: 500 × 768 = 384,000 参数
- 视觉向量: 500 × 768 = 384,000 参数
- **总计**: 768,000 个额外参数

**SIMS数据集**:
- 音频向量: 400 × 768 = 307,200 参数
- 视觉向量: 55 × 768 = 42,240 参数
- **总计**: 349,440 个额外参数

## 🚀 使用方法

### 1. 启用可学习向量

```python
# 在对应的config.py文件中设置
use_learnable_vectors = True
```

**效果**:
- ✅ 使用可学习提示向量增强跨模态交互
- ✅ 可能获得更好的模型性能
- ❌ 增加模型参数数量和训练时间

### 2. 禁用可学习向量

```python
# 在对应的config.py文件中设置
use_learnable_vectors = False
```

**效果**:
- ✅ 减少模型参数数量
- ✅ 降低过拟合风险
- ✅ 加快训练速度
- ❌ 可能略微降低模型性能

## 🧪 实验建议

### 对比实验

建议进行以下对比实验来评估可学习向量的效果：

1. **基线实验**: 使用 `use_learnable_vectors = True` 训练模型
2. **对照实验**: 使用 `use_learnable_vectors = False` 训练模型
3. **性能对比**: 比较两种配置的验证集性能

### 评估指标

- **准确率/F1分数**: 主要性能指标
- **训练时间**: 每个epoch的训练时间
- **内存使用**: GPU内存占用
- **参数数量**: 模型总参数数

## 📝 代码修改说明

### 配置文件修改

在每个数据集的 `config.py` 中添加：

```python
# ========== 可学习向量配置 ==========
# 控制是否使用可学习提示向量 (Learnable Prompt Vectors)
use_learnable_vectors = True  # True: 使用可学习向量, False: 不使用可学习向量
```

### 模型文件修改

在模型初始化中：

```python
# 根据配置决定是否初始化可学习向量
self.use_learnable_vectors = config.DATASET.downStream.use_learnable_vectors

if self.use_learnable_vectors:
    self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))
    self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))
else:
    self.prompta_m = None
    self.promptv_m = None
```

在前向传播中：

```python
# 视觉特征处理
proj_vision = self.proj_v(vision).permute(1, 0, 2)
if self.use_learnable_vectors and self.promptv_m is not None:
    proj_vision = proj_vision + self.promptv_m.unsqueeze(1)

# 音频特征处理  
proj_audio = self.proj_a(audio).permute(1, 0, 2)
if self.use_learnable_vectors and self.prompta_m is not None:
    proj_audio = proj_audio + self.prompta_m.unsqueeze(1)
```

## 🔍 测试验证

### 配置测试
运行配置测试脚本验证配置是否正确：

```bash
python test_learnable_vectors_config.py
```

该脚本会：
1. 测试每个数据集的配置
2. 验证参数数量差异
3. 确认模型能正常初始化

### 优化器修复测试
运行优化器测试脚本验证修复是否成功：

```bash
python test_optimizer_fix.py
```

该脚本会：
1. 测试禁用可学习向量时的优化器初始化
2. 验证不会出现 NoneType 错误
3. 确认训练能正常进行

## 💡 最佳实践

1. **首次实验**: 建议先使用默认的 `use_learnable_vectors = True`
2. **性能优化**: 如果模型过拟合，尝试设置为 `False`
3. **资源受限**: 在GPU内存不足时，可以禁用可学习向量
4. **消融研究**: 通过对比实验量化可学习向量的贡献

## ⚠️ 注意事项

1. **模型兼容性**: 更改配置后需要重新训练模型
2. **预训练模型**: 如果使用预训练模型，确保配置一致
3. **保存/加载**: 不同配置的模型不能互相加载权重
4. **实验记录**: 记录每次实验的配置以便复现结果
5. **优化器修复**: 已修复禁用可学习向量时的优化器 NoneType 错误

## 📞 支持

如果在使用过程中遇到问题，请检查：
1. 配置文件语法是否正确
2. 模型是否重新初始化
3. 是否有相关的错误日志

---

**更新日期**: 2025-08-04  
**版本**: 1.0  
**适用范围**: MOSI, MOSEI, SIMS 数据集
