
        
import torch
from torch import nn
import torch.nn.functional as F
import os
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier

class VisionEncoder(nn.Module):
    """
    视觉编码器模块

    该模块用于将原始视觉特征编码为高级语义表示，主要用于：
    1. 视觉模态的预训练
    2. 作为TVA融合模型中的视觉编码组件
    3. 在融合训练中作为冻结的教师模型提供知识蒸馏

    架构设计：
    - 线性投影层：将35维视觉特征映射到768维统一空间
    - Transformer编码器：使用自注意力机制建模视觉序列的时序依赖关系
    - 输出：取第一个时间步的表示作为整个视觉序列的句子级表示

    视觉特征说明：
    - 输入特征通常是从视频帧中提取的高级特征（如CNN特征、人脸特征等）
    - 35维特征可能包含：面部表情、头部姿态、眼神方向、动作特征等
    - 500个时间步对应视频的时序信息
    """

    def __init__(self, config):
        """
        初始化视觉编码器

        Args:
            config: 配置对象，包含所有超参数设置
        """
        super(VisionEncoder, self).__init__()

        # ========== 从配置文件中获取超参数 ==========
        self.vision_fea_dim = config.MOSEI.downStream.vision_fea_dim        # 35:  原始视觉特征维度
        self.vision_seq_len = config.MOSEI.downStream.vision_seq_len        # 500: 视觉序列长度

        self.encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim      # 768: 统一编码器特征维度
        self.vision_nhead = config.MOSEI.downStream.vision_nhead            # 8:   多头注意力的头数
        self.vision_tf_num_layers = config.MOSEI.downStream.vision_tf_num_layers  # 4: Transformer层数
        self.attn_dropout = config.MOSEI.downStream.vision_drop_out         # 0.0: 注意力dropout率
        self.attn_mask = config.MOSEI.downStream.vision_attn_mask           # False: 是否使用注意力掩码

        # ========== 网络层定义 ==========
        # 线性投影层：将原始视觉特征(35维)映射到统一的编码器维度(768维)
        # 这样可以与文本和音频特征在同一空间中进行处理
        self.proj_a = nn.Linear(self.vision_fea_dim, self.encoder_fea_dim)

        # Transformer编码器：用于建模视觉序列的时序依赖关系
        # 使用自注意力机制捕获不同时间步之间的视觉关联性
        self.trans_encoder_a = TransformerEncoder(
            embed_dim=self.encoder_fea_dim,         # 768: 嵌入维度
            num_heads=self.vision_nhead,            # 8:   多头注意力头数
            layers=self.vision_tf_num_layers,       # 4:   Transformer层数（比音频更深）
            attn_dropout=self.attn_dropout,         # 0.0: 注意力dropout
            relu_dropout=self.attn_dropout,         # 0.0: ReLU dropout
            res_dropout=self.attn_dropout,          # 0.0: 残差连接dropout
            embed_dropout=self.attn_dropout,        # 0.0: 嵌入dropout
            attn_mask=self.attn_mask                # False: 注意力掩码
        )

    def forward(self, vision):
        """
        前向传播

        Args:
            vision: 输入视觉特征 [batch_size, seq_len, vision_fea_dim]
                   形状为 [bs, 500, 35]
                   注意：参数名应该是vision而不是audio

        Returns:
            last_h_v: 视觉的句子级表示 [batch_size, encoder_fea_dim]
                     形状为 [bs, 768]

        处理流程：
        1. 线性投影：35维 → 768维
        2. 维度转换：适配Transformer输入格式
        3. Transformer编码：建模时序依赖关系（4层，比音频更深）
        4. 提取表示：取第一个时间步作为句子级表示
        """

        # 步骤1: 线性投影 - 将视觉特征映射到统一的编码器维度
        # 输入: [bs, 500, 35] → 输出: [bs, 500, 768]
        proj_x_v = self.proj_a(vision)  # 注意：这里应该处理vision而不是audio

        # 步骤2: 维度转换 - Transformer期望的输入格式为 [seq_len, batch_size, embed_dim]
        # [bs, 500, 768] → [500, bs, 768]
        proj_x_v = proj_x_v.permute(1, 0, 2)

        # 步骤3: Transformer编码 - 使用自注意力机制建模序列依赖关系
        # 输入: [500, bs, 768] → 输出: [500, bs, 768]
        # 视觉使用4层Transformer，比音频的2层更深，用于捕获更复杂的视觉模式
        h_vs = self.trans_encoder_a(proj_x_v)

        # 处理可能的元组返回值（某些Transformer实现会返回(output, attention_weights)）
        if type(h_vs) == tuple:
            h_vs = h_vs[0]  # 只取输出，忽略注意力权重

        # 步骤4: 提取句子级表示 - 取第一个时间步的输出作为整个序列的表示
        # [500, bs, 768] → [bs, 768]
        last_h_v = h_vs[0]   # 相当于 h_vs[0, :, :]

        return last_h_v

    def set_froze(self):
        """
        冻结模型参数

        在TVA融合训练中，预训练的视觉编码器会被冻结，
        作为教师模型提供知识蒸馏信号，指导融合模型的学习。
        冻结后的模型参数不会在反向传播中更新。
        """
        for param in self.parameters():
            param.requires_grad = False



# 删除VisionPretrain类 - 不再需要单模态预训练
    """
    视觉预训练模型

    该模型用于视觉模态的自监督预训练，主要目标是：
    1. 学习视觉特征的语义表示：通过情感回归任务让模型理解视觉内容的情感含义
    2. 为融合模型提供初始化：预训练的编码器作为TVA融合模型的初始化权重
    3. 知识蒸馏的教师模型：在融合训练中作为冻结的教师模型指导学习
    4. 视觉-情感映射：建立视觉特征与情感强度之间的映射关系

    架构组成：
    - VisionEncoder: 视觉编码器，将35维视觉特征编码为768维语义表示
    - BaseClassifier: 回归分类器，将编码特征映射到情感预测值

    训练策略：
    - 使用MOSEI数据集的情感标签作为监督信号
    - 通过MSE损失进行回归训练
    - 基于验证集损失保存最佳模型
    - 使用4层Transformer进行更深层的视觉特征学习
    """

    def __init__(self, config, encoder_fea_dim=None):
        """
        初始化视觉预训练模型

        Args:
            config: 配置对象，包含所有超参数设置
            encoder_fea_dim: 编码器特征维度，默认为None时使用配置文件中的值(768)
        """
        super(VisionPretrain, self).__init__()

        # 设置编码器特征维度，默认使用配置文件中的768维
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim  # 768

        # ========== 核心组件初始化 ==========
        # 视觉编码器：将原始视觉特征(35维×500序列)编码为语义表示(768维)
        # 使用4层Transformer，比音频的2层更深，用于捕获复杂的视觉模式
        self.encoder = VisionEncoder(config)

        # 回归分类器：将编码特征映射到情感预测值
        # 网络结构：768 → 384 → 96 → 1
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,                                    # 768: 输入维度
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],  # [384, 96]: 隐藏层维度
            output_size=1,                                                 # 1: 输出情感回归值
            name='VisionRegClassifier'                                     # 分类器名称标识
        )

        # ========== 训练相关配置 ==========
        self.device = config.DEVICE                    # 设备配置(CPU/GPU)
        self.criterion = torch.nn.MSELoss()           # MSE损失函数，用于回归任务
        self.config = config                          # 保存配置对象

        # 模型保存路径：根据随机种子创建独立的保存目录
        # 例如：save_models/uni_fea_encoder/MOSEI/1111/
        self.model_path = config.MOSEI.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)  # 确保目录存在

    def forward(self, vision, label, return_loss=True, device=None):
        """
        前向传播

        Args:
            vision: 输入视觉特征 [batch_size, seq_len, vision_fea_dim]
                   形状为 [bs, 500, 35]
                   注意：参数名应该是vision而不是audio
            label: 情感标签 [batch_size] 或 [batch_size, 1]
                  MOSEI数据集的回归标签，范围通常在[-3, 3]
            return_loss: 是否返回损失值，训练时为True，推理时为False
            device: 计算设备，默认使用初始化时的设备配置

        Returns:
            如果return_loss=True: (pred, loss)
                pred: 情感预测值 [batch_size]
                loss: MSE损失标量值
            如果return_loss=False: pred
                pred: 情感预测值 [batch_size]

        处理流程：
        1. 视觉编码：通过VisionEncoder将视觉特征编码为语义表示
        2. 情感预测：通过BaseClassifier将编码特征映射到情感值
        3. 损失计算：如果需要，计算预测值与真实标签的MSE损失

        视觉特征说明：
        - 35维特征可能包含面部表情、头部姿态、眼神等信息
        - 500个时间步对应视频的时序变化
        - 通过4层Transformer学习复杂的视觉-情感映射关系
        """
        # 设备配置检查
        if device is None:
            device = self.device

        # 步骤1: 视觉编码 - 将原始视觉特征编码为高级语义表示
        # 输入: [bs, 500, 35] → 输出: [bs, 768]
        x = self.encoder(vision)  # 修正：应该传入vision而不是audio

        # 步骤2: 情感预测 - 将编码特征映射到情感预测值
        # 输入: [bs, 768] → 输出: [bs, 1] → squeeze → [bs]
        pred = self.classifier(x).squeeze()

        # 步骤3: 损失计算（可选）
        if return_loss:
            # 计算预测值与真实标签之间的MSE损失
            # 确保两个张量的形状一致，都是[bs]
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, loss
        else:
            # 推理模式，只返回预测值
            return pred

    def save_model(self, name='best_loss'):
        """
        保存模型权重

        将视觉编码器和分类器的权重分别保存到不同的文件中，
        这样设计的好处是：
        1. 模块化保存：可以单独加载编码器用于融合模型
        2. 灵活性：可以只加载需要的部分（编码器或分类器）
        3. 复用性：编码器可以在不同任务中复用
        4. 知识蒸馏：编码器可以作为教师模型使用

        Args:
            name: 保存文件的前缀名，默认为'best_loss'
                 最终文件名格式：{name}_vision_encoder.pt 和 {name}_vision_decoder.pt

        保存文件：
        - 编码器权重：{model_path}/{name}_vision_encoder.pt
        - 分类器权重：{model_path}/{name}_vision_decoder.pt
        """
        # 构建保存路径
        encoder_path = self.model_path + name + '_vision_encoder.pt'
        decoder_path = self.model_path + name + '_vision_decoder.pt'

        # 分别保存编码器和分类器的状态字典
        torch.save(self.encoder.state_dict(), encoder_path)      # 保存视觉编码器权重
        torch.save(self.classifier.state_dict(), decoder_path)  # 保存回归分类器权重

        # 打印保存路径信息
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        """
        加载模型权重

        支持灵活的模型加载策略，可以选择性地加载不同的模块：
        1. 只加载编码器：用于融合模型的初始化
        2. 只加载分类器：用于特定的分类任务
        3. 加载完整模型：用于继续训练或完整推理

        Args:
            name: 加载文件的前缀名，默认为'best_loss'
                 对应的文件名：{name}_vision_encoder.pt 和 {name}_vision_decoder.pt
            module: 指定加载的模块，可选值：
                   - 'encoder': 只加载视觉编码器
                   - 'decoder': 只加载回归分类器
                   - 'all' 或 None: 加载完整模型（默认）

        使用场景：
        - module='encoder': 在TVA融合模型中初始化视觉编码器
        - module='decoder': 在特定分类任务中复用分类器
        - module='all': 完整的视觉预训练模型推理或继续训练
        """
        # 构建加载路径
        encoder_path = self.model_path + name + '_vision_encoder.pt'
        decoder_path = self.model_path + name + '_vision_decoder.pt'
        print('model loaded from:')

        # 根据module参数选择性加载
        if module == 'encoder':
            # 只加载视觉编码器权重
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)

        elif module == 'decoder':
            # 只加载回归分类器权重
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)

        elif module == 'all' or module is None:
            # 加载完整模型（编码器 + 分类器）
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)

