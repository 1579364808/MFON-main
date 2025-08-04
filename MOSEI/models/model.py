import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
sys.path.append(os.path.dirname(path))

from classifier import BaseClassifier
from Text_encoder import TextEncoder
from Vision_encoder import VisionEncoder
from Audio_encoder  import AudioEncoder
from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier


def check_dir(path):
    """
    检查并创建目录

    Args:
        path: 目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path)


class MLPLayer(nn.Module):
    """
    多层感知机层

    使用1D卷积实现的MLP层，等价于线性变换但支持序列处理。
    注意：当前实现中is_Fusion参数没有实际作用，两个分支完全相同。
    """

    def __init__(self, dim, embed_dim, is_Fusion=False):
        """
        初始化MLP层

        Args:
            dim: 输入维度
            embed_dim: 输出维度
            is_Fusion: 是否用于融合（当前版本中未使用）
        """
        super().__init__()
        # 注意：两个分支完全相同，is_Fusion参数实际未起作用
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()  # GELU激活函数

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            激活后的输出张量
        """
        return self.act(self.conv(x))



class TVA_fusion(nn.Module):
    """
    TVA多模态融合模型

    这是MFON架构的核心模型，实现了文本(T)、视觉(V)、音频(A)三模态的深度融合。

    主要创新点：
    1. 跨模态注意力机制：使用文本作为Query，视觉/音频作为Key-Value进行交互
    2. 可学习提示符：为视觉和音频模态添加可学习的提示向量
    3. 知识蒸馏：使用预训练的单模态编码器作为教师模型
    4. 对比学习：通过InfoNCE损失增强模态间的语义对齐

    架构组成：
    - 文本编码器：基于BERT的文本理解
    - 跨模态Transformer：文本-视觉、文本-音频交互
    - 冻结编码器：用于知识蒸馏的教师模型
    - 融合分类器：最终的情感预测

    损失函数：
    - 主任务损失：情感回归的MSE损失
    - 知识蒸馏损失：KL散度损失
    - 对比学习损失：InfoNCE损失
    """

    def __init__(self, config):
        """
        初始化TVA融合模型

        Args:
            config: 配置对象，包含所有超参数设置
        """
        super(TVA_fusion, self).__init__()
        self.config = config

        # ========== 基础配置参数 ==========
        self.text_dropout = config.MOSEI.downStream.text_drop_out  # 0.3: 文本dropout率

        encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim  # 768: 统一编码器维度

        # 音频-文本交互的Transformer配置
        audio_text_nhead = config.MOSEI.downStream.audio_text_nhead              # 8: 注意力头数
        audio_text_tf_num_layers = config.MOSEI.downStream.audio_text_tf_num_layers  # 3: Transformer层数

        self.audio_fea_dim = config.MOSEI.downStream.audio_fea_dim  # 74: 原始音频特征维度
        self.a_len = config.MOSEI.downStream.audio_seq_len          # 500: 音频序列长度

        # 视觉-文本交互的Transformer配置
        vision_text_nhead = config.MOSEI.downStream.vision_text_nhead              # 8: 注意力头数
        vision_text_tf_num_layers = config.MOSEI.downStream.vision_text_tf_num_layers  # 2: Transformer层数

        attn_dropout = config.MOSEI.downStream.drop_out  # 0.1: 注意力dropout率
        attn_mask = config.MOSEI.downStream.attn_mask    # True: 使用注意力掩码

        # 各模态的原始特征维度
        audio_fea_dim = config.MOSEI.downStream.audio_fea_dim    # 74: 音频特征维度
        vision_fea_dim = config.MOSEI.downStream.vision_fea_dim  # 35: 视觉特征维度
        text_fea_dim = config.MOSEI.downStream.text_fea_dim      # 768: 文本特征维度(BERT)

        # 序列长度配置
        self.vlen, self.alen = config.MOSEI.downStream.vlen, config.MOSEI.downStream.alen  # 500, 500

        # ========== 可学习提示符 (Learnable Prompts) ==========
        # 为音频和视觉模态添加可学习的提示向量，增强跨模态交互
        self.use_learnable_vectors = config.MOSEI.downStream.use_learnable_vectors

        # 根据配置决定是否初始化可学习向量
        if self.use_learnable_vectors:
            self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))  # [500, 768] 音频提示
            self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))  # [500, 768] 视觉提示
            print(f"✅ 使用可学习向量: 音频向量 {self.prompta_m.shape}, 视觉向量 {self.promptv_m.shape}")
        else:
            self.prompta_m = None
            self.promptv_m = None
            print("❌ 不使用可学习向量")

        # ========== 文本处理模块 ==========
        self.text_encoder = TextEncoder(config=config)  # BERT文本编码器
        self.proj_t = nn.Linear(text_fea_dim, encoder_fea_dim)  # 文本投影层: 768→768

        # ========== 视觉处理模块 ==========
        self.proj_v = nn.Linear(vision_fea_dim, encoder_fea_dim)  # 视觉投影层: 35→768

        # 视觉-文本跨模态Transformer
        # Query: 文本表示, Key&Value: 视觉表示
        self.vision_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim,                    # 768: 嵌入维度
            num_heads=vision_text_nhead,                  # 8: 注意力头数
            layers=vision_text_tf_num_layers,             # 2: Transformer层数
            attn_dropout=attn_dropout,                    # 0.1: 各种dropout率
            relu_dropout=attn_dropout,
            res_dropout=attn_dropout,
            embed_dropout=attn_dropout,
            attn_mask=attn_mask                           # True: 使用注意力掩码
        )

        # ========== 音频处理模块 ==========
        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)  # 音频投影层: 74→768

        # 音频-文本跨模态Transformer
        # Query: 文本表示, Key&Value: 音频表示
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim,                    # 768: 嵌入维度
            num_heads=audio_text_nhead,                   # 8: 注意力头数
            layers=audio_text_tf_num_layers,              # 3: Transformer层数(比视觉更深)
            attn_dropout=attn_dropout,                    # 0.1: 各种dropout率
            relu_dropout=attn_dropout,
            res_dropout=attn_dropout,
            embed_dropout=attn_dropout,
            attn_mask=attn_mask                           # True: 使用注意力掩码
        )

        # 删除知识蒸馏模块 - 不再使用

        # ========== 创新点：自适应引导模块 (Adaptive Guidance Module, AGM) ==========
        # 门控网络：动态选举最适合担任向导的模态
        # 灵感来源：Mixture of Experts (MoE) 和 门控循环单元 (GRU)
        self.gating_network = nn.Sequential(
            nn.Linear(encoder_fea_dim * 3, 128),  # 输入：三模态句子级特征拼接 [bs, 2304] → [bs, 128]
            nn.ReLU(),                            # 非线性激活
            nn.Dropout(0.1),                      # 防止过拟合
            nn.Linear(128, 64),                   # 进一步降维 [bs, 128] → [bs, 64]
            nn.ReLU(),
            nn.Linear(64, 3),                     # 输出三个模态的权重 [bs, 64] → [bs, 3]
            nn.Softmax(dim=-1)                    # 确保权重和为1，形成概率分布
        )

        # ========== 最终融合分类器 ==========
        # 将三模态特征融合后进行情感预测
        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,                                      # 2304: 三模态拼接(768×3)
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],  # [768, 384, 96]: 渐进降维
            output_size=1                                                        # 1: 情感回归值
        )

        # ========== 训练相关配置 ==========
        self.device = config.DEVICE                                    # 设备配置
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'  # 模型保存路径
        check_dir(self.model_path)  # 确保保存目录存在

    # 删除load_froze函数 - 不再需要加载冻结编码器
       
    def forward(self, text, vision, audio, mode='train'):
        """
        TVA融合模型的前向传播

        Args:
            text: 文本输入，字符串列表，例如 ["I love this movie", ...]
            vision: 视觉特征 [batch_size, 500, 35]
            audio: 音频特征 [batch_size, 500, 74]
            mode: 模式，'train'时计算所有损失，'eval'时只返回预测

        Returns:
            pred: 情感预测值 [batch_size]
            losses: 训练模式下返回 (loss_v, loss_a, loss_nce)，推理模式下返回 None

        处理流程：
        1. 文本编码：BERT编码 + 投影 + dropout
        2. 跨模态交互：文本作为Query，与视觉/音频进行注意力交互
        3. 特征融合：拼接三模态的句子级表示
        4. 情感预测：通过分类器得到最终预测
        5. 损失计算：知识蒸馏损失 + 对比学习损失
        """

        # ========== 步骤1: 文本编码和处理 ==========
        # 通过BERT获得文本的序列表示
        last_hidden_text = self.text_encoder(text)  # [bs, seq_len, 768]

        # 文本投影和dropout：维度转换 + 正则化
        # permute(1,0,2): [bs, seq, 768] → [seq, bs, 768] (适配Transformer格式)
        last_hidden_text = F.dropout(
            self.proj_t(last_hidden_text.permute(1, 0, 2)),
            p=self.text_dropout,  # 0.3
            training=self.training
        )  # [seq, bs, 768]

        # 提取文本的句子级表示（取第一个位置，通常是[CLS]）
        x_t_embed_initial = last_hidden_text[0]  # [bs, 768]

        # ========== 步骤1.5: 预处理其他模态特征（用于门控网络） ==========
        # 视觉和音频特征的初步处理，获取句子级表示用于门控网络
        proj_vision_initial = self.proj_v(vision)  # [bs, 500, 768]
        proj_audio_initial = self.proj_a(audio)    # [bs, 500, 768]

        # 对视觉和音频序列进行平均池化，得到句子级表示
        x_v_embed_initial = proj_vision_initial.mean(dim=1)  # [bs, 768]
        x_a_embed_initial = proj_audio_initial.mean(dim=1)   # [bs, 768]

        # ========== 创新点：自适应引导机制 (Adaptive Guidance Module, AGM) ==========
        #
        # 问题背景：
        # MFON原始设计假设文本永远是最好的"向导"（Query），但当语气或表情极度夸张
        # 而文字内容平淡时，这个假设会失效。
        #
        # 解决方案：
        # 设计门控网络动态选举最适合担任向导的模态，从"固定向导"升级为"动态选举"
        #
        # 理论基础：
        # 1. Mixture of Experts (MoE)：门控网络根据输入选择最合适的"专家"
        # 2. 门控循环单元 (GRU)：动态控制信息流的门控机制

        # 步骤1: 准备门控网络的输入
        # 将三个模态的句子级特征拼接，形成全局上下文表示
        gating_input = torch.cat([x_t_embed_initial, x_v_embed_initial, x_a_embed_initial], dim=-1)  # [bs, 2304]

        # 步骤2: 动态选举向导模态
        # 门控网络输出三个权重 [ω_t, ω_v, ω_a]，代表每个模态作为"向导"的置信度
        # 权重满足：ω_t + ω_v + ω_a = 1（通过Softmax保证）
        modality_weights = self.gating_network(gating_input)  # [bs, 3]
        omega_t, omega_v, omega_a = modality_weights[:, 0], modality_weights[:, 1], modality_weights[:, 2]  # [bs]

        # 步骤3: 构建动态Query
        # 不再使用固定的文本作为Query，而是使用加权组合：
        # Q_dynamic = ω_t * f^t + ω_v * f^v + ω_a * f^a

        # 扩展权重维度以支持广播乘法 [bs] → [bs, 1, 1]
        omega_t_expanded = omega_t.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]
        omega_v_expanded = omega_v.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]
        omega_a_expanded = omega_a.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1]

        # 准备序列格式的特征用于动态Query构建
        text_seq = last_hidden_text  # [seq_t, bs, 768]

        # 创建动态Query（以文本序列长度为准，保持与原架构的兼容性）
        # 对于视觉和音频，使用句子级表示扩展到序列长度
        dynamic_query = (
            omega_t_expanded * text_seq.permute(1, 0, 2) +  # 文本序列 * 文本权重
            omega_v_expanded * x_v_embed_initial.unsqueeze(1).expand(-1, text_seq.size(0), -1) +  # 视觉表示扩展
            omega_a_expanded * x_a_embed_initial.unsqueeze(1).expand(-1, text_seq.size(0), -1)    # 音频表示扩展
        )
        dynamic_query = dynamic_query.permute(1, 0, 2)  # [seq_t, bs, 768]

        # 动态Query的优势：
        # 1. 当文本信息丰富时，ω_t 较大，主要依赖文本引导
        # 2. 当表情夸张时，ω_v 较大，视觉模态主导跨模态交互
        # 3. 当语气强烈时，ω_a 较大，音频模态成为主要向导

        # ========== 步骤2: 视觉-动态Query跨模态交互 ==========
        # 视觉特征投影并根据配置添加可学习提示符
        proj_vision = self.proj_v(vision).permute(1, 0, 2)  # [bs, 500, 35] → [500, bs, 768]
        if self.use_learnable_vectors and self.promptv_m is not None:
            proj_vision = proj_vision + self.promptv_m.unsqueeze(1)  # 添加可学习视觉向量
        # [500, bs, 768] + [500, 1, 768] → [500, bs, 768] (如果使用可学习向量)

        # 跨模态注意力：Query=动态Query, Key&Value=视觉
        h_tv = self.vision_with_text(dynamic_query, proj_vision, proj_vision)
        # 输入: Q=[seq_t, bs, 768], K=V=[500, bs, 768]
        # 输出: [seq_t, bs, 768] (文本序列长度保持不变)

        # ========== 步骤3: 音频-动态Query跨模态交互 ==========
        # 音频特征投影并根据配置添加可学习提示符
        proj_audio = self.proj_a(audio).permute(1, 0, 2)  # [bs, 500, 74] → [500, bs, 768]
        if self.use_learnable_vectors and self.prompta_m is not None:
            proj_audio = proj_audio + self.prompta_m.unsqueeze(1)  # 添加可学习音频向量
        # [500, bs, 768] + [500, 1, 768] → [500, bs, 768] (如果使用可学习向量)

        # 跨模态注意力：Query=动态Query, Key&Value=音频
        h_ta = self.audio_with_text(dynamic_query, proj_audio, proj_audio)
        # 输入: Q=[seq_t, bs, 768], K=V=[500, bs, 768]
        # 输出: [seq_t, bs, 768] (文本序列长度保持不变)

        # ========== 步骤4: 提取各模态的句子级表示 ==========
        x_t_embed = dynamic_query[0]  # 动态Query的句子级表示 [bs, 768]
        x_v_embed = h_tv[0]          # 视觉增强的动态表示 [bs, 768]
        x_a_embed = h_ta[0]          # 音频增强的动态表示 [bs, 768]

        # ========== 步骤5: 多模态特征融合 ==========
        # 拼接三个模态的句子级表示
        x = torch.cat([x_t_embed, x_v_embed, x_a_embed], dim=-1)  # [bs, 2304]

        # 通过融合分类器得到最终的情感预测
        pred = self.TVA_decoder(x).view(-1)  # [bs, 1] → [bs]

        # ========== 步骤6: 损失计算（仅训练模式） ==========
        loss_v = loss_a = 0
        loss_nce = 0

        # 删除知识蒸馏和对比学习 - 只返回预测结果
        if mode == 'train':
            return pred, None  # 不再返回额外损失
        else:
            return pred, None
        
    def save_model(self, name=None):
        """
        保存TVA融合模型

        Args:
            name: 保存文件的前缀名，默认为None时使用默认命名

        保存策略：
        - 保存完整的模型状态字典
        - 包含所有可训练参数（文本编码器、跨模态Transformer、分类器等）
        - 不包含冻结的教师模型（需要单独加载）
        """
        # 构建保存路径
        if name == None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = self.model_path + str(name) + 'TVA_fusion' + '_model.pt'

        print('model saved at:\n', mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name=None):
        """
        加载TVA融合模型

        Args:
            name: 加载文件的路径或前缀名
                 - None: 使用默认路径
                 - 字符串路径: 直接使用该路径
                 - 前缀名: 构建标准路径

        加载策略：
        - 使用过滤加载，只加载匹配的参数
        - strict=False 允许部分参数不匹配
        - 适应模型结构的变化和版本兼容性
        """
        # 构建加载路径
        if name == None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = name  # 直接使用提供的路径

        print('model loaded from:\n', mode_path)

        # 安全加载策略：过滤不匹配的参数
        checkpoint = torch.load(mode_path, map_location=self.device)
        model_state_dict = self.state_dict()

        # 只加载当前模型中存在的参数
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}

        # 非严格加载，允许部分参数缺失
        self.load_state_dict(filtered_checkpoint, strict=False)

    # 删除所有损失函数 - 不再使用知识蒸馏和对比学习
    
    
