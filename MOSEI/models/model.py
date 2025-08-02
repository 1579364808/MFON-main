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
        self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))  # [500, 768] 音频提示
        self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))  # [500, 768] 视觉提示

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

        # ========== 知识蒸馏模块 ==========
        # 冻结的预训练编码器，用作教师模型
        self.vision_encoder_froze = VisionEncoder(config=config)  # 冻结的视觉编码器
        self.audio_encoder_froze = AudioEncoder(config=config)    # 冻结的音频编码器

        # ========== 最终融合分类器 ==========
        # 将三模态特征融合后进行情感预测
        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,                                      # 2304: 三模态拼接(768×3)
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],  # [768, 384, 96]: 渐进降维
            output_size=1                                                        # 1: 情感回归值
        )

        # ========== 训练相关配置 ==========
        self.device = config.DEVICE                                    # 设备配置
        self.criterion = nn.KLDivLoss(reduction='batchmean')          # KL散度损失(用于知识蒸馏)
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'  # 模型保存路径
        check_dir(self.model_path)  # 确保保存目录存在
     
    def load_froze(self):
        """
        加载并冻结预训练的单模态编码器

        这是知识蒸馏的关键步骤：
        1. 加载预训练的音频和视觉编码器权重
        2. 冻结这些编码器的参数，使其作为教师模型
        3. 在训练过程中，这些冻结的编码器提供蒸馏信号

        知识蒸馏的作用：
        - 保持单模态的专业知识
        - 指导融合模型学习更好的特征表示
        - 防止多模态训练中的模态偏移
        """
        # 构建预训练模型的加载路径
        model_path = self.config.MOSEI.path.encoder_path + str(self.config.seed) + '/'

        # 加载预训练的音频编码器权重
        self.audio_encoder_froze.load_state_dict(
            torch.load(model_path + 'best_loss_audio_encoder.pt', map_location=self.device)
        )

        # 加载预训练的视觉编码器权重
        self.vision_encoder_froze.load_state_dict(
            torch.load(model_path + 'best_loss_vision_encoder.pt', map_location=self.device)
        )

        # 冻结编码器参数，使其在训练中不更新
        self.audio_encoder_froze.set_froze()   # 设置requires_grad=False
        self.vision_encoder_froze.set_froze()  # 设置requires_grad=False
       
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
        x_t_embed = last_hidden_text[0]  # [bs, 768]

        # ========== 步骤2: 视觉-文本跨模态交互 ==========
        # 视觉特征投影并添加可学习提示符
        proj_vision = self.proj_v(vision).permute(1, 0, 2) + self.promptv_m.unsqueeze(1)
        # [bs, 500, 35] → [bs, 500, 768] → [500, bs, 768] + [500, 1, 768] → [500, bs, 768]

        # 跨模态注意力：Query=文本, Key&Value=视觉
        h_tv = self.vision_with_text(last_hidden_text, proj_vision, proj_vision)
        # 输入: Q=[seq_t, bs, 768], K=V=[500, bs, 768]
        # 输出: [seq_t, bs, 768] (文本序列长度保持不变)

        # ========== 步骤3: 音频-文本跨模态交互 ==========
        # 音频特征投影并添加可学习提示符
        proj_audio = self.proj_a(audio).permute(1, 0, 2) + self.prompta_m.unsqueeze(1)
        # [bs, 500, 74] → [bs, 500, 768] → [500, bs, 768] + [500, 1, 768] → [500, bs, 768]

        # 跨模态注意力：Query=文本, Key&Value=音频
        h_ta = self.audio_with_text(last_hidden_text, proj_audio, proj_audio)
        # 输入: Q=[seq_t, bs, 768], K=V=[500, bs, 768]
        # 输出: [seq_t, bs, 768] (文本序列长度保持不变)

        # ========== 步骤4: 提取各模态的句子级表示 ==========
        x_v_embed = h_tv[0]  # 视觉增强的文本表示 [bs, 768]
        x_a_embed = h_ta[0]  # 音频增强的文本表示 [bs, 768]

        # ========== 步骤5: 多模态特征融合 ==========
        # 拼接三个模态的句子级表示
        x = torch.cat([x_t_embed, x_v_embed, x_a_embed], dim=-1)  # [bs, 2304]

        # 通过融合分类器得到最终的情感预测
        pred = self.TVA_decoder(x).view(-1)  # [bs, 1] → [bs]

        # ========== 步骤6: 损失计算（仅训练模式） ==========
        loss_v = loss_a = 0
        loss_nce = 0

        if mode == 'train':
            # 使用冻结的编码器获得教师信号
            x_v_embed_froze = self.vision_encoder_froze(vision).squeeze()  # [bs, 768]
            x_a_embed_froze = self.audio_encoder_froze(audio).squeeze()    # [bs, 768]

            # 知识蒸馏损失：学生模型向教师模型学习
            loss_v = self.get_KL_loss(x_v_embed, x_v_embed_froze)  # 视觉蒸馏损失
            loss_a = self.get_KL_loss(x_a_embed, x_a_embed_froze)  # 音频蒸馏损失

            # 对比学习损失：增强模态间的语义对齐
            loss_nce = (self.get_InfoNCE_loss(x_v_embed, x_t_embed) +
                       self.get_InfoNCE_loss(x_a_embed, x_t_embed))

            return pred, (loss_v, loss_a, loss_nce)
        else:
            # 推理模式，只返回预测结果
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

    def get_distill_loss(self, input1, input2):
        """
        计算蒸馏损失（MSE版本）

        Args:
            input1: 学生模型的输出特征
            input2: 教师模型的输出特征

        Returns:
            diff_loss: 均方误差损失

        注意：当前代码中未使用此函数，实际使用的是KL散度版本
        """
        diff_loss = torch.mean((input1 - input2) * (input1 - input2))
        return diff_loss

    def get_KL_loss(self, x_embed, x_embed_target):
        """
        计算知识蒸馏的KL散度损失

        Args:
            x_embed: 学生模型的特征表示 [batch_size, feature_dim]
            x_embed_target: 教师模型的特征表示 [batch_size, feature_dim]

        Returns:
            loss: KL散度损失

        原理：
        - 将特征转换为概率分布
        - 学生模型学习模仿教师模型的分布
        - 保持预训练模型的知识
        """
        # 学生模型：log_softmax（用于KL散度的第一个参数）
        x_embed1 = F.log_softmax(x_embed, dim=1)

        # 教师模型：softmax（用于KL散度的第二个参数）
        x_embed_target1 = F.softmax(x_embed_target, dim=1)

        # 计算KL散度：KL(teacher || student)
        loss = self.criterion(x_embed1, x_embed_target1)
        return loss

    def get_InfoNCE_loss(self, input1, input2):
        """
        计算InfoNCE对比学习损失

        Args:
            input1: 第一个模态的特征表示 [batch_size, feature_dim]
            input2: 第二个模态的特征表示 [batch_size, feature_dim]

        Returns:
            nce_loss: InfoNCE损失

        原理：
        - 正样本：同一样本的不同模态特征应该相似
        - 负样本：不同样本的特征应该不相似
        - 通过对比学习增强模态间的语义对齐

        计算过程：
        1. L2归一化：将特征向量归一化到单位球面
        2. 正样本相似度：同一样本的模态间相似度
        3. 负样本相似度：与batch内其他样本的相似度
        4. InfoNCE损失：最大化正样本相似度，最小化负样本相似度
        """
        # 步骤1: L2归一化 - 将特征向量投影到单位球面
        x1 = input1 / input1.norm(dim=1, keepdim=True)  # [bs, dim]
        x2 = input2 / input2.norm(dim=1, keepdim=True)  # [bs, dim]

        # 步骤2: 计算正样本相似度 - 同一样本的不同模态
        pos = torch.sum(x1 * x2, dim=-1)  # [bs] 逐元素相乘后求和

        # 步骤3: 计算负样本相似度 - 与batch内所有样本的相似度
        # x1 @ x2.T: [bs, dim] @ [dim, bs] = [bs, bs]
        # logsumexp: 数值稳定的log(sum(exp(x)))计算
        neg = torch.logsumexp(torch.matmul(x1, x2.t()), dim=-1)  # [bs]

        # 步骤4: InfoNCE损失 - 最大化 pos-neg
        # 等价于最大化正样本概率：exp(pos) / sum(exp(all_similarities))
        nce_loss = -(pos - neg).mean()

        return nce_loss
    
    
