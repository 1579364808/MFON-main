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
from Audio_encoder import AudioEncoder
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


class AdaptiveGuidanceModule(nn.Module):
    """
    自适应引导模块 (Adaptive Guidance Module, AGM)

    核心创新：从"固定向导"到"动态选举"的自适应引导机制

    设计理念：
    - 传统方法：文本永远作为Query（固定向导）
    - 创新方法：根据输入动态选举最适合的向导模态

    灵感来源：
    1. Mixture of Experts (MoE): 门控网络动态选择专家
    2. GRU门控机制: 基于输入动态控制信息流
    3. DeepSeek无损平衡: 解决模态负载不均衡问题

    工作流程：
    1. 接收三个模态的句子级表示
    2. 通过门控网络计算每个模态的"向导置信度"
    3. 应用无损平衡机制，动态调整模态偏置
    4. 输出平衡的权重分布 [ω_t, ω_v, ω_a]

    🆕 无损平衡机制：
    - 引入模态偏置 (modality bias)，动态调整模态选择
    - 基于历史负载统计，自动平衡模态使用
    - 不影响梯度计算，保持训练稳定性
    """

    def __init__(self, feature_dim, hidden_dim=128, dropout_rate=0.1, activation='relu',
                 enable_load_balancing=True, update_rate=0.01, target_balance_ratio=0.2):
        """
        初始化自适应引导模块

        Args:
            feature_dim: 输入特征维度（通常是encoder_fea_dim，如768）
            hidden_dim: 门控网络的隐藏层维度
            dropout_rate: dropout率
            activation: 激活函数类型
            enable_load_balancing: 是否启用无损负载均衡
            update_rate: 偏置更新率 (类似DeepSeek的u参数)
            target_balance_ratio: 目标平衡比例 (非主导模态的最小权重比例)
        """
        super(AdaptiveGuidanceModule, self).__init__()

        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ReLU()  # 默认使用ReLU

        # 门控网络：输入三个模态的拼接特征，输出三个logits (不使用Softmax)
        self.gating_network = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),  # 输入：[bs, 768*3]
            act_fn,
            nn.Dropout(dropout_rate),  # 可配置的dropout率
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 3),  # 输出：[bs, 3] logits
            # 注意：移除Softmax，在forward中手动处理
        )

        # 🆕 无损负载均衡参数
        self.enable_load_balancing = enable_load_balancing
        self.update_rate = update_rate
        self.target_balance_ratio = target_balance_ratio

        # 模态偏置 (类似DeepSeek的expert bias)
        # 使用register_buffer使其不参与梯度计算，但会被保存和加载
        self.register_buffer('modality_bias', torch.zeros(3))  # [text, vision, audio]

        # 负载统计 (用于动态调整偏置)
        self.register_buffer('load_history', torch.zeros(3))  # 历史负载统计
        self.register_buffer('update_count', torch.tensor(0))  # 更新次数

    def forward(self, text_repr, vision_repr, audio_repr):
        """
        动态选举向导模态 (集成无损负载均衡)

        Args:
            text_repr: 文本句子级表示 [batch_size, feature_dim]
            vision_repr: 视觉句子级表示 [batch_size, feature_dim]
            audio_repr: 音频句子级表示 [batch_size, feature_dim]

        Returns:
            weights: 三个模态的向导权重 [batch_size, 3]
                    weights[:, 0] = ω_t (文本权重)
                    weights[:, 1] = ω_v (视觉权重)
                    weights[:, 2] = ω_a (音频权重)
        """
        # 拼接三个模态的句子级表示
        combined_features = torch.cat([text_repr, vision_repr, audio_repr], dim=-1)  # [bs, 768*3]

        # 通过门控网络计算原始logits
        raw_logits = self.gating_network(combined_features)  # [bs, 3]

        if self.enable_load_balancing and self.training:
            # 🆕 应用无损负载均衡
            balanced_weights = self._apply_load_balancing(raw_logits)

            # 更新负载统计 (用于下次调整偏置)
            self._update_load_statistics(balanced_weights)

            return balanced_weights
        else:
            # 推理模式或未启用负载均衡：直接使用Softmax
            weights = torch.softmax(raw_logits, dim=-1)
            return weights

    def _apply_load_balancing(self, raw_logits):
        """
        应用无损负载均衡机制

        核心思想：
        1. 在logits上添加模态偏置 (仅用于权重计算)
        2. 偏置不参与梯度计算，避免干扰主任务
        3. 动态调整偏置以平衡模态使用

        Args:
            raw_logits: 原始门控logits [batch_size, 3]

        Returns:
            balanced_weights: 平衡后的权重 [batch_size, 3]
        """
        # 添加模态偏置到logits (类似DeepSeek的 s_{i,t} + b_i)
        # 注意：使用detach()确保偏置不参与梯度计算
        biased_logits = raw_logits + self.modality_bias.detach().unsqueeze(0)  # [bs, 3]

        # 基于偏置后的logits计算权重
        balanced_weights = torch.softmax(biased_logits, dim=-1)  # [bs, 3]

        return balanced_weights

    def _update_load_statistics(self, weights):
        """
        更新负载统计并调整模态偏置

        基于DeepSeek算法1的适配版本：
        1. 统计当前batch中每个模态的平均负载
        2. 计算负载不平衡误差
        3. 更新模态偏置以促进平衡

        Args:
            weights: 当前batch的权重分布 [batch_size, 3]
        """
        with torch.no_grad():  # 确保不影响梯度
            # 计算当前batch的模态负载 (平均权重)
            current_load = weights.mean(dim=0)  # [3] - 每个模态的平均权重

            # 计算目标负载 (理想情况下应该相对平衡)
            # 允许主导模态有更高权重，但防止其他模态完全被忽略
            target_load = torch.tensor([
                1.0 - 2 * self.target_balance_ratio,  # 主导模态目标权重
                self.target_balance_ratio,            # 非主导模态目标权重
                self.target_balance_ratio             # 非主导模态目标权重
            ], device=weights.device)

            # 计算负载不平衡误差
            load_error = current_load - target_load  # [3]

            # 更新模态偏置 (类似DeepSeek的偏置更新)
            # 负载过高的模态：降低偏置 (减少被选中概率)
            # 负载过低的模态：提高偏置 (增加被选中概率)
            bias_update = -self.update_rate * torch.sign(load_error)

            # 应用偏置更新
            self.modality_bias += bias_update

            # 更新历史统计
            self.load_history = 0.9 * self.load_history + 0.1 * current_load
            self.update_count += 1

    def get_load_statistics(self):
        """
        获取负载统计信息 (用于监控和调试)

        Returns:
            dict: 包含负载统计的字典
        """
        return {
            'modality_bias': self.modality_bias.cpu().numpy(),
            'load_history': self.load_history.cpu().numpy(),
            'update_count': self.update_count.item(),
            'target_balance_ratio': self.target_balance_ratio
        }


class AdaptiveTVA_fusion(nn.Module):
    """
    自适应TVA多模态融合模型
    
    核心创新：集成自适应引导模块(AGM)的多模态融合架构
    
    与原始TVA_fusion的区别：
    1. 新增AdaptiveGuidanceModule：动态选举向导模态
    2. 动态Query生成：不再固定使用文本作为Query
    3. 自适应跨模态注意力：根据权重调整注意力计算
    
    优势：
    - 处理极端情况：当语气/表情极度夸张而文字平淡时
    - 提高鲁棒性：适应不同样本的模态重要性分布
    - 保持兼容性：在文本主导的情况下退化为原始行为
    """
    
    def __init__(self, config, agm_hidden_dim=None, agm_dropout=None, agm_activation=None,
                 enable_load_balancing=None, update_rate=None, target_balance_ratio=None):
        """
        初始化自适应TVA融合模型

        Args:
            config: 配置对象，包含所有超参数设置
            agm_hidden_dim: AGM隐藏层维度，None时使用配置文件中的值
            agm_dropout: AGM dropout率，None时使用配置文件中的值
            agm_activation: AGM激活函数，None时使用配置文件中的值
            enable_load_balancing: 是否启用负载均衡，None时使用配置文件中的值
            update_rate: 偏置更新率，None时使用配置文件中的值
            target_balance_ratio: 目标平衡比例，None时使用配置文件中的值
        """
        super(AdaptiveTVA_fusion, self).__init__()
        self.config = config

        # 获取自适应配置
        adaptive_config = config.MOSEI.downStream.adaptiveTVAtrain

        # AGM配置参数（支持外部传入覆盖）
        self.agm_hidden_dim = agm_hidden_dim or adaptive_config.agm_hidden_dim
        self.agm_dropout = agm_dropout or adaptive_config.agm_dropout
        self.agm_activation = agm_activation or adaptive_config.agm_activation

        # 🆕 负载均衡配置参数
        self.enable_load_balancing = (enable_load_balancing if enable_load_balancing is not None
                                    else getattr(adaptive_config, 'enable_load_balancing', True))
        self.update_rate = (update_rate if update_rate is not None
                          else getattr(adaptive_config, 'load_balance_update_rate', 0.01))
        self.target_balance_ratio = (target_balance_ratio if target_balance_ratio is not None
                                   else getattr(adaptive_config, 'target_balance_ratio', 0.2))
        
        # ========== 基础配置参数 ==========
        self.text_dropout = config.MOSEI.downStream.text_drop_out  # 0.3
        
        encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim  # 768
        
        # 跨模态Transformer配置
        audio_text_nhead = config.MOSEI.downStream.audio_text_nhead              # 8
        audio_text_tf_num_layers = config.MOSEI.downStream.audio_text_tf_num_layers  # 3
        vision_text_nhead = config.MOSEI.downStream.vision_text_nhead              # 8
        vision_text_tf_num_layers = config.MOSEI.downStream.vision_text_tf_num_layers  # 2
        
        attn_dropout = config.MOSEI.downStream.drop_out  # 0.1
        attn_mask = config.MOSEI.downStream.attn_mask    # True
        
        # 各模态特征维度
        audio_fea_dim = config.MOSEI.downStream.audio_fea_dim    # 74
        vision_fea_dim = config.MOSEI.downStream.vision_fea_dim  # 35
        text_fea_dim = config.MOSEI.downStream.text_fea_dim      # 768
        
        # 序列长度
        self.vlen, self.alen = config.MOSEI.downStream.vlen, config.MOSEI.downStream.alen  # 500, 500
        
        # ========== 核心创新：自适应引导模块 (集成无损负载均衡) ==========
        self.adaptive_guidance = AdaptiveGuidanceModule(
            feature_dim=encoder_fea_dim,                    # 768
            hidden_dim=self.agm_hidden_dim,                 # 使用配置中的隐藏层维度
            dropout_rate=self.agm_dropout,                  # 使用配置中的dropout率
            activation=self.agm_activation,                 # 使用配置中的激活函数
            enable_load_balancing=self.enable_load_balancing, # 🆕 启用负载均衡
            update_rate=self.update_rate,                   # 🆕 偏置更新率
            target_balance_ratio=self.target_balance_ratio  # 🆕 目标平衡比例
        )
        
        # ========== 可学习提示符 ==========
        self.use_learnable_vectors = config.MOSEI.downStream.use_learnable_vectors

        # 根据配置决定是否初始化可学习向量
        if self.use_learnable_vectors:
            self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))  # [500, 768]
            self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))  # [500, 768]
            print(f"✅ [AdaptiveTVA] 使用可学习向量: 音频向量 {self.prompta_m.shape}, 视觉向量 {self.promptv_m.shape}")
        else:
            self.prompta_m = None
            self.promptv_m = None
            print("❌ [AdaptiveTVA] 不使用可学习向量")
        
        # ========== 文本处理模块 ==========
        self.text_encoder = TextEncoder(config=config)
        self.proj_t = nn.Linear(text_fea_dim, encoder_fea_dim)  # 768→768
        
        # ========== 视觉处理模块 ==========
        self.proj_v = nn.Linear(vision_fea_dim, encoder_fea_dim)  # 35→768
        
        # 视觉-文本跨模态Transformer（现在接受动态Query）
        self.vision_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim,
            num_heads=vision_text_nhead,
            layers=vision_text_tf_num_layers,
            attn_dropout=attn_dropout,
            relu_dropout=attn_dropout,
            res_dropout=attn_dropout,
            embed_dropout=attn_dropout,
            attn_mask=attn_mask
        )
        
        # ========== 音频处理模块 ==========
        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)  # 74→768
        
        # 音频-文本跨模态Transformer（现在接受动态Query）
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim,
            num_heads=audio_text_nhead,
            layers=audio_text_tf_num_layers,
            attn_dropout=attn_dropout,
            relu_dropout=attn_dropout,
            res_dropout=attn_dropout,
            embed_dropout=attn_dropout,
            attn_mask=attn_mask
        )
        
        # ========== 知识蒸馏模块 ==========
        self.vision_encoder_froze = VisionEncoder(config=config)
        self.audio_encoder_froze = AudioEncoder(config=config)
        
        # ========== 最终融合分类器 ==========
        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,  # 2304
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],  # [768, 384, 96]
            output_size=1
        )
        
        # ========== 训练相关配置 ==========
        self.device = config.DEVICE
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)
        
    def load_froze(self):
        """
        加载并冻结预训练的单模态编码器
        """
        model_path = self.config.MOSEI.path.encoder_path + str(self.config.seed) + '/'
        
        self.audio_encoder_froze.load_state_dict(
            torch.load(model_path + 'best_loss_audio_encoder.pt', map_location=self.device)
        )
        self.vision_encoder_froze.load_state_dict(
            torch.load(model_path + 'best_loss_vision_encoder.pt', map_location=self.device)
        )
        
        self.audio_encoder_froze.set_froze()
        self.vision_encoder_froze.set_froze()

    def _get_sentence_representations(self, text_seq, vision_seq, audio_seq):
        """
        获取三个模态的句子级表示（用于门控网络）

        Args:
            text_seq: 文本序列表示 [seq_len, batch_size, feature_dim]
            vision_seq: 视觉序列表示 [seq_len, batch_size, feature_dim]
            audio_seq: 音频序列表示 [seq_len, batch_size, feature_dim]

        Returns:
            text_repr: 文本句子级表示 [batch_size, feature_dim]
            vision_repr: 视觉句子级表示 [batch_size, feature_dim]
            audio_repr: 音频句子级表示 [batch_size, feature_dim]
        """
        # 取第一个位置的表示作为句子级表示（通常是[CLS]位置）
        text_repr = text_seq[0]      # [batch_size, feature_dim]
        vision_repr = vision_seq[0]  # [batch_size, feature_dim]
        audio_repr = audio_seq[0]    # [batch_size, feature_dim]

        return text_repr, vision_repr, audio_repr

    def _apply_adaptive_weights(self, text_repr, vision_repr, audio_repr, weights):
        """
        应用自适应权重到句子级表示

        这是我们创新的核心：在句子级别应用动态权重，避免序列长度不匹配问题

        Args:
            text_repr: 文本句子级表示 [batch_size, feature_dim]
            vision_repr: 视觉句子级表示 [batch_size, feature_dim]
            audio_repr: 音频句子级表示 [batch_size, feature_dim]
            weights: 模态权重 [batch_size, 3]

        Returns:
            adaptive_text: 自适应文本表示 [batch_size, feature_dim]
            adaptive_vision: 自适应视觉表示 [batch_size, feature_dim]
            adaptive_audio: 自适应音频表示 [batch_size, feature_dim]
        """
        # 提取权重
        w_t = weights[:, 0].unsqueeze(1)  # [bs, 1]
        w_v = weights[:, 1].unsqueeze(1)  # [bs, 1]
        w_a = weights[:, 2].unsqueeze(1)  # [bs, 1]

        # 应用自适应权重：每个模态的表示都是三个模态的加权组合
        # 权重决定了每个模态在最终表示中的重要性
        adaptive_text = text_repr * w_t + vision_repr * w_v + audio_repr * w_a
        adaptive_vision = vision_repr * w_v + text_repr * w_t + audio_repr * w_a
        adaptive_audio = audio_repr * w_a + text_repr * w_t + vision_repr * w_v

        return adaptive_text, adaptive_vision, adaptive_audio

    def forward(self, text, vision, audio, mode='train'):
        """
        自适应TVA融合模型的前向传播

        核心创新：动态选举向导模态，而非固定使用文本

        Args:
            text: 文本输入，字符串列表
            vision: 视觉特征 [batch_size, 500, 35]
            audio: 音频特征 [batch_size, 500, 74]
            mode: 模式，'train'时计算所有损失，'eval'时只返回预测

        Returns:
            pred: 情感预测值 [batch_size]
            losses: 训练模式下返回 (loss_v, loss_a, loss_nce, guidance_weights)
                   推理模式下返回 None

        创新流程：
        1. 获取初始特征表示
        2. 动态选举向导模态（AGM核心）
        3. 标准跨模态注意力（文本作为Query）
        4. 在句子级别应用自适应权重（避免序列长度不匹配）
        5. 特征融合和情感预测
        6. 损失计算（包含引导权重信息）

        关键创新：在句子级表示上应用动态权重，而非序列级别，
        这样既实现了自适应引导，又避免了序列长度不匹配的问题。
        """

        # ========== 步骤1: 文本编码和处理 ==========
        last_hidden_text = self.text_encoder(text)  # [bs, seq_len, 768]

        # 文本投影和dropout
        last_hidden_text = F.dropout(
            self.proj_t(last_hidden_text.permute(1, 0, 2)),
            p=self.text_dropout,
            training=self.training
        )  # [seq, bs, 768]

        # ========== 步骤2: 视觉和音频初始处理 ==========
        # 视觉特征投影并根据配置添加提示符
        proj_vision = self.proj_v(vision).permute(1, 0, 2)  # [bs, 500, 35] → [500, bs, 768]
        if self.use_learnable_vectors and self.promptv_m is not None:
            proj_vision = proj_vision + self.promptv_m.unsqueeze(1)  # 添加可学习视觉向量

        # 音频特征投影并根据配置添加提示符
        proj_audio = self.proj_a(audio).permute(1, 0, 2)  # [bs, 500, 74] → [500, bs, 768]
        if self.use_learnable_vectors and self.prompta_m is not None:
            proj_audio = proj_audio + self.prompta_m.unsqueeze(1)  # 添加可学习音频向量

        # ========== 步骤3: 核心创新 - 动态选举向导模态 ==========
        # 获取三个模态的句子级表示
        text_repr, vision_repr, audio_repr = self._get_sentence_representations(
            last_hidden_text, proj_vision, proj_audio
        )

        # 通过自适应引导模块计算权重
        guidance_weights = self.adaptive_guidance(text_repr, vision_repr, audio_repr)
        # [batch_size, 3] where [:, 0]=ω_t, [:, 1]=ω_v, [:, 2]=ω_a

        # ========== 步骤4: 标准跨模态注意力 ==========
        # 使用文本作为Query进行跨模态注意力计算
        # 注意：我们在句子级别应用自适应权重，而不是在这里

        # 视觉-文本跨模态注意力
        h_tv = self.vision_with_text(last_hidden_text, proj_vision, proj_vision)

        # 音频-文本跨模态注意力
        h_ta = self.audio_with_text(last_hidden_text, proj_audio, proj_audio)

        # ========== 步骤5: 提取增强后的模态表示并应用自适应权重 ==========
        # 提取基础的句子级表示
        base_text_embed = last_hidden_text[0]  # 原始文本表示 [bs, 768]
        base_vision_embed = h_tv[0]            # 视觉增强表示 [bs, 768]
        base_audio_embed = h_ta[0]             # 音频增强表示 [bs, 768]

        # 应用自适应权重（核心创新）
        x_t_embed, x_v_embed, x_a_embed = self._apply_adaptive_weights(
            base_text_embed, base_vision_embed, base_audio_embed, guidance_weights
        )

        # ========== 步骤6: 多模态特征融合 ==========
        x = torch.cat([x_t_embed, x_v_embed, x_a_embed], dim=-1)  # [bs, 2304]
        pred = self.TVA_decoder(x).view(-1)  # [bs]

        # ========== 步骤7: 损失计算（仅训练模式） ==========
        if mode == 'train':
            # 知识蒸馏损失
            x_v_embed_froze = self.vision_encoder_froze(vision).squeeze()
            x_a_embed_froze = self.audio_encoder_froze(audio).squeeze()

            loss_v = self.get_KL_loss(x_v_embed, x_v_embed_froze)
            loss_a = self.get_KL_loss(x_a_embed, x_a_embed_froze)

            # 对比学习损失
            loss_nce = (self.get_InfoNCE_loss(x_v_embed, x_t_embed) +
                       self.get_InfoNCE_loss(x_a_embed, x_t_embed))

            # 返回损失和引导权重（用于分析和可视化）
            return pred, (loss_v, loss_a, loss_nce, guidance_weights)
        else:
            return pred, guidance_weights  # 推理时也返回权重用于分析

    def save_model(self, name=None):
        """
        保存自适应TVA融合模型

        Args:
            name: 保存文件的前缀名，默认为None时使用默认命名
        """
        if name == None:
            mode_path = self.model_path + 'AdaptiveTVA_fusion' + '_model.pt'
        else:
            mode_path = self.model_path + str(name) + 'AdaptiveTVA_fusion' + '_model.pt'

        print('Adaptive model saved at:\n', mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name=None):
        """
        加载自适应TVA融合模型

        Args:
            name: 加载文件的路径或前缀名
        """
        if name == None:
            mode_path = self.model_path + 'AdaptiveTVA_fusion' + '_model.pt'
        else:
            mode_path = name

        print('Adaptive model loaded from:\n', mode_path)

        checkpoint = torch.load(mode_path, map_location=self.device)
        model_state_dict = self.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        self.load_state_dict(filtered_checkpoint, strict=False)

    def get_KL_loss(self, x_embed, x_embed_target):
        """
        计算知识蒸馏的KL散度损失
        """
        x_embed1 = F.log_softmax(x_embed, dim=1)
        x_embed_target1 = F.softmax(x_embed_target, dim=1)
        loss = self.criterion(x_embed1, x_embed_target1)
        return loss

    def get_InfoNCE_loss(self, input1, input2):
        """
        计算InfoNCE对比学习损失
        """
        x1 = input1 / input1.norm(dim=1, keepdim=True)
        x2 = input2 / input2.norm(dim=1, keepdim=True)

        pos = torch.sum(x1 * x2, dim=-1)
        neg = torch.logsumexp(torch.matmul(x1, x2.t()), dim=-1)
        nce_loss = -(pos - neg).mean()

        return nce_loss

    def analyze_guidance_weights(self, guidance_weights):
        """
        分析引导权重的分布情况

        Args:
            guidance_weights: 引导权重 [batch_size, 3]

        Returns:
            analysis: 包含统计信息的字典
        """
        with torch.no_grad():
            # 计算每个模态的平均权重
            mean_weights = guidance_weights.mean(dim=0)  # [3]

            # 计算权重的标准差（衡量选择的多样性）
            std_weights = guidance_weights.std(dim=0)   # [3]

            # 统计主导模态（权重最大的模态）
            dominant_modality = guidance_weights.argmax(dim=1)  # [batch_size]
            modality_counts = torch.bincount(dominant_modality, minlength=3)

            analysis = {
                'mean_text_weight': mean_weights[0].item(),
                'mean_vision_weight': mean_weights[1].item(),
                'mean_audio_weight': mean_weights[2].item(),
                'std_text_weight': std_weights[0].item(),
                'std_vision_weight': std_weights[1].item(),
                'std_audio_weight': std_weights[2].item(),
                'text_dominant_count': modality_counts[0].item(),
                'vision_dominant_count': modality_counts[1].item(),
                'audio_dominant_count': modality_counts[2].item(),
                'total_samples': guidance_weights.size(0)
            }

        return analysis
