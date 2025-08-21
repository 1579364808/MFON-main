import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
import math

from models.classifier import BaseClassifier
from models.Text_encoder import TextEncoder
from models.Vision_encoder import VisionEncoder
from models.Audio_encoder import AudioEncoder
from models.trans.transformer import TransformerEncoder
from models.atcaf_plugin import init_atcaf_plugin, get_atcaf_plugin




def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))



class TVA_fusion(nn.Module):
    def __init__(self, config):
        super(TVA_fusion, self).__init__()
        self.config = config
        
        self.text_dropout = config.SIMS.downStream.text_drop_out
        
        encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        audio_text_nhead = config.SIMS.downStream.audio_text_nhead
        audio_text_tf_num_layers = config.SIMS.downStream.audio_text_tf_num_layers
        
        self.audio_fea_dim = config.SIMS.downStream.audio_fea_dim
        self.a_len = config.SIMS.downStream.audio_seq_len
        
        vision_text_nhead = config.SIMS.downStream.vision_text_nhead
        vision_text_tf_num_layers = config.SIMS.downStream.vision_text_tf_num_layers
        
        
        attn_dropout= config.SIMS.downStream.drop_out
        attn_mask = config.SIMS.downStream.attn_mask
        
        audio_fea_dim = config.SIMS.downStream.audio_fea_dim
        vision_fea_dim = config.SIMS.downStream.vision_fea_dim
        text_fea_dim = config.SIMS.downStream.text_fea_dim
        
        self.vlen, self.alen = config.SIMS.downStream.vlen, config.SIMS.downStream.alen

        # ========== 可学习向量配置 ==========
        self.use_learnable_vectors = config.SIMS.downStream.use_learnable_vectors

        # 根据配置决定是否初始化可学习向量
        if self.use_learnable_vectors:
            # 支持独立配置可学习向量长度
            learnable_vision_len = getattr(config.SIMS.downStream, 'learnable_vision_len', self.vlen)
            learnable_audio_len = getattr(config.SIMS.downStream, 'learnable_audio_len', self.alen)

            self.prompta_m = nn.Parameter(torch.rand(learnable_audio_len, encoder_fea_dim))
            self.promptv_m = nn.Parameter(torch.rand(learnable_vision_len, encoder_fea_dim))

            # 计算参数量
            vision_params = learnable_vision_len * encoder_fea_dim
            audio_params = learnable_audio_len * encoder_fea_dim
            total_params = vision_params + audio_params

            print(f"✅ 使用可学习向量:")
            print(f"   音频向量: {self.prompta_m.shape} = {audio_params:,} 参数")
            print(f"   视觉向量: {self.promptv_m.shape} = {vision_params:,} 参数")
            print(f"   总计增加: {total_params:,} 参数")
            print(f"   原始序列长度: 音频={self.alen}, 视觉={self.vlen}")
        else:
            self.prompta_m = None
            self.promptv_m = None
            print("❌ 不使用可学习向量")
        
        self.text_encoder = TextEncoder(config=config)
        self.proj_t = nn.Linear(text_fea_dim, encoder_fea_dim)
         
        self.proj_v = nn.Linear(vision_fea_dim, encoder_fea_dim)
        self.vision_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=vision_text_nhead, layers=vision_text_tf_num_layers,
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        )  # Q:text, KV:vision

        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=audio_text_nhead, layers=audio_text_tf_num_layers,
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        )  # Q:text, KV:audio

        # ========== 双向交叉注意力配置 ==========
        self.use_bidirectional_attention = getattr(config.SIMS.downStream, 'use_bidirectional_attention', False)
        self.bidirectional_fusion_mode = getattr(config.SIMS.downStream, 'bidirectional_fusion_mode', 'concat')
        self.bidirectional_fusion_weight = getattr(config.SIMS.downStream, 'bidirectional_fusion_weight', 0.5)

        if self.use_bidirectional_attention:
            # 反向注意力：vision/audio -> text
            self.text_with_vision = TransformerEncoder(
                embed_dim=encoder_fea_dim, num_heads=vision_text_nhead, layers=vision_text_tf_num_layers,
                attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
                attn_mask=attn_mask
            )  # Q:vision, KV:text

            self.text_with_audio = TransformerEncoder(
                embed_dim=encoder_fea_dim, num_heads=audio_text_nhead, layers=audio_text_tf_num_layers,
                attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
                attn_mask=attn_mask
            )  # Q:audio, KV:text

            # 门控融合层（仅在 gated 模式使用）
            if self.bidirectional_fusion_mode == 'gated':
                self.vision_gate = nn.Sequential(
                    nn.Linear(encoder_fea_dim * 2, encoder_fea_dim),
                    nn.Sigmoid()
                )
                self.audio_gate = nn.Sequential(
                    nn.Linear(encoder_fea_dim * 2, encoder_fea_dim),
                    nn.Sigmoid()
                )

            print(f"✅ 启用双向交叉注意力，融合模式: {self.bidirectional_fusion_mode}")
            if self.bidirectional_fusion_mode == 'weighted_sum':
                print(f"   融合权重: {self.bidirectional_fusion_weight}")
        else:
            self.text_with_vision = None
            self.text_with_audio = None
            print("❌ 使用单向交叉注意力")
        
        # 删除冻结编码器 - 不再使用知识蒸馏

        # 根据双向注意力模式调整解码器输入维度
        decoder_input_size = encoder_fea_dim * 3  # text + vision + audio
        if self.use_bidirectional_attention and self.bidirectional_fusion_mode == 'concat':
            decoder_input_size = encoder_fea_dim * 5  # text + vision_forward + audio_forward + vision_backward + audio_backward

        self.TVA_decoder = BaseClassifier(
            input_size=decoder_input_size,
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],
            output_size=1
        )

        # 辅助分类头（可开关）
        self.enable_aux_cls = getattr(config.SIMS.downStream, 'enable_aux_cls', False)
        self.cls_head = nn.Linear(decoder_input_size, 1) if self.enable_aux_cls else None

        self.device = config.DEVICE
        self.model_path = config.SIMS.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)

        # ========== 初始化 AtCAF 插件（可关闭） ==========
        self.atcaf_plugin = init_atcaf_plugin(config)



        # ========== 全局字典（CS-ATT）移除 ==========
        self.use_plug_causal_attn = False
        self.use_confounder_dict = getattr(config.SIMS.downStream, 'use_confounder_dict', True)  # 该项仅供AtCAF使用
        # 移除 GlobalDictionary 相关引用

    # 删除load_froze函数 - 不再需要加载冻结编码器
       
    def forward(self, text, vision, audio, mode='train', labels=None):
        last_hidden_text  = self.text_encoder(text)   # [bs, seq, h] [bs, h]
        last_hidden_text = F.dropout(self.proj_t(last_hidden_text.permute(1, 0, 2)),
            p=self.text_dropout, training=self.training
        )
        x_t_embed_orig = last_hidden_text[0]  # [batch, D]

        # 已移除全局字典（CS-ATT）相关逻辑
        dict_v = dict_a = None
        
        # ========== 特征处理与因果干预函数 ==========
        def process_features(input_text, input_vision, input_audio, is_counterfactual):
            # 视觉特征处理
            proj_vision = self.proj_v(input_vision).permute(1, 0, 2)
            if self.use_learnable_vectors and self.promptv_m is not None:
                proj_vision = proj_vision + self.promptv_m.unsqueeze(1)

            # 音频特征处理
            proj_audio = self.proj_a(input_audio).permute(1, 0, 2)
            if self.use_learnable_vectors and self.prompta_m is not None:
                proj_audio = proj_audio + self.prompta_m.unsqueeze(1)

            # --- AtCAF 路径的特征收集/干预（若开启）---
            if self.atcaf_plugin.enabled and mode == 'train':
                self.atcaf_plugin.collect_features_for_kmeans(proj_vision, proj_audio)
                if is_counterfactual and self.atcaf_plugin.vision_dict.initialized:
                    vision_confounder = self.atcaf_plugin.vision_dict.get_confounder_features(proj_vision)
                    audio_confounder = self.atcaf_plugin.audio_dict.get_confounder_features(proj_audio)
                    proj_vision = proj_vision - vision_confounder
                    proj_audio = proj_audio - audio_confounder



            # --- 跨模态交互（支持双向注意力） ---
            # 单向注意力：text -> vision/audio
            h_tv = self.vision_with_text(input_text, proj_vision, proj_vision)
            x_v_embed = h_tv[0]

            h_ta = self.audio_with_text(input_text, proj_audio, proj_audio)
            x_a_embed = h_ta[0]

            # 双向注意力：vision/audio -> text（如果启用）
            if self.use_bidirectional_attention and self.text_with_vision is not None:
                h_vt = self.text_with_vision(proj_vision, input_text, input_text)
                x_v_to_t_embed = h_vt[0]

                h_at = self.text_with_audio(proj_audio, input_text, input_text)
                x_a_to_t_embed = h_at[0]

                # 根据融合模式处理双向特征
                if self.bidirectional_fusion_mode == 'concat':
                    # 拼接模式：保留所有特征，让解码器学习组合
                    pass  # x_v_embed, x_a_embed, x_v_to_t_embed, x_a_to_t_embed 都会在后续拼接
                elif self.bidirectional_fusion_mode == 'weighted_sum':
                    # 加权求和模式：固定权重融合
                    alpha = self.bidirectional_fusion_weight
                    x_v_embed = alpha * x_v_embed + (1 - alpha) * x_v_to_t_embed
                    x_a_embed = alpha * x_a_embed + (1 - alpha) * x_a_to_t_embed
                elif self.bidirectional_fusion_mode == 'gated':
                    # 门控融合模式：动态学习融合权重
                    v_concat = torch.cat([x_v_embed, x_v_to_t_embed], dim=-1)
                    v_gate = self.vision_gate(v_concat)
                    x_v_embed = v_gate * x_v_embed + (1 - v_gate) * x_v_to_t_embed

                    a_concat = torch.cat([x_a_embed, x_a_to_t_embed], dim=-1)
                    a_gate = self.audio_gate(a_concat)
                    x_a_embed = a_gate * x_a_embed + (1 - a_gate) * x_a_to_t_embed

            # 返回：跨模态句向量
            # 对于 concat 模式，需要返回额外的反向特征
            if self.use_bidirectional_attention and self.bidirectional_fusion_mode == 'concat':
                # 确保反向特征存在
                if 'x_v_to_t_embed' in locals() and 'x_a_to_t_embed' in locals():
                    return x_v_embed, x_a_embed, x_v_to_t_embed, x_a_to_t_embed
                else:
                    # 如果没有反向特征（如反事实分支），返回零特征占位
                    zero_feat = torch.zeros_like(x_v_embed)
                    return x_v_embed, x_a_embed, zero_feat, zero_feat
            else:
                return x_v_embed, x_a_embed

        # ========== 因果推理处理 ==========
        atcaf_loss = torch.tensor(0.0, device=last_hidden_text.device, dtype=last_hidden_text.dtype)

        if self.atcaf_plugin.enabled:
            self.atcaf_plugin.set_training_mode(mode == 'train')

            # 1. 计算反事实（去混淆）特征
            cf_results = process_features(last_hidden_text, vision, audio, is_counterfactual=True)
            x_v_embed_cf, x_a_embed_cf = cf_results[:2]

            # 2. 计算事实（原始）特征
            real_results = process_features(last_hidden_text, vision, audio, is_counterfactual=False)
            x_v_embed_real, x_a_embed_real = real_results[:2]

            # 处理 concat 模式的额外返回值（在因果干预中使用）

            # 3. 应用因果干预方法
            if self.use_bidirectional_attention and self.bidirectional_fusion_mode == 'concat':
                # concat 模式：需要包含反向特征
                x_v_to_t_real, x_a_to_t_real = real_results[2:4]
                x_v_to_t_cf, x_a_to_t_cf = cf_results[2:4]

                factual_features = torch.cat([x_t_embed_orig, x_v_embed_real, x_a_embed_real, x_v_to_t_real, x_a_to_t_real], dim=-1)
                counterfactual_features = torch.cat([x_t_embed_orig, x_v_embed_cf, x_a_embed_cf, x_v_to_t_cf, x_a_to_t_cf], dim=-1)
            else:
                # weighted_sum 或 gated 模式：特征已融合
                factual_features = torch.cat([x_t_embed_orig, x_v_embed_real, x_a_embed_real], dim=-1)
                counterfactual_features = torch.cat([x_t_embed_orig, x_v_embed_cf, x_a_embed_cf], dim=-1)

            final_features, _ = self.atcaf_plugin.apply_causal_intervention(
                factual_features, counterfactual_features, mode)

            # 4. 计算预测（完全保守：主输出使用事实特征）
            if getattr(self.config.SIMS.downStream, 'atcaf_conservative', False):
                pred = self.TVA_decoder(factual_features).view(-1)
            else:
                pred = self.TVA_decoder(final_features).view(-1)

            # 训练且启用时，计算分类logits（AtCAF分支）
            cls_logits = None
            if mode == 'train' and self.enable_aux_cls and self.cls_head is not None:
                cls_logits = self.cls_head(factual_features).view(-1)

            # 5. 计算因果损失（仅训练时）
            if mode == 'train':
                pred_factual = self.TVA_decoder(factual_features).view(-1)
                pred_counterfactual = self.TVA_decoder(counterfactual_features).view(-1)
                atcaf_loss = self.atcaf_plugin.compute_causal_loss(pred_factual, pred_counterfactual, labels)
        else:
            # 不使用AtCAF时的标准处理
            results = process_features(last_hidden_text, vision, audio, is_counterfactual=False)
            x_v_embed, x_a_embed = results[:2]

            if self.use_bidirectional_attention and self.bidirectional_fusion_mode == 'concat':
                x_v_to_t, x_a_to_t = results[2:4]
                final_features = torch.cat([x_t_embed_orig, x_v_embed, x_a_embed, x_v_to_t, x_a_to_t], dim=-1)
            else:
                final_features = torch.cat([x_t_embed_orig, x_v_embed, x_a_embed], dim=-1)
            pred = self.TVA_decoder(final_features).view(-1)

            # 训练且启用时，计算分类logits
            cls_logits = None
            if mode == 'train' and self.enable_aux_cls and self.cls_head is not None:
                cls_logits = self.cls_head(final_features).view(-1)

        # ========== 返回结果 ==========
        if mode == 'train':
            # 训练时额外返回三模态句向量用于对比学习（均为事实特征）
            if self.atcaf_plugin.enabled:
                train_tuple = (x_t_embed_orig, x_v_embed_real, x_a_embed_real)
            else:
                train_tuple = (x_t_embed_orig, x_v_embed, x_a_embed)

            if self.enable_aux_cls and self.cls_head is not None:
                return pred, atcaf_loss, train_tuple, cls_logits
            else:
                return pred, atcaf_loss, train_tuple
        else:
            # 推理时返回特征用于分析
            if self.atcaf_plugin.enabled:
                cf_results = process_features(last_hidden_text, vision, audio, is_counterfactual=True)
                x_v_embed_cf, x_a_embed_cf = cf_results[0], cf_results[1]
                return pred, (x_t_embed_orig, x_v_embed_cf, x_a_embed_cf)
            else:
                results = process_features(last_hidden_text, vision, audio, is_counterfactual=False)
                x_v_embed, x_a_embed = results[0], results[1]
                return pred, (x_t_embed_orig, x_v_embed, x_a_embed)

    def save_model(self,name=None):
        # save all modules
        if name==None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = self.model_path + str(name)+'TVA_fusion' + '_model.pt'
        print('model saved at:\n', mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name=None):
        if name==None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = name
        print('model loaded from:\n', mode_path)
        # self.load_state_dict(torch.load(mode_path, map_location=self.device))
        checkpoint = torch.load(mode_path, map_location=self.device)
        model_state_dict = self.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        self.load_state_dict(filtered_checkpoint, strict=False)
    
    
