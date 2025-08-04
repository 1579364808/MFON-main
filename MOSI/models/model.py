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
        
        self.text_dropout = config.MOSI.downStream.text_drop_out
        
        encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        audio_text_nhead = config.MOSI.downStream.audio_text_nhead
        audio_text_tf_num_layers = config.MOSI.downStream.audio_text_tf_num_layers
        
        self.audio_fea_dim = config.MOSI.downStream.audio_fea_dim
        self.a_len = config.MOSI.downStream.audio_seq_len
        
        vision_text_nhead = config.MOSI.downStream.vision_text_nhead
        vision_text_tf_num_layers = config.MOSI.downStream.vision_text_tf_num_layers
        
        
        attn_dropout= config.MOSI.downStream.drop_out
        attn_mask = config.MOSI.downStream.attn_mask
        
        audio_fea_dim = config.MOSI.downStream.audio_fea_dim
        vision_fea_dim = config.MOSI.downStream.vision_fea_dim
        text_fea_dim = config.MOSI.downStream.text_fea_dim
        
        self.vlen, self.alen = config.MOSI.downStream.vlen, config.MOSI.downStream.alen

        # ========== 可学习向量配置 ==========
        self.use_learnable_vectors = config.MOSI.downStream.use_learnable_vectors

        # 根据配置决定是否初始化可学习向量
        if self.use_learnable_vectors:
            self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))  # 音频可学习向量 [375, 768]
            self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))  # 视觉可学习向量 [500, 768]
            print(f"✅ 使用可学习向量: 音频向量 {self.prompta_m.shape}, 视觉向量 {self.promptv_m.shape}")
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
        ) # Q:text, KV:vision
         
        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=audio_text_nhead, layers=audio_text_tf_num_layers, 
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        ) # Q:text, KV:audio
        
        # 删除冻结编码器 - 不再使用知识蒸馏
        
        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],
            output_size=1
        )
        self.device = config.DEVICE
        self.model_path = config.MOSI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)

    # 删除load_froze函数 - 不再需要加载冻结编码器
       
    def forward(self, text, vision, audio, mode='train'):
        last_hidden_text  = self.text_encoder(text)   # [bs, seq, h] [bs, h]
        last_hidden_text = F.dropout(self.proj_t(last_hidden_text.permute(1, 0, 2)), 
            p=self.text_dropout, training=self.training
        )
        x_t_embed = last_hidden_text[0]
        
        # ========== 视觉特征处理 ==========
        proj_vision = self.proj_v(vision).permute(1, 0, 2)  # [seq_v, bs, 768]
        if self.use_learnable_vectors and self.promptv_m is not None:
            proj_vision = proj_vision + self.promptv_m.unsqueeze(1)  # 添加可学习视觉向量
        h_tv = self.vision_with_text(last_hidden_text, proj_vision, proj_vision)    # [seq-v, bs, 768] [seq-t, bs, 768]--> [seq, bs,h]
        x_v_embed = h_tv[0]

        # ========== 音频特征处理 ==========
        proj_audio = self.proj_a(audio).permute(1, 0, 2)  # [seq_a, bs, 768]
        if self.use_learnable_vectors and self.prompta_m is not None:
            proj_audio = proj_audio + self.prompta_m.unsqueeze(1)  # 添加可学习音频向量
        h_ta = self.audio_with_text(last_hidden_text, proj_audio, proj_audio)
        x_a_embed = h_ta[0]
        
        x = torch.cat([x_t_embed, x_v_embed, x_a_embed], dim=-1) # [bs, 3h]
        pred = self.TVA_decoder(x).view(-1) # [bs]

        # 删除知识蒸馏和对比学习 - 只返回预测结果
        if mode == 'train':
            return pred, None  # 不再返回额外损失
        else:
            return pred, (x_t_embed, x_v_embed, x_a_embed)
        
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
        

    # 删除所有损失函数 - 不再使用知识蒸馏和对比学习
    
    
