
        
import torch
from torch import nn
import torch.nn.functional as F
import os

from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()

        self.encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        self.audio_fea_dim = config.SIMS.downStream.audio_fea_dim
        self.audio_seq_len = config.SIMS.downStream.audio_seq_len
        self.audio_nhead = config.SIMS.downStream.audio_nhead
        self.audio_tf_num_layers = config.SIMS.downStream.audio_tf_num_layers
        self.attn_dropout = config.SIMS.downStream.audio_drop_out
        self.attn_mask = config.SIMS.downStream.audio_attn_mask
        
        self.proj_a = nn.Linear(self.audio_fea_dim, self.encoder_fea_dim)
        
        self.trans_encoder_a = TransformerEncoder(embed_dim=self.encoder_fea_dim, 
                                num_heads=self.audio_nhead, # 8
                                layers=self.audio_tf_num_layers, # 2
                                attn_dropout=self.attn_dropout, 
                                relu_dropout=self.attn_dropout, 
                                res_dropout=self.attn_dropout, 
                                embed_dropout=self.attn_dropout,
                                attn_mask=self.attn_mask) # True
            
    def forward(self,audio):
        
        proj_x_a = self.proj_a(audio) # [bs, seq, h]
        proj_x_a = proj_x_a.permute(1, 0, 2)
        
        h_as = self.trans_encoder_a(proj_x_a) 
        if type(h_as) == tuple:
            h_as = h_as[0]# [seq, bs,h]
        last_h_a = h_as[0]   # Take the last output for prediction # [bs, h]
        return last_h_a
        
    def set_froze(self):
        for param in self.parameters():
            param.requires_grad = False


# 删除AudioPretrain类 - 不再需要单模态预训练
