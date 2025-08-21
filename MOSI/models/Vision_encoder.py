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
    def __init__(self, config):
        super(VisionEncoder, self).__init__()
        self.vision_fea_dim = config.MOSI.downStream.vision_fea_dim
        self.vision_seq_len = config.MOSI.downStream.vision_seq_len
        
        self.encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        self.vision_nhead = config.MOSI.downStream.vision_nhead
        self.vision_tf_num_layers = config.MOSI.downStream.vision_tf_num_layers
        self.attn_dropout = config.MOSI.downStream.vision_drop_out
        self.attn_mask = config.MOSI.downStream.vision_attn_mask
        
        self.proj_a = nn.Linear(self.vision_fea_dim, self.encoder_fea_dim)
        
        self.trans_encoder_a = TransformerEncoder(
                embed_dim=self.encoder_fea_dim, 
                num_heads=self.vision_nhead, # 8
                layers=self.vision_tf_num_layers, # 2
                attn_dropout=self.attn_dropout, 
                relu_dropout=self.attn_dropout, 
                res_dropout=self.attn_dropout, 
                embed_dropout=self.attn_dropout,
                attn_mask=self.attn_mask # True
        ) 
            
    def forward(self, audio):
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
