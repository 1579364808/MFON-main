import torch
import os

seed = 1111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(__file__)

LOGPATH = os.path.join(root_path, 'log/')

if not os.path.exists(LOGPATH): 
    os.makedirs(LOGPATH)
                

class SIMS:
    class path:
        raw_data_path = '../data/SIMS/unaligned_39.pkl'
        model_path = os.path.join(root_path, 'save_models/all_model/SIMS/')
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        # 删除uni_fea_encoder路径 - 不再需要预训练单模态编码器

    class downStream:
        language = 'cn'
      
        encoder_fea_dim = 768
        
        text_fea_dim = 768
        
        vision_fea_dim = 709
        vision_seq_len = 55
        
        audio_fea_dim = 33
        audio_seq_len = 400
        
        vision_drop_out, audio_drop_out = 0.1, 0.5
        vision_nhead, audio_nhead = 8, 8
        vision_dim_feedforward = encoder_fea_dim 
        audio_dim_feedforward = encoder_fea_dim 
        vision_tf_num_layers, audio_tf_num_layers = 5, 3
        vision_attn_mask, audio_attn_mask = True, True
        
        audio_text_nhead, vision_text_nhead = 8, 8
        vision_text_tf_num_layers, audio_text_tf_num_layers = 5, 2
        drop_out = 0.6
        text_drop_out = 0.5
        attn_mask = True

        batch_size = 32
        update_epochs = 4
        
        alen=audio_seq_len
        vlen=vision_seq_len
        p_len= 3

        # ========== 可学习向量配置 ==========
        # 控制是否使用可学习提示向量 (Learnable Prompt Vectors)
        use_learnable_vectors = True  # True: 使用可学习向量, False: 不使用可学习向量
        
        # 删除预训练配置 - 不再需要单模态预训练
      
        class TVAtrain:	
            text_lr = 5e-5
            audio_lr = 1e-3
            vision_lr = 1e-3
            other_lr = 1e-3
            
            text_decay = 1e-3
            audio_decay = 1e-3
            vision_decay = 1e-2
            other_decay = 1e-2
            
            epoch = 25
            
            # 删除知识蒸馏和对比学习权重 - 不再使用
           
