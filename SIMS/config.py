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
        raw_data_path = '/home/liulei/Projects/MFON-main/data/SIMS/Processed/unaligned_39.pkl'
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

        # ========== AtCAF因果推理配置（与 MOSI/MOSEI 对齐） ==========
        use_atcaf_plugin = True            # 是否启用AtCAF插件

        # 因果推理基础配置
        atcaf_conservative = True          # 保守模式：只使用事实特征进行主预测
        atcaf_loss_consistency_only = True # 只使用一致性损失，不包含准确性项
        atcaf_warmup_epochs = 3            # AtCAF损失预热轮数
        intervention_alpha = 0.7           # 因果干预混合系数

        # 反事实注意力配置
        counterfactual_type = 'shuffle'    # 反事实类型: 'shuffle', 'random', 'uniform', 'reversed'
        counterfactual_weight = 10         # 反事实损失权重（与MOSI对齐）
        causal_method = 'current'          # 固定使用 current 方法

        # 混淆因子字典配置
        use_confounder_dict = True         # 是否使用混淆因子字典
        vision_dict_size = 40              # 视觉混淆因子字典大小（SIMS中文数据集，适中大小）
        audio_dict_size = 25               # 音频混淆因子字典大小（SIMS音频特征维度较小）

        # 查询池化策略配置
        confounder_query_pooling = 'first' # 混淆因子查询池化：'first' | 'mean' | 'attn'

        # K-means初始化超参数
        vision_kmeans_n_init = 10          # 视觉K-means初始化次数
        vision_kmeans_random_state = 42    # 视觉K-means随机种子
        audio_kmeans_n_init = 10           # 音频K-means初始化次数
        audio_kmeans_random_state = 42     # 音频K-means随机种子

        # 预置的npy路径（若存在则直接加载）
        dict_npy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'npy_folder')
        vision_dict_npy = os.path.join(dict_npy_dir, f'kmeans_sims-{vision_dict_size}_vision.npy')
        audio_dict_npy = os.path.join(dict_npy_dir, f'kmeans_sims-{audio_dict_size}_audio.npy')

        # ========== 可学习向量配置 ==========
        # 控制是否使用可学习提示向量 (Learnable Prompt Vectors)
        use_learnable_vectors = True  # True: 使用可学习向量, False: 不使用可学习向量

        # ========== 双向交叉注意力配置 ==========
        use_bidirectional_attention = True   # True: 启用双向注意力, False: 仅使用单向注意力
        bidirectional_fusion_mode = 'concat'  # 'concat' | 'weighted_sum' | 'gated'
        bidirectional_fusion_weight = 0.5   # 双向特征融合权重 (仅在 weighted_sum 模式使用)

        # ========== 混合对比学习 (HCL) 配置 ==========
        enable_hcl = True                # 是否启用 HCL（SCL/IAMCL/IEMCL）

        # contrastive_alpha: 模态边距参数，定义"理想的模态间相似度目标"
        #   - 在SCL中：作为同一样本内三模态对的目标相似度 (sim_tv, sim_ta, sim_va → alpha)
        #   - 在IEMCL中：作为跨模态正样本相似度的精炼目标 (pos_sims → alpha)
        #   - 不影响IAMCL：IAMCL的精炼项目标固定为1
        #   - 取值建议：0.7-0.9，过高(>0.95)可能导致模态过度对齐失去独特性
        #   - SIMS中文数据集建议：0.75（中文语义特点，适中对齐强度）
        contrastive_alpha = 0.75

        # 三种对比学习损失的权重配置：
        # SCL (Semi-Contrastive Learning): 同一样本内不同模态对齐
        #   - 作用：让文本-视觉、文本-音频、视觉-音频的相似度接近 contrastive_alpha
        #   - 目标：模态融合，减少模态间的表示差异
        #   - SIMS设置：0.02（中文数据集，模态对齐更保守）
        lambda_scl = 0.02

        # IAMCL (Intra-modal Contrastive Learning): 同模态内监督对比
        #   - 作用：在同一模态内，拉近同类样本，推远异类样本
        #   - 目标：增强单模态特征的判别性，提升分类边界清晰度
        #   - 包含：SupCon基础项 + 精炼项(正样本相似度→1)
        lambda_iamcl = 0.08

        # IEMCL (Inter-modal Contrastive Learning): 跨模态监督对比
        #   - 作用：不同模态间，拉近同类样本，推远异类样本
        #   - 目标：增强跨模态语义一致性，同类样本在不同模态下表示相近
        #   - 包含：SupCon基础项 + 精炼项(正样本相似度→contrastive_alpha)
        lambda_iemcl = 0.08

        # ========== 辅助分类头 + Focal Loss（可开关） ==========
        enable_aux_cls = True         # 开启独立分类头与Focal Loss
        focal_weight = 0.08           # 总损失中Focal的权重（SIMS稍保守）
        focal_gamma = 2.0             # Focal的gamma
        focal_alpha = None            # None=按batch正负比例动态设置；或设为固定(如0.5/0.75)

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
           
