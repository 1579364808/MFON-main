import torch
import os

seed = 1111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(__file__)

LOGPATH = os.path.join(root_path, 'log/')

if not os.path.exists(LOGPATH): 
    os.makedirs(LOGPATH)
       

class MOSEI:
    class path:
        raw_data_path = '/home/liulei/Projects/MFON-main/data/MOSEI/unaligned_50.pkl'
        model_path = os.path.join(root_path, 'save_models/all_model/MOSEI/')
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        # 删除uni_fea_encoder路径 - 不再需要预训练单模态编码器

    class downStream:
        language = 'en'
      
        encoder_fea_dim = 768
        
        text_fea_dim = 768
        
        vision_fea_dim = 35
        vision_seq_len = 500
        
        audio_fea_dim = 74
        audio_seq_len = 500
        
        vision_drop_out, audio_drop_out = 0, 0
        vision_nhead, audio_nhead = 8, 8
        vision_dim_feedforward = encoder_fea_dim 
        audio_dim_feedforward = encoder_fea_dim 
        vision_tf_num_layers, audio_tf_num_layers = 4, 2
        vision_attn_mask, audio_attn_mask = False, False
        
        audio_text_nhead, vision_text_nhead = 8, 8
        vision_text_tf_num_layers, audio_text_tf_num_layers = 2, 3
        drop_out = 0.1
        text_drop_out = 0.3
        attn_mask = True
                
        alen=audio_seq_len
        vlen=vision_seq_len
        batch_size = 16
        update_epochs = 8

        # ========== 可学习向量配置 ==========
        # 控制是否使用可学习提示向量 (Learnable Prompt Vectors)
        use_learnable_vectors = False  # True: 使用可学习向量, False: 不使用可学习向量

        # ========== AtCAF因果推理配置（修复后） ==========
        use_atcaf_plugin = True        # 启用修复后的AtCAF因果推理
        causal_method = 'current'        # 固定为 'current'

        # Current方法配置（修复后）
        intervention_alpha = 0.5         # 因果干预权重（训练推理一致）

        # 完全保守模式：主预测仅用事实特征；反事实仅作正则
        atcaf_conservative = True
        atcaf_loss_consistency_only = True   # 因果损失仅用一致性项
        atcaf_warmup_epochs = 1              # 反事实正则权重的 warm-up 轮数

        # 反事实注意力配置
        counterfactual_type = 'shuffle'  # 支持: 'shuffle' | 'random' | 'uniform' | 'reversed'
        counterfactual_weight = 10

        # 检索查询池化策略：'first'（默认）|'mean'|'attn'
        confounder_query_pooling = 'mean'


        # ========== 全局字典与K-means配置 ==========
        use_confounder_dict = True       # 使用全局字典（来自K-means中心）

        # --- 视觉混淆因子配置 ---
        vision_dict_size = 10           # 视觉字典大小 (K)
        vision_kmeans_n_init = 10        # 视觉K-means初始化次数
        vision_kmeans_random_state = 42  # 视觉K-means随机种子

        # --- 音频混淆因子配置 ---
        audio_dict_size = 50             # 音频字典大小 (K)
        audio_kmeans_n_init = 10         # 音频K-means初始化次数
        audio_kmeans_random_state = 42   # 音频K-means随机种子

        # 预置的npy路径（若存在则直接加载）
        dict_npy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'npy_folder')
        vision_dict_npy = os.path.join(dict_npy_dir, f'kmeans_mosei-{vision_dict_size}_vision_{confounder_query_pooling}.npy')
        audio_dict_npy = os.path.join(dict_npy_dir, f'kmeans_mosei-{audio_dict_size}_audio_{confounder_query_pooling}.npy')





        # ========== 混淆因子字典与K-means配置 ==========
        # --- 视觉混淆因子配置 ---
        vision_dict_size = 10            # 视觉字典大小 (K-means中的K) - MOSEI较大，使用更大字典
        vision_kmeans_n_init = 10        # 视觉K-means初始化次数
        vision_kmeans_random_state = 42  # 视觉K-means随机种子

        # --- 音频混淆因子配置 ---
        audio_dict_size = 50             # 音频字典大小 (K-means中的K) - MOSEI较大，使用更大字典
        audio_kmeans_n_init = 10         # 音频K-means初始化次数
        audio_kmeans_random_state = 42   # 音频K-means随机种子

        # ========== 双向交叉注意力配置 ==========
        use_bidirectional_attention = True   # True: 启用双向注意力, False: 仅使用单向注意力
        bidirectional_fusion_mode = 'concat'  # 'concat' | 'weighted_sum' | 'gated'
        bidirectional_fusion_weight = 0.5   # 双向特征融合权重 (仅在 weighted_sum 模式使用)

        # ========== 混合对比学习 (HCL) 配置 ==========
        enable_hcl = True                # 是否启用 HCL（SCL/IAMCL/IEMCL）

        # contrastive_alpha: 模态边距参数，定义"理想的模态间相似度目标"
        #   - 在SCL中：作为同一样本内三模态对的目标相似度 (sim_tv, sim_ta, sim_va → alpha)
        #   - 在IEMCL中：作为跨模态正样本相似度的精炼目标 (pos_sims → alpha)
        #   - 不影响IAMCL：IAMCL的精炼目标固定为1
        #   - 取值建议：0.7-0.9，过高(>0.95)可能导致模态过度对齐失去独特性
        #   - MOSEI数据集建议：0.8（相比MOSI稍保守，因为MOSEI数据更复杂）
        contrastive_alpha = 0.95

        # 三种对比学习损失的权重配置：
        # SCL (Semi-Contrastive Learning): 同一样本内不同模态对齐
        #   - 作用：让文本-视觉、文本-音频、视觉-音频的相似度接近 contrastive_alpha
        #   - 目标：模态融合，减少模态间的表示差异
        #   - MOSEI设置：0.05（MOSEI数据更复杂，模态对齐权重适当降低）
        lambda_scl = 0.1

        # IAMCL (Intra-modal Contrastive Learning): 同模态内监督对比
        #   - 作用：在同一模态内，拉近同类样本，推远异类样本
        #   - 目标：增强单模态特征的判别性，提升分类边界清晰度
        #   - 包含：SupCon基础项 + 精炼项(正样本相似度→1)
        lambda_iamcl = 0.01

        # IEMCL (Inter-modal Contrastive Learning): 跨模态监督对比
        #   - 作用：不同模态间，拉近同类样本，推远异类样本
        #   - 目标：增强跨模态语义一致性，同类样本在不同模态下表示相近
        #   - 包含：SupCon基础项 + 精炼项(正样本相似度→contrastive_alpha)
        lambda_iemcl = 0.01

        # ========== 辅助分类头 + Focal Loss（可开关） ==========
        enable_aux_cls = True         # 开启独立分类头与Focal Loss
        focal_weight = 1            # 总损失中Focal的权重（建议0.05~0.2起步）
        focal_gamma = 2.0             # Focal的gamma
        focal_alpha = None            # None=按batch正负比例动态设置；或设为固定(如0.5/0.75)

        # 删除预训练配置 - 不再需要单模态预训练
     
        class TVAtrain:
            text_lr = 5e-5
            audio_lr = 1e-4
            vision_lr = 1e-4
            other_lr = 1e-4

            text_decay = 1e-3
            audio_decay = 1e-3
            vision_decay = 1e-3
            other_decay = 1e-3

            epoch = 25

            # 删除知识蒸馏和对比学习权重 - 不再使用
