import torch
import os

seed = 1111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(__file__)

LOGPATH = os.path.join(root_path, 'log/')

if not os.path.exists(LOGPATH):
    os.makedirs(LOGPATH)


class MOSI:
    class path:
        raw_data_path = '/home/liulei/Projects/MFON-main/data/MOSI/unaligned_50.pkl'
        model_path = os.path.join(root_path, 'save_models/all_model/MOSI/')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 删除uni_fea_encoder路径 - 不再需要预训练单模态编码器

    class downStream:
        language = 'en'

        encoder_fea_dim = 768

        text_fea_dim = 768

        vision_fea_dim = 20
        vision_seq_len = 500

        audio_fea_dim = 5
        audio_seq_len = 375

        vision_drop_out, audio_drop_out = 0.3, 0.0
        vision_nhead, audio_nhead = 8, 8


        vision_tf_num_layers, audio_tf_num_layers = 4, 3
        vision_attn_mask, audio_attn_mask = False, False

        audio_text_nhead, vision_text_nhead = 8, 8
        vision_text_tf_num_layers, audio_text_tf_num_layers = 2, 5

        drop_out = 0.1
        text_drop_out = 0.3
        attn_mask = False

        batch_size = 32
        update_epochs = 4

        alen=audio_seq_len
        vlen=vision_seq_len


        # ========== AtCAF因果推理配置（修复后） ==========
        use_atcaf_plugin = True        # 启用修复后的AtCAF因果推理
        causal_method = 'current'        # 固定为 'current'

        # Current方法配置（修复后）
        intervention_alpha = 0.5         # 因果干预权重（训练推理一致）

        # 完全保守模式：主预测仅用事实特征；反事实仅作正则
        atcaf_conservative = True
        atcaf_loss_consistency_only = True   # 因果损失仅用一致性项
        atcaf_warmup_epochs = 3              # 反事实正则权重的 warm-up 轮数

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
        vision_dict_npy = os.path.join(dict_npy_dir, f'kmeans_mosi-{vision_dict_size}_vision.npy')
        audio_dict_npy = os.path.join(dict_npy_dir, f'kmeans_mosi-{audio_dict_size}_audio.npy')

        # ========== 可学习向量配置 ==========
        use_learnable_vectors = False  # True: 使用可学习向量, False: 不使用可学习向量

        # ========== 可学习向量长度配置建议 ==========
        # 理论依据：可学习向量作为位置编码的增强，长度不需要与序列完全匹配
        #
        # 推荐设置（按优先级）：
        # 1. 保守型（推荐）: vision=64, audio=32  -> 73K参数，性价比高
        # 2. 平衡型: vision=128, audio=64         -> 147K参数，表达能力较强
        # 3. 激进型: vision=256, audio=128        -> 295K参数，表达能力最强
        # 4. MFON原版: vision=500, audio=375     -> 672K参数，参数量大
        #
        # 选择原则：
        # - 数据集小(<5K样本): 使用保守型，避免过拟合
        # - 数据集中(5K-20K): 使用平衡型
        # - 数据集大(>20K): 可尝试激进型
        # - 计算资源有限: 使用保守型
        #
        # MOSI数据集(~2.2K样本)建议使用保守型设置
        learnable_vision_len = 64   # 推荐64，原序列长度500，减少87%参数
        learnable_audio_len = 32    # 推荐32，原序列长度375，减少91%参数

        # 其他可选设置：
        # learnable_vision_len = vision_seq_len  # 与MFON-main一致 (500)
        # learnable_audio_len = audio_seq_len    # 与MFON-main一致 (375)
        # 可学习向量长度（默认与序列长度相同，可独立调节）
        learnable_vision_len = vision_seq_len  # 视觉可学习向量长度
        learnable_audio_len = audio_seq_len    # 音频可学习向量长度

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
        #   - 当前0.95：适配你的高相似度分布(0.9+)，但接近过拟合边界
        contrastive_alpha = 0.95

        # 调试建议：逐步测试权重和alpha
        # 如果SCL损失反升，尝试以下alpha值：0.1, 0.2, 0.3, 0.5
        # 如果相似度普遍很低(<0.3)，可以尝试更小的alpha如0.1或0.2
        # 三种对比学习损失的权重配置：
        # SCL (Semi-Contrastive Learning): 同一样本内不同模态对齐
        #   - 作用：让文本-视觉、文本-音频、视觉-音频的相似度接近 contrastive_alpha
        #   - 目标：模态融合，减少模态间的表示差异
        #   - 当前设置：0.01（较小权重，因为模态相似度已经很高 0.9+）
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
        focal_gamma = 2             # Focal的gamma
        focal_alpha = None            # None=按batch正负比例动态设置；或设为固定(如0.5/0.75)




        class TVAtrain:
            text_lr = 5e-5
            audio_lr = 1e-3
            vision_lr = 1e-3
            other_lr = 1e-3

            text_decay = 1e-3
            audio_decay = 1e-3
            vision_decay = 1e-3
            other_decay = 1e-3

            epoch = 6

            # 删除知识蒸馏和对比学习权重 - 不再使用

def get_config(args, mode):
    # 这是一个简单的示例函数，返回MOSI配置类
    # 在您的实际项目中，这里可能会有更复杂的逻辑来选择不同的配置

    # 创建一个配置对象，模拟主程序中的结构
    class Config:
        def __init__(self):
            self.DEVICE = DEVICE
            self.seed = seed
            if mode.upper() == 'MOSI':
                self.MOSI = MOSI()
            # 可以为其他数据集添加 else if 分支
            # elif mode.upper() == 'MOSEI':
            #     self.MOSEI = MOSEI()
            else:
                raise ValueError(f"未知的模式: {mode}")

    return Config()
