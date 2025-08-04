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
        raw_data_path = '../data/MOSEI/unaligned_50.pkl'
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
        
        vision_drop_out, audio_drop_out = 0., 0
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
        batch_size = 32
        update_epochs = 4

        # ========== 可学习向量配置 ==========
        # 控制是否使用可学习提示向量 (Learnable Prompt Vectors)
        use_learnable_vectors = True  # True: 使用可学习向量, False: 不使用可学习向量
        
        class visionPretrain:
            lr = 1e-4
            epoch = 25
            decay = 1e-3
	
        class audioPretrain:
            lr = 1e-4
            epoch = 25
            decay = 1e-3
     
        class TVAtrain:
            text_lr = 5e-5
            audio_lr = 1e-3
            vision_lr = 1e-4
            other_lr = 1e-3

            text_decay = 1e-2
            audio_decay = 1e-3
            vision_decay = 1e-3
            other_decay = 1e-3

            epoch = 25

            # 删除知识蒸馏和对比学习权重 - 不再使用

        class adaptiveTVAtrain:
            """
            自适应TVA融合模型的训练配置

            只包含实际使用的配置参数，保持简洁和实用性。
            与原始TVAtrain的区别：
            1. 新增AGM相关的超参数配置
            2. 为AGM模块提供专门的学习率和权重衰减
            3. 集成早停机制配置
            4. 保持与原始配置的完全解耦
            """

            # ========== 基础学习率配置 ==========
            text_lr = 5e-5      # 文本编码器学习率（BERT微调）
            audio_lr = 1e-3     # 音频相关模块学习率
            vision_lr = 1e-4    # 视觉相关模块学习率
            other_lr = 1e-3     # 其他模块学习率
            agm_lr = 1e-3       # AGM门控网络学习率

            # ========== 权重衰减配置 ==========
            text_decay = 1e-2   # 文本模块权重衰减
            audio_decay = 1e-3  # 音频模块权重衰减
            vision_decay = 1e-3 # 视觉模块权重衰减
            other_decay = 1e-3  # 其他模块权重衰减
            agm_decay = 1e-4    # AGM模块权重衰减

            # ========== 训练轮数配置 ==========
            epoch = 25          # 总训练轮数

            # ========== 损失权重配置 ==========
            # 总损失 = 主任务损失 + delta_va * 知识蒸馏损失 + delta_nce * 对比学习损失
            # 使用网格搜索优化这些权重参数
            delta_va = 0.3      # 知识蒸馏损失权重 (控制预训练知识保持)
            delta_nce = 0.001   # 对比学习损失权重 (控制模态间语义对齐)

            # 网格搜索建议范围:
            # delta_va: [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0] - 知识蒸馏权重
            # delta_nce: [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] - 对比学习权重

            # ========== AGM配置 ==========
            agm_hidden_dim = 128        # AGM门控网络隐藏层维度
            agm_dropout = 0.1           # AGM内部dropout率
            agm_activation = 'relu'     # AGM激活函数类型

            # 🆕 无损负载均衡配置 (基于DeepSeek方法)
            enable_load_balancing = True        # 是否启用负载均衡
            load_balance_update_rate = 0.01     # 偏置更新率 (类似DeepSeek的u参数)
            target_balance_ratio = 0.15         # 非主导模态的目标权重比例

            # ========== 分析配置 ==========
            weight_smoothing = True             # 是否启用权重平滑（显示用）
            analyze_weight_distribution = True  # 是否分析权重分布

            # ========== 早停机制配置 ==========
            early_stopping = True              # 是否启用早停机制
            early_stopping_patience = 2       # 早停耐心值
            early_stopping_min_delta = 0  # 最小改善阈值
            early_stopping_restore_best = True # 早停时是否恢复最佳模型权重
            early_stopping_monitor = 'val_loss'  # 监控的指标
            early_stopping_mode = 'min'          # 监控模式

            @classmethod
            def validate_config(cls):
                """
                验证自适应TVA配置的合理性

                Returns:
                    bool: 配置是否有效
                    list: 警告信息列表
                """
                warnings = []
                is_valid = True

                # 验证学习率合理性
                if cls.agm_lr > cls.other_lr * 10:
                    warnings.append("AGM学习率可能过大，建议与other_lr保持相近")

                # 验证激活函数
                valid_activations = ['relu', 'gelu', 'tanh']
                if cls.agm_activation not in valid_activations:
                    warnings.append(f"不支持的激活函数: {cls.agm_activation}，支持的函数: {valid_activations}")
                    is_valid = False

                # 🆕 验证负载均衡配置
                if cls.enable_load_balancing:
                    if cls.load_balance_update_rate <= 0 or cls.load_balance_update_rate > 1:
                        warnings.append("load_balance_update_rate应该在(0, 1]范围内")
                        is_valid = False

                    if cls.target_balance_ratio <= 0 or cls.target_balance_ratio >= 0.5:
                        warnings.append("target_balance_ratio应该在(0, 0.5)范围内")
                        is_valid = False

                # 验证早停配置
                if cls.early_stopping:
                    if cls.early_stopping_patience < 1:
                        warnings.append("early_stopping_patience应该大于等于1")
                        is_valid = False

                    valid_monitors = ['val_loss', 'val_acc', 'val_f1']
                    if cls.early_stopping_monitor not in valid_monitors:
                        warnings.append(f"不支持的监控指标: {cls.early_stopping_monitor}，支持的指标: {valid_monitors}")
                        is_valid = False

                    valid_modes = ['min', 'max']
                    if cls.early_stopping_mode not in valid_modes:
                        warnings.append(f"不支持的监控模式: {cls.early_stopping_mode}，支持的模式: {valid_modes}")
                        is_valid = False

                return is_valid, warnings

            @classmethod
            def print_config_summary(cls):
                """
                打印配置摘要
                """
                print("="*60)
                print("🔧 自适应TVA配置摘要")
                print("="*60)
                print(f"训练轮数: {cls.epoch}")
                print(f"AGM隐藏层维度: {cls.agm_hidden_dim}")
                print(f"AGM学习率: {cls.agm_lr}")
                print(f"AGM权重衰减: {cls.agm_decay}")
                print(f"权重平滑: {cls.weight_smoothing}")
                print(f"分析权重分布: {cls.analyze_weight_distribution}")

                # 🆕 负载均衡配置
                print(f"负载均衡: {cls.enable_load_balancing}")
                if cls.enable_load_balancing:
                    print(f"  更新率: {cls.load_balance_update_rate}")
                    print(f"  目标平衡比例: {cls.target_balance_ratio}")

                print(f"早停机制: {cls.early_stopping} (耐心值: {cls.early_stopping_patience})")
                print(f"早停监控: {cls.early_stopping_monitor} ({cls.early_stopping_mode})")

                # 验证配置
                is_valid, warnings = cls.validate_config()
                if warnings:
                    print("\n⚠️  配置警告:")
                    for warning in warnings:
                        print(f"  - {warning}")

                if is_valid:
                    print("\n✅ 配置验证通过")
                else:
                    print("\n❌ 配置验证失败，请检查上述警告")

                print("="*60)
