import torch
import datetime
from tqdm import tqdm
from utils import write_log, set_random_seed
from models.adaptive_tva_fusion import AdaptiveTVA_fusion
from utils import write_config
import copy


class EarlyStopping:
    """
    早停机制类

    监控验证指标，当指标在指定的耐心值内没有改善时停止训练。
    支持保存和恢复最佳模型权重。

    Args:
        patience: 耐心值，连续多少个epoch没有改善就停止
        min_delta: 最小改善阈值，小于此值认为没有改善
        monitor: 监控的指标名称
        mode: 监控模式，'min'表示越小越好，'max'表示越大越好
        restore_best_weights: 是否在早停时恢复最佳权重
        verbose: 是否打印详细信息
    """

    def __init__(self, patience=2, min_delta=1e-6, monitor='val_loss',
                 mode='min', restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # 内部状态
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == 'min':
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best - min_delta
        else:  # mode == 'max'
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, current_score, model, epoch):
        """
        检查是否应该早停

        Args:
            current_score: 当前epoch的监控指标值
            model: 当前模型
            epoch: 当前epoch数

        Returns:
            bool: 是否应该停止训练
        """
        if self.is_better(current_score, self.best_score):
            # 发现更好的结果
            self.best_score = current_score
            self.wait = 0

            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())

            if self.verbose:
                print(f"📈 Epoch {epoch}: {self.monitor} improved to {current_score:.6f}")
        else:
            # 没有改善
            self.wait += 1
            if self.verbose:
                print(f"📊 Epoch {epoch}: {self.monitor} = {current_score:.6f}, "
                      f"no improvement for {self.wait}/{self.patience} epochs")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    print(f"   Best {self.monitor}: {self.best_score:.6f}")
                return True

        return False

    def restore_best_model(self, model):
        """
        恢复最佳模型权重

        Args:
            model: 要恢复权重的模型
        """
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"🔄 Restored best model weights (best {self.monitor}: {self.best_score:.6f})")
        else:
            if self.verbose:
                print("⚠️  No best weights to restore")


def AdaptiveTVA_train_fusion(config, metrics, seed, train_data, valid_data, test_data):
    """
    自适应TVA融合模型的训练函数
    
    核心创新：集成自适应引导模块(AGM)的训练流程
    
    与原始TVA_train_fusion的区别：
    1. 使用AdaptiveTVA_fusion模型
    2. 将AGM参数加入优化器
    3. 记录和分析引导权重的分布
    4. 支持引导权重的可视化和统计
    
    Args:
        config: 配置对象
        metrics: 评估指标对象
        seed: 随机种子
        train_data: 训练数据加载器
        valid_data: 验证数据加载器
    """
    print('---------------Adaptive TVA_EXP---------------')

    set_random_seed(seed)

    # 显示配置信息
    print("🔧 使用自适应TVA配置:")
    print(f"  总训练轮数: {config.MOSEI.downStream.adaptiveTVAtrain.epoch}")
    print(f"  AGM隐藏维度: {config.MOSEI.downStream.adaptiveTVAtrain.agm_hidden_dim}")
    print(f"  AGM学习率: {config.MOSEI.downStream.adaptiveTVAtrain.agm_lr}")
    print(f"  权重平滑: {config.MOSEI.downStream.adaptiveTVAtrain.weight_smoothing}")
    print()
    
    # ========== 训练配置 ==========
    update_epochs = config.MOSEI.downStream.update_epochs
    
    # ========== 使用自适应TVA专用配置 ==========
    adaptive_config = config.MOSEI.downStream.adaptiveTVAtrain

    # 学习率配置
    text_lr = adaptive_config.text_lr
    audio_lr = adaptive_config.audio_lr
    vision_lr = adaptive_config.vision_lr
    other_lr = adaptive_config.other_lr
    agm_lr = adaptive_config.agm_lr  # AGM专用学习率

    # 权重衰减配置
    text_decay = adaptive_config.text_decay
    audio_decay = adaptive_config.audio_decay
    vision_decay = adaptive_config.vision_decay
    other_decay = adaptive_config.other_decay
    agm_decay = adaptive_config.agm_decay  # AGM专用权重衰减

    # 损失权重配置
    delta_va = adaptive_config.delta_va     # 知识蒸馏权重
    delta_nce = adaptive_config.delta_nce   # 对比学习权重

    # AGM特有配置
    agm_hidden_dim = adaptive_config.agm_hidden_dim
    weight_smoothing = adaptive_config.weight_smoothing
    analyze_weights = adaptive_config.analyze_weight_distribution
    
    # ========== 模型初始化 ==========
    # 传递AGM配置到模型
    model = AdaptiveTVA_fusion(config, agm_hidden_dim=agm_hidden_dim).to(config.DEVICE)
    print("Adaptive TVA Fusion Model:")
    print(model)
    print(f"AGM Hidden Dimension: {agm_hidden_dim}")
    print(f"Weight Smoothing: {weight_smoothing}")

    # 加载预训练的冻结编码器
    model.load_froze()
    
    # ========== 参数分组（关键创新：包含AGM参数） ==========
    # 文本相关参数
    text_params = (list(model.proj_t.named_parameters()) + 
                  list(model.text_encoder.named_parameters()))
    text_params = [p for _, p in text_params]
    
    # 视觉相关参数
    vision_params = (list(model.proj_v.named_parameters()) +
                    list(model.vision_with_text.named_parameters()))
    vision_params = [p for _, p in vision_params]
    # 只有在使用可学习向量时才添加到优化器参数中
    if model.promptv_m is not None:
        vision_params.append(model.promptv_m)

    # 音频相关参数
    audio_params = (list(model.proj_a.named_parameters()) +
                   list(model.audio_with_text.named_parameters()))
    audio_params = [p for _, p in audio_params]
    # 只有在使用可学习向量时才添加到优化器参数中
    if model.prompta_m is not None:
        audio_params.append(model.prompta_m)
    
    # 其他参数（不包含AGM）
    model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n]

    # AGM参数（核心创新：使用专用配置）
    agm_params = list(model.adaptive_guidance.parameters())

    # ========== 优化器配置（精细化参数分组） ==========
    optimizer_grouped_parameters = [
        {'params': text_params, 'weight_decay': text_decay, 'lr': text_lr},
        {'params': audio_params, 'weight_decay': audio_decay, 'lr': audio_lr},
        {'params': vision_params, 'weight_decay': vision_decay, 'lr': vision_lr},
        {'params': model_params_other, 'weight_decay': other_decay, 'lr': other_lr},
        {'params': agm_params, 'weight_decay': agm_decay, 'lr': agm_lr}  # AGM专用配置
    ]

    print(f"参数分组配置:")
    print(f"  文本参数: lr={text_lr}, decay={text_decay}")
    print(f"  音频参数: lr={audio_lr}, decay={audio_decay}")
    print(f"  视觉参数: lr={vision_lr}, decay={vision_decay}")
    print(f"  其他参数: lr={other_lr}, decay={other_decay}")
    print(f"  AGM参数: lr={agm_lr}, decay={agm_decay}")  # 新增日志
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    
    # ========== 日志配置 ==========
    log_path = (config.LOGPATH + "MOSEI_AdaptiveTVA_Train." +
               datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log')
    write_config(config, log_path)
    
    # ========== 早停机制初始化 ==========
    early_stopping = None
    if adaptive_config.early_stopping:
        early_stopping = EarlyStopping(
            patience=adaptive_config.early_stopping_patience,
            min_delta=adaptive_config.early_stopping_min_delta,
            monitor=adaptive_config.early_stopping_monitor,
            mode=adaptive_config.early_stopping_mode,
            restore_best_weights=adaptive_config.early_stopping_restore_best,
            verbose=True
        )
        print(f"🛑 早停机制已启用:")
        print(f"   监控指标: {adaptive_config.early_stopping_monitor}")
        print(f"   耐心值: {adaptive_config.early_stopping_patience} epochs")
        print(f"   最小改善阈值: {adaptive_config.early_stopping_min_delta}")
        print()

    # ========== 训练循环 ==========
    best_loss = 1e8
    guidance_stats_history = []  # 记录引导权重统计历史
    training_stopped_early = False  # 早停标志

    for epoch in range(1, adaptive_config.epoch + 1):  # 使用自适应配置的epoch数
        model.train()
        left_epochs = update_epochs
        
        # 用于统计本epoch的引导权重
        epoch_guidance_weights = []
        
        bar = tqdm(train_data, disable=False)
        for index, sample in enumerate(bar):
            bar.set_description("Epoch:%d|" % epoch)
            
            if left_epochs == update_epochs:
                optimizer.zero_grad()
            left_epochs -= 1
            
            # ========== 数据准备 ==========
            text = sample['raw_text']
            vision = sample['vision'].clone().detach().to(config.DEVICE).float()
            audio = sample['audio'].clone().detach().to(config.DEVICE).float()
            label = sample['labels']['M'].clone().detach().to(config.DEVICE).float()
            
            # ========== 前向传播（核心创新：返回引导权重） ==========
            pred, (loss_v, loss_a, loss_nce, guidance_weights) = model(
                text, vision, audio, mode='train'
            )
            
            # 收集引导权重用于分析
            epoch_guidance_weights.append(guidance_weights.detach().cpu())
            
            # ========== 损失计算 ==========
            # 主任务损失（情感回归）
            main_loss = torch.nn.MSELoss()(pred, label.squeeze())
            
            # 总损失：主任务 + 知识蒸馏 + 对比学习
            total_loss = (main_loss + 
                         delta_va * (loss_v + loss_a) + 
                         delta_nce * loss_nce)
            
            total_loss.backward()
            
            if not left_epochs:
                optimizer.step()
                left_epochs = update_epochs
                
        # 处理最后一个batch
        if not left_epochs:
            optimizer.step()
        
        # ========== 验证和保存 ==========
        # 每轮训练后评估所有数据集并打印结果
        train_results, train_loss = eval_adaptive(model, metrics, train_data, config.DEVICE)
        valid_results, valid_loss = eval_adaptive(model, metrics, valid_data, config.DEVICE)
        test_results, test_loss = eval_adaptive(model, metrics, test_data, config.DEVICE)

        # 打印完整的评估结果
        print(f"\n📊 Epoch {epoch}/{adaptive_config.epoch} 完整评估结果:")
        print("=" * 80)

        # 训练集结果
        print("🔵 训练集结果:")
        print(f"   Loss: {train_results['Loss']:.6f}")
        print(f"   MAE: {train_results['MAE']:.6f}")
        print(f"   Corr: {train_results['Corr']:.6f}")
        print(f"   Has0_acc_2: {train_results['Has0_acc_2']:.6f}")
        print(f"   Has0_F1_score: {train_results['Has0_F1_score']:.6f}")
        print(f"   Non0_acc_2: {train_results['Non0_acc_2']:.6f}")
        print(f"   Non0_F1_score: {train_results['Non0_F1_score']:.6f}")
        print(f"   Mult_acc_5: {train_results['Mult_acc_5']:.6f}")
        print(f"   Mult_acc_7: {train_results['Mult_acc_7']:.6f}")

        # 验证集结果
        print("🟡 验证集结果:")
        print(f"   Loss: {valid_results['Loss']:.6f}")
        print(f"   MAE: {valid_results['MAE']:.6f}")
        print(f"   Corr: {valid_results['Corr']:.6f}")
        print(f"   Has0_acc_2: {valid_results['Has0_acc_2']:.6f}")
        print(f"   Has0_F1_score: {valid_results['Has0_F1_score']:.6f}")
        print(f"   Non0_acc_2: {valid_results['Non0_acc_2']:.6f}")
        print(f"   Non0_F1_score: {valid_results['Non0_F1_score']:.6f}")
        print(f"   Mult_acc_5: {valid_results['Mult_acc_5']:.6f}")
        print(f"   Mult_acc_7: {valid_results['Mult_acc_7']:.6f}")

        # 测试集结果
        print("🔴 测试集结果:")
        print(f"   Loss: {test_results['Loss']:.6f}")
        print(f"   MAE: {test_results['MAE']:.6f}")
        print(f"   Corr: {test_results['Corr']:.6f}")
        print(f"   Has0_acc_2: {test_results['Has0_acc_2']:.6f}")
        print(f"   Has0_F1_score: {test_results['Has0_F1_score']:.6f}")
        print(f"   Non0_acc_2: {test_results['Non0_acc_2']:.6f}")
        print(f"   Non0_F1_score: {test_results['Non0_F1_score']:.6f}")
        print(f"   Mult_acc_5: {test_results['Mult_acc_5']:.6f}")
        print(f"   Mult_acc_7: {test_results['Mult_acc_7']:.6f}")

        # 使用验证集结果进行后续处理
        eval_results = valid_results
        result_loss = valid_loss

        # 分析本epoch的引导权重分布
        if epoch_guidance_weights:
            epoch_weights = torch.cat(epoch_guidance_weights, dim=0)  # [total_samples, 3]
            guidance_analysis = model.analyze_guidance_weights(epoch_weights)

            # 🆕 获取负载均衡统计
            load_balance_stats = model.adaptive_guidance.get_load_statistics()
            guidance_analysis['load_balance_stats'] = load_balance_stats
            guidance_stats_history.append(guidance_analysis)

            # 记录引导权重统计
            guidance_log = (f"Epoch {epoch} Guidance Analysis:\n"
                          f"  Mean weights - Text: {guidance_analysis['mean_text_weight']:.3f}, "
                          f"Vision: {guidance_analysis['mean_vision_weight']:.3f}, "
                          f"Audio: {guidance_analysis['mean_audio_weight']:.3f}\n"
                          f"  Dominant counts - Text: {guidance_analysis['text_dominant_count']}, "
                          f"Vision: {guidance_analysis['vision_dominant_count']}, "
                          f"Audio: {guidance_analysis['audio_dominant_count']}")

            # 🆕 添加负载均衡信息
            if model.adaptive_guidance.enable_load_balancing:
                balance_log = (f"  Load Balancing Status:\n"
                             f"    Modality bias - Text: {load_balance_stats['modality_bias'][0]:.4f}, "
                             f"Vision: {load_balance_stats['modality_bias'][1]:.4f}, "
                             f"Audio: {load_balance_stats['modality_bias'][2]:.4f}\n"
                             f"    Load history - Text: {load_balance_stats['load_history'][0]:.3f}, "
                             f"Vision: {load_balance_stats['load_history'][1]:.3f}, "
                             f"Audio: {load_balance_stats['load_history'][2]:.3f}\n"
                             f"    Target balance ratio: {load_balance_stats['target_balance_ratio']:.3f}")
                guidance_log += "\n" + balance_log

            print(guidance_log)
            write_log(guidance_log, log_path)

        # 保存最佳模型
        if result_loss < best_loss:
            best_loss = result_loss
            model.save_model()
            print(f"\n🎉 新的最佳模型已保存! (验证集 Best Loss: {best_loss:.6f})")

            # 记录最佳模型的引导权重分布
            if epoch_guidance_weights:
                best_guidance_log = f"Best model guidance distribution saved at epoch {epoch}"
                write_log(best_guidance_log, log_path)
        else:
            print(f"\n📈 当前最佳: 验证集 Loss: {best_loss:.6f}")
        print("=" * 80)

        # ========== 早停检查 ==========
        if early_stopping is not None:
            # 根据配置的监控指标进行早停检查
            if adaptive_config.early_stopping_monitor == 'val_loss':
                monitor_value = result_loss
            elif adaptive_config.early_stopping_monitor == 'val_acc':
                # 假设使用Has0_acc_2作为准确率指标
                monitor_value = eval_results.get('Has0_acc_2', 0.0)
            elif adaptive_config.early_stopping_monitor == 'val_f1':
                # 假设使用Has0_F1_score作为F1指标
                monitor_value = eval_results.get('Has0_F1_score', 0.0)
            else:
                monitor_value = result_loss  # 默认使用验证损失

            # 检查是否应该早停
            if early_stopping(monitor_value, model, epoch):
                training_stopped_early = True
                early_stop_log = (f"🛑 Training stopped early at epoch {epoch}\n"
                                f"   Best {adaptive_config.early_stopping_monitor}: {early_stopping.best_score:.6f}\n"
                                f"   No improvement for {adaptive_config.early_stopping_patience} consecutive epochs")
                print(early_stop_log)
                write_log(early_stop_log, log_path)

                # 恢复最佳模型权重
                if adaptive_config.early_stopping_restore_best:
                    early_stopping.restore_best_model(model)
                    model.save_model(name='early_stopped_best')
                    restore_log = "🔄 Best model weights restored and saved"
                    print(restore_log)
                    write_log(restore_log, log_path)

                break  # 跳出训练循环

        # 记录训练结果
        log_info = {
            'epoch': epoch,
            'train_loss': total_loss.item(),
            'eval_loss': result_loss,
            **eval_results
        }
        write_log(log_info, log_path)
    
    # ========== 训练完成总结 ==========
    training_summary = {
        'guidance_stats_history': guidance_stats_history,
        'stopped_early': training_stopped_early,
        'total_epochs': epoch,
        'best_loss': best_loss
    }

    if training_stopped_early:
        training_summary['early_stop_epoch'] = early_stopping.stopped_epoch
        training_summary['best_monitor_score'] = early_stopping.best_score

        final_summary = (f"\n🎯 训练完成总结:\n"
                        f"   早停触发: 是 (第 {early_stopping.stopped_epoch} epoch)\n"
                        f"   最佳 {adaptive_config.early_stopping_monitor}: {early_stopping.best_score:.6f}\n"
                        f"   总训练轮数: {epoch}/{adaptive_config.epoch}")
    else:
        final_summary = (f"\n🎯 训练完成总结:\n"
                        f"   早停触发: 否\n"
                        f"   完整训练轮数: {epoch}\n"
                        f"   最佳验证损失: {best_loss:.6f}")

    print(final_summary)
    write_log(final_summary, log_path)

    return training_summary  # 返回完整的训练总结


def eval_adaptive(model, metrics, eval_data, device):
    """
    自适应TVA模型的评估函数
    
    Args:
        model: 自适应TVA融合模型
        metrics: 评估指标对象
        eval_data: 评估数据加载器
        device: 计算设备
        
    Returns:
        eval_results: 评估结果字典
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pred_list = []
        truth_list = []
        guidance_weights_list = []
        
        for sample in eval_data:
            text = sample['raw_text']
            vision = sample['vision'].clone().detach().to(device).float()
            audio = sample['audio'].clone().detach().to(device).float()
            label = sample['labels']['M'].clone().detach().to(device).float()
            
            # 推理模式：返回预测和引导权重
            pred, guidance_weights = model(text, vision, audio, mode='eval')
            
            # 计算损失
            loss = torch.nn.MSELoss()(pred, label.squeeze())
            total_loss += loss.item() * len(label)
            total_samples += len(label)
            
            # 收集结果
            pred_list.append(pred.cpu())
            truth_list.append(label.cpu())
            guidance_weights_list.append(guidance_weights.cpu())
        
        # 合并结果
        all_preds = torch.cat(pred_list).squeeze()
        all_truths = torch.cat(truth_list).squeeze()
        all_guidance_weights = torch.cat(guidance_weights_list, dim=0)
        
        # 计算评估指标
        eval_results = metrics.eval_mosei_regression(all_truths, all_preds)
        
        # 分析引导权重分布
        guidance_analysis = model.analyze_guidance_weights(all_guidance_weights)
        eval_results.update({
            'guidance_text_weight': guidance_analysis['mean_text_weight'],
            'guidance_vision_weight': guidance_analysis['mean_vision_weight'],
            'guidance_audio_weight': guidance_analysis['mean_audio_weight']
        })
    
    model.train()
    return eval_results, total_loss / total_samples


def AdaptiveTVA_test_fusion(config, metrics, test_data):
    """
    自适应TVA融合模型的测试函数
    
    Args:
        config: 配置对象
        metrics: 评估指标对象
        test_data: 测试数据加载器
    """
    log_path = (config.LOGPATH + "MOSEI_AdaptiveTVA_Test." +
               datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log')
    write_config(config, log_path)
    
    # 加载模型
    model = AdaptiveTVA_fusion(config).to(config.DEVICE)
    model.load_froze()
    model.load_model()
    
    # 测试
    result, loss = eval_adaptive(model, metrics, test_data, config.DEVICE)
    
    # 记录结果
    log = ('\nAdaptive TVA Test Results:\n'
           f'\tHas0_acc_2: {result["Has0_acc_2"]}\n'
           f'\tHas0_F1_score: {result["Has0_F1_score"]}\n'
           f'\tNon0_acc_2: {result["Non0_acc_2"]}\n'
           f'\tNon0_F1_score: {result["Non0_F1_score"]}\n'
           f'\tMult_acc_5: {result["Mult_acc_5"]}\n'
           f'\tMult_acc_7: {result["Mult_acc_7"]}\n'
           f'\tMAE: {result["MAE"]}\n'
           f'\tCorr: {result["Corr"]}\n'
           f'\tLoss: {loss}\n'
           f'\tGuidance - Text: {result["guidance_text_weight"]:.3f}, '
           f'Vision: {result["guidance_vision_weight"]:.3f}, '
           f'Audio: {result["guidance_audio_weight"]:.3f}\n'
           '------------------------------------------')
    
    print(log)
    write_log(log, log_path)
