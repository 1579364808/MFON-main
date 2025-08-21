import torch
import datetime
import os
from tqdm import tqdm

from utils import write_log, set_random_seed, write_config
from models.model import TVA_fusion
import torch.nn.functional as F

def focal_bce_with_logits(logits, targets, alpha=0.5, gamma=2.0):
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()



def TVA_train_fusion(config, metrics, seed, train_data, valid_data, test_data):
    print('---------------TVA_EXP---------------')

    set_random_seed(seed)

    update_epochs = config.SIMS.downStream.update_epochs

    text_lr = config.SIMS.downStream.TVAtrain.text_lr
    audio_lr = config.SIMS.downStream.TVAtrain.audio_lr
    vision_lr = config.SIMS.downStream.TVAtrain.vision_lr
    other_lr = config.SIMS.downStream.TVAtrain.other_lr

    text_decay = config.SIMS.downStream.TVAtrain.text_decay
    audio_decay = config.SIMS.downStream.TVAtrain.audio_decay
    vision_decay = config.SIMS.downStream.TVAtrain.vision_decay
    other_decay = config.SIMS.downStream.TVAtrain.other_decay

    # 删除知识蒸馏和对比学习权重 - 不再使用

    model = TVA_fusion(config).to(config.DEVICE)
    # 删除load_froze调用 - 不再需要加载冻结编码器

    text_params = list(model.proj_t.named_parameters()) + list(model.text_encoder.named_parameters())
    text_params = [p for _, p in text_params]
    vision_params = list(model.proj_v.named_parameters()) +\
                list(model.vision_with_text.named_parameters())
    vision_params = [p for _, p in vision_params]
    # 只有在使用可学习向量时才添加到优化器参数中
    if model.promptv_m is not None:
        vision_params.append(model.promptv_m)

    audio_params = list(model.proj_a.named_parameters()) +\
                list(model.audio_with_text.named_parameters())
    audio_params = [p for _, p in audio_params]
    # 只有在使用可学习向量时才添加到优化器参数中
    if model.prompta_m is not None:
        audio_params.append(model.prompta_m)
    model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n]

    # 辅助分类头参数
    aux_params = []
    if getattr(model, 'enable_aux_cls', False) and getattr(model, 'cls_head', None) is not None:
        aux_params = list(model.cls_head.parameters())

    optimizer_grouped_parameters = [
        {'params': text_params, 'weight_decay': text_decay, 'lr': text_lr},
        {'params': audio_params, 'weight_decay': audio_decay, 'lr': audio_lr},
        {'params': vision_params, 'weight_decay': vision_decay, 'lr': vision_lr},
        {'params': model_params_other, 'weight_decay': other_decay, 'lr': other_lr}
    ] + ([{'params': aux_params, 'weight_decay': other_decay, 'lr': other_lr}] if aux_params else [])
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)

    loss, best_loss  = 0, 1e8
    pred_loss = 0  # 只保留预测损失
    device = config.DEVICE
    total_epoch = config.SIMS.downStream.TVAtrain.epoch
    best_epoch = 1

    # ========== AtCAF K-means 初始化检查 ==========
    need_atcaf_init = config.SIMS.downStream.use_atcaf_plugin and not model.atcaf_plugin.vision_dict.initialized

    if need_atcaf_init:
        print("\n🔄 检测到混淆因子字典未初始化，开始收集特征...")
        model.train()

        feature_bar = tqdm(train_data, desc="收集特征用于K-means初始化", disable=False)
        for batch_data in feature_bar:
            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            with torch.no_grad():
                _ = model(text, vision, audio, mode='train')

        print("✅ 特征收集完成，开始K-means聚类...")
        model.atcaf_plugin.initialize_kmeans_dictionaries()
        print("🎯 混淆因子字典初始化完成！")

    # 初始化损失统计变量
    running_pred_loss = 0.0
    running_atcaf_loss = 0.0
    running_total_loss = 0.0
    batch_count = 0

    # 用于保存最佳模型的变量
    best_valid_loss = float('inf')
    best_epoch = -1
    best_model_path = None

    for epoch in range(1, total_epoch + 1):
        print(f"\n🚀 开始第 {epoch}/{total_epoch} 个Epoch")

        model.train()
        left_epochs = update_epochs
        bar = tqdm(train_data, disable=False)

        # 重置每个epoch的损失统计
        running_pred_loss = 0.0
        running_atcaf_loss = 0.0
        running_atcaf_weighted_loss = 0.0  # 新增：加权后的AtCAF损失
        running_total_loss = 0.0
        batch_count = 0



        for batch_data in bar:
            try:
                # 动态构建进度条描述，只显示启用的损失项
                desc_parts = [f"Epoch:{epoch}", f"Total:[{loss.item():.4f}]", f"Pred:[{pred_loss.item():.4f}]"]

                # AtCAF损失（如果启用）
                if getattr(model.config.SIMS.downStream, 'use_atcaf_plugin', False):
                    desc_parts.append(f"AtCAF(raw):[{atcaf_loss_raw.item():.6f}]")
                    atcaf_w_val = atcaf_loss_weighted.item() if 'atcaf_loss_weighted' in locals() and torch.is_tensor(atcaf_loss_weighted) else (atcaf_loss_weighted if 'atcaf_loss_weighted' in locals() else 0.0)
                    desc_parts.append(f"AtCAF(w):[{atcaf_w_val:.6f}]")

                # HCL损失（只显示启用的）
                if getattr(model.config.SIMS.downStream, 'enable_hcl', False):
                    lambda1 = getattr(model.config.SIMS.downStream, 'lambda_scl', 0.0)
                    lambda2 = getattr(model.config.SIMS.downStream, 'lambda_iamcl', 0.0)
                    lambda3 = getattr(model.config.SIMS.downStream, 'lambda_iemcl', 0.0)

                    if lambda1 != 0:
                        scl_val = L_scl.item() if 'L_scl' in locals() and torch.is_tensor(L_scl) else 0.0
                        desc_parts.append(f"SCL:[{scl_val:.4f}]")
                    if lambda2 != 0:
                        iamcl_val = L_iamcl.item() if 'L_iamcl' in locals() and torch.is_tensor(L_iamcl) else 0.0
                        desc_parts.append(f"IAMCL:[{iamcl_val:.4f}]")
                    if lambda3 != 0:
                        iemcl_val = L_iemcl.item() if 'L_iemcl' in locals() and torch.is_tensor(L_iemcl) else 0.0
                        desc_parts.append(f"IEMCL:[{iemcl_val:.4f}]")

                    # 如果有任何HCL损失启用，显示混合损失
                    if lambda1 != 0 or lambda2 != 0 or lambda3 != 0:
                        hybrid_val = L_hybrid.item() if 'L_hybrid' in locals() and torch.is_tensor(L_hybrid) else 0.0
                        desc_parts.append(f"Hybrid:[{hybrid_val:.4f}]")

                # Focal Loss（如果启用）
                if getattr(model.config.SIMS.downStream, 'enable_aux_cls', False):
                    focal_val = L_focal.item() if 'L_focal' in locals() and torch.is_tensor(L_focal) else 0.0
                    desc_parts.append(f"FLoss:[{focal_val:.4f}]")

                bar.set_description("|".join(desc_parts) + "|")
            except:
                bar.set_description("Epoch:%d|损失计算中...|" % epoch)

            if left_epochs == update_epochs:
                optimizer.zero_grad()
            left_epochs -= 1

            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().to(device).float()

            if getattr(model.config.SIMS.downStream, 'enable_aux_cls', False):
                pred, atcaf_loss_raw, embeds, cls_logits = model(text, vision, audio, mode='train', labels=label)
            else:
                pred, atcaf_loss_raw, embeds = model(text, vision, audio, mode='train', labels=label)
                cls_logits = None
            x_t, x_v, x_a = embeds  # [B, D] each

            # 调试：检查原始嵌入
            if batch_count < 3 and epoch == 1:
                print(f"\n[调试] 原始嵌入形状: x_t={x_t.shape}, x_v={x_v.shape}, x_a={x_a.shape}")
                print(f"[调试] 原始嵌入范数: t={torch.norm(x_t, dim=1).mean().item():.4f}, v={torch.norm(x_v, dim=1).mean().item():.4f}, a={torch.norm(x_a, dim=1).mean().item():.4f}")
                print(f"[调试] 嵌入是否需要梯度: t={x_t.requires_grad}, v={x_v.requires_grad}, a={x_a.requires_grad}")

            # L2-normalize embeddings before any contrastive losses
            from train.contrastive_losses import l2_normalize, compute_scl_loss, compute_iamcl_loss, compute_iemcl_loss
            x_t, x_v, x_a = l2_normalize(x_t, x_v, x_a)

            pred_loss = torch.mean((pred-label)*(pred-label))  # [bs]

            # 初始化当前批次的总损失
            loss = pred_loss
            atcaf_loss_weighted = torch.tensor(0.0, device=device, dtype=torch.float32) # 初始化加权损失

            # ===== 混合对比损失 =====
            if getattr(model.config.SIMS.downStream, 'enable_hcl', False):
                alpha = getattr(model.config.SIMS.downStream, 'contrastive_alpha', 0.5)
                lambda1 = getattr(model.config.SIMS.downStream, 'lambda_scl', 0.0)
                lambda2 = getattr(model.config.SIMS.downStream, 'lambda_iamcl', 0.0)
                lambda3 = getattr(model.config.SIMS.downStream, 'lambda_iemcl', 0.0)

                # 转换连续标签为二元类别（>0 正类，否则负类）
                cls_labels = (label.view(-1) > 0).long()

                # 调试信息：每100个batch打印一次类别分布和相似度统计
                if batch_count % 100 == 0:
                    pos_count = (cls_labels == 1).sum().item()
                    neg_count = (cls_labels == 0).sum().item()

                    # 计算相似度统计
                    with torch.no_grad():
                        sim_tv = torch.sum(x_t * x_v, dim=1)
                        sim_ta = torch.sum(x_t * x_a, dim=1)
                        sim_va = torch.sum(x_v * x_a, dim=1)

                        print(f"\n[调试] Batch {batch_count}:")
                        print(f"  类别分布: 正类={pos_count}, 负类={neg_count}, 比例={pos_count/(pos_count+neg_count):.3f}")
                        print(f"  相似度统计 (alpha={alpha:.1f}):")
                        print(f"    sim_tv: mean={sim_tv.mean():.3f}, std={sim_tv.std():.3f}, range=[{sim_tv.min():.3f}, {sim_tv.max():.3f}]")
                        print(f"    sim_ta: mean={sim_ta.mean():.3f}, std={sim_ta.std():.3f}, range=[{sim_ta.min():.3f}, {sim_ta.max():.3f}]")
                        print(f"    sim_va: mean={sim_va.mean():.3f}, std={sim_va.std():.3f}, range=[{sim_va.min():.3f}, {sim_va.max():.3f}]")

                        # 检查特征本身的分布
                        print(f"  特征范数检查:")
                        print(f"    x_t norm: mean={torch.norm(x_t, dim=1).mean():.3f}")
                        print(f"    x_v norm: mean={torch.norm(x_v, dim=1).mean():.3f}")
                        print(f"    x_a norm: mean={torch.norm(x_a, dim=1).mean():.3f}")

                        # 预期SCL损失
                        expected_scl = ((sim_tv - alpha)**2 + (sim_ta - alpha)**2 + (sim_va - alpha)**2).mean() / 3.0
                        print(f"  预期SCL损失: {expected_scl:.4f}")
            # ===== Focal Loss（辅助分类）=====
            L_focal = torch.tensor(0.0, device=device)
            if cls_logits is not None:
                binary_labels = (label.view(-1) > 0).float()
                fa = getattr(model.config.SIMS.downStream, 'focal_alpha', None)
                if fa is None:
                    pos = (binary_labels == 1).sum().item()
                    neg = (binary_labels == 0).sum().item()
                    tot = max(pos + neg, 1)
                    alpha_pos = neg / tot
                else:
                    alpha_pos = float(fa)
                gamma = getattr(model.config.SIMS.downStream, 'focal_gamma', 2.0)
                focal_w = getattr(model.config.SIMS.downStream, 'focal_weight', 0.1)
                L_focal = focal_bce_with_logits(cls_logits, binary_labels, alpha=alpha_pos, gamma=gamma)
                loss = loss + focal_w * L_focal

                # 调试SCL：前几个batch打印详细信息
                debug_scl = (batch_count < 3 and epoch == 1)
                L_scl = compute_scl_loss(x_t, x_v, x_a, alpha, debug=debug_scl) if lambda1 != 0 else torch.tensor(0.0, device=device)
                L_iamcl = compute_iamcl_loss(x_t, x_v, x_a, cls_labels) if lambda2 != 0 else torch.tensor(0.0, device=device)
                L_iemcl = compute_iemcl_loss(x_t, x_v, x_a, cls_labels, alpha) if lambda3 != 0 else torch.tensor(0.0, device=device)

                L_hybrid = lambda1 * L_scl + lambda2 * L_iamcl + lambda3 * L_iemcl
                loss = loss + L_hybrid



            # 添加AtCAF损失（如果启用）
            if torch.is_tensor(atcaf_loss_raw) and atcaf_loss_raw.item() > 0:
                counterfactual_weight = model.config.SIMS.downStream.counterfactual_weight

                # warm-up: 前 atcaf_warmup_epochs 轮，将权重线性从 0 -> counterfactual_weight
                warmup_epochs = getattr(model.config.SIMS.downStream, 'atcaf_warmup_epochs', 0)
                if warmup_epochs and epoch <= warmup_epochs:
                    scale = float(epoch) / float(warmup_epochs)
                    eff_weight = counterfactual_weight * scale
                else:
                    eff_weight = counterfactual_weight

                atcaf_loss_weighted = eff_weight * atcaf_loss_raw
                loss = loss + atcaf_loss_weighted
            else:
                atcaf_loss_raw = torch.tensor(0.0, device=device, dtype=torch.float32)







            # 累积损失统计 (使用原始AtCAF损失进行记录)
            running_pred_loss += pred_loss.item()
            running_atcaf_loss += atcaf_loss_raw.item() # 原始AtCAF
            running_atcaf_weighted_loss += atcaf_loss_weighted.item() # 加权AtCAF（用于对账）
            running_total_loss += loss.item()
            batch_count += 1

            loss.backward()

            if not left_epochs:
                optimizer.step()
                left_epochs = update_epochs

        if not left_epochs:
            optimizer.step()





        # 每轮训练后评估所有数据集并打印结果
        train_results, _ = eval(model, metrics, train_data, device)
        valid_results, valid_loss = eval(model, metrics, valid_data, device)
        test_results, _ = eval(model, metrics, test_data, device)

        # 打印完整的评估结果
        print(f"\n📊 Epoch {epoch}/{total_epoch} 完整评估结果:")
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

        print("=" * 80)

        # 检查并保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

            # 删除旧的最佳模型（如果存在）
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            # 保存新的最佳模型
            model_name = f'best_model_epoch_{epoch}_loss_{valid_loss:.4f}.pt'
            best_model_path = os.path.join(model.model_path, model_name)
            torch.save(model.state_dict(), best_model_path)

            print(f"\n⭐ 新的最佳模型已保存! Epoch: {epoch}, 最佳验证损失: {valid_loss:.4f}")
            print(f"   模型路径: {best_model_path}")

    # ========== 训练结束，加载最佳模型进行最终评估 ==========
    print("\n" + "="*50)
    print("🏁 训练完成! 正在加载最佳模型进行最终评估...")
    print("="*50)

    if best_model_path and os.path.exists(best_model_path):
        print(f"🔄 正在加载在第 {best_epoch} 个epoch找到的最佳模型...")
        print(f"   最佳验证损失: {best_valid_loss:.4f}")
        print(f"   模型路径: {best_model_path}")

        # 加载最佳模型的状态字典
        model.load_state_dict(torch.load(best_model_path))

        print("\n🧪 使用最佳模型在测试集上进行最终评估...")
        final_test_results, _ = eval(model, metrics, test_data, device)

        print("\n" + "="*50)
        print("🎉 基于最佳验证损失的最终测试结果 🎉")
        print(f"📊 最佳验证损失: {best_valid_loss:.4f} (第 {best_epoch} 个epoch)")
        print(f"  - MAE: {final_test_results['MAE']:.4f}")
        print(f"  - Corr: {final_test_results['Corr']:.4f}")
        print(f"  - Accuracy-7: {final_test_results['Mult_acc_7']:.4f}")
        print(f"  - Accuracy-2 (Has0): {final_test_results['Has0_acc_2']:.4f}")
        print(f"  - F1-Score (Has0): {final_test_results['Has0_F1_score']:.4f}")
        print(f"  - Accuracy-2 (Non0): {final_test_results['Non0_acc_2']:.4f}")
        print(f"  - F1-Score (Non0): {final_test_results['Non0_F1_score']:.4f}")
        print("="*50)
    else:
        print("❌ 未找到可用的最佳模型，无法进行最终测试。")


def eval(model, metrics, data, device):
    model.eval()
    with torch.no_grad():
        pred, truth = [], []
        loss = 0
        lens = 0
        for batch_data in data:
            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().view(-1).to(device).float()
            _pred, _ = model(text, vision, audio, mode='test')
            pred.append(_pred.view(-1))
            truth.append(label)
            _loss = torch.mean((label-_pred)*(label-_pred))
            loss += _loss.item() * len(label)
            lens += len(label)
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu')).squeeze()
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / lens
    model.train()
    return eval_results, loss / lens


def TVA_test_fusion(config, metric, test_data):

    seed = config.seed
    log_path = config.LOGPATH + "MOSI_TVA_Test." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'

    write_config(config, log_path)

    model = TVA_fusion(config=config)

    device = config.DEVICE
    model.to(device)

    model.load_model()
    result, loss = eval(model,metric, test_data, device)

    log = '\nTVA_Test\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2:%s\n\t' \
        'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
        '------------------------------------------' % (
        result['Has0_acc_2'], result['Has0_F1_score'], result['Non0_acc_2'], result['Non0_F1_score'],
        result['Mult_acc_5'], result['Mult_acc_7'], result['MAE'], result['Corr'], loss
    )
    print(log)
    write_log(log, log_path)

    return result

