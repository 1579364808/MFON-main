import torch
import datetime
from tqdm import tqdm
from utils import write_log, set_random_seed
from models.model import TVA_fusion
from utils import write_config


def TVA_train_fusion(config, metrics, seed, train_data, valid_data, test_data):
    print('---------------TVA_EXP---------------')
    
    set_random_seed(seed)
    
    update_epochs = config.MOSEI.downStream.update_epochs
    
    text_lr = config.MOSEI.downStream.TVAtrain.text_lr
    audio_lr = config.MOSEI.downStream.TVAtrain.audio_lr
    vision_lr = config.MOSEI.downStream.TVAtrain.vision_lr
    other_lr = config.MOSEI.downStream.TVAtrain.other_lr
    
    text_decay = config.MOSEI.downStream.TVAtrain.text_decay
    audio_decay = config.MOSEI.downStream.TVAtrain.audio_decay
    vision_decay = config.MOSEI.downStream.TVAtrain.vision_decay
    other_decay = config.MOSEI.downStream.TVAtrain.other_decay
            
    # 删除知识蒸馏和对比学习权重 - 不再使用
    
    model = TVA_fusion(config).to(config.DEVICE)
    print(model)
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

    optimizer_grouped_parameters = [
        {'params': text_params, 'weight_decay': text_decay, 'lr': text_lr},
        {'params': audio_params, 'weight_decay': audio_decay, 'lr': audio_lr},
        {'params': vision_params, 'weight_decay': vision_decay, 'lr': vision_lr},
        {'params': model_params_other, 'weight_decay': other_decay, 'lr': other_lr}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
   
    loss, best_loss  = 0, 1e8
    pred_loss = 0  # 只保留预测损失
    device = config.DEVICE  
    total_epoch = config.MOSEI.downStream.TVAtrain.epoch
    best_epoch = 1
    for epoch in range(1, total_epoch + 1):
        
        model.train()
        left_epochs = update_epochs
        bar = tqdm(train_data, disable=False)
        for batch_data in bar:
            try:
                bar.set_description("Epoch:%d|Loss:[%.4f]|" % (epoch, loss.item()))
            except:
                bar.set_description("Epoch:%d|Loss:[%.4f]|" % (epoch, loss))
            if left_epochs == update_epochs:
                optimizer.zero_grad()
            left_epochs -= 1

            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().to(device)
        
            pred, _ = model(text, vision, audio, mode='train')

            pred_loss = torch.mean((pred-label)*(pred-label))  # [bs]

            loss = pred_loss  # 只使用预测损失
            
            loss.backward()
           
            if not left_epochs:
                optimizer.step()
                left_epochs = update_epochs
                
        if not left_epochs:
            optimizer.step()

        # 每轮训练后评估所有数据集并打印结果
        train_results, train_loss = eval(model, metrics, train_data, device)
        valid_results, valid_loss = eval(model, metrics, valid_data, device)
        test_results, test_loss = eval(model, metrics, test_data, device)

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

        # 基于验证集损失保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            model.save_model()
            print(f"\n🎉 新的最佳模型已保存! (验证集 Best Loss: {best_loss:.6f})")
        else:
            print(f"\n📈 当前最佳: 验证集 Loss: {best_loss:.6f}")
        print("=" * 80)
           

def eval(model, metrics, eval_data, device):
    model.eval()
    with torch.no_grad():
        pred, truth = [], []
        loss = 0
        lens = 0 
        for index, batch_data in enumerate(eval_data):
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
    log_path = config.LOGPATH + "MOSEI_TVA_Test." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'

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