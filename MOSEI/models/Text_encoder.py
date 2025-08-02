from transformers import BertModel, BertTokenizer
from torch import nn
from models.classifier import BaseClassifier
import torch
import os


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class TextEncoder(nn.Module):
    """
    文本编码器模块

    该模块基于预训练的BERT模型，用于将原始文本编码为高级语义表示，主要特点：
    1. 多语言支持：支持英文(MOSEI)和中文(SIMS)数据集
    2. 预训练优势：利用BERT的预训练知识理解文本语义
    3. 统一接口：为不同语言提供统一的编码接口
    4. 序列输出：返回完整的序列表示，支持后续的注意力机制

    架构特点：
    - 基于Transformer的双向编码器
    - 自注意力机制捕获长距离依赖
    - 预训练权重提供丰富的语言理解能力
    - 输出768维的语义表示（与音频、视觉统一）
    """

    def __init__(self, config, name=None):
        """
        初始化文本编码器

        Args:
            config: 配置对象，包含语言设置和设备配置
            name: 编码器名称标识，用于调试和日志记录
        """
        super(TextEncoder, self).__init__()
        self.name = name

        # 根据配置的语言类型选择对应的BERT模型
        language = config.MOSEI.downStream.language

        if language == 'en':
            # 英文BERT模型 - 用于MOSEI等英文数据集
            # bert-base-uncased: 12层Transformer，768维隐藏层，12个注意力头
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.extractor = BertModel.from_pretrained('bert-base-uncased')

        elif language == 'cn':
            # 中文BERT模型 - 用于SIMS等中文数据集
            # bert-base-chinese: 专门为中文优化的BERT模型
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.extractor = BertModel.from_pretrained('bert-base-chinese')

        # 设备配置，支持CPU/GPU计算
        self.device = config.DEVICE

    def forward(self, text):
        """
        前向传播

        Args:
            text: 输入文本列表，例如：
                 ["I love this movie", "This is terrible", ...]
                 每个元素是一个字符串

        Returns:
            last_hidden_state: BERT的最后一层隐藏状态 [batch_size, seq_len, 768]
                              包含每个token的上下文化表示

        处理流程：
        1. 文本分词：将原始文本转换为token序列
        2. 添加特殊token：[CLS], [SEP]等
        3. BERT编码：通过12层Transformer获得上下文表示
        4. 返回序列表示：保留完整的序列信息用于后续处理
        """

        # 步骤1: 文本预处理和分词
        # padding=True: 将batch内的序列填充到相同长度
        # truncation=True: 截断超过最大长度的序列
        # max_length=256: 设置最大序列长度为256个token
        # return_tensors="pt": 返回PyTorch张量格式
        x = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        # 步骤2: BERT编码
        # 输入包含：input_ids, attention_mask, token_type_ids
        # 输出包含：last_hidden_state, pooler_output, hidden_states等
        x = self.extractor(**x)

        # 步骤3: 提取序列表示
        # last_hidden_state: [batch_size, seq_len, 768]
        # 包含每个token位置的768维上下文化表示
        last_hidden_state = x['last_hidden_state']

        return last_hidden_state  # [bs, seq, 768]
        
        
class TextPretrain(nn.Module):
    """
    文本预训练模型

    该模型用于文本模态的预训练，虽然使用了预训练的BERT，但仍需要在特定任务上进行微调。
    主要目标是：
    1. 任务适应：让BERT适应情感回归任务的特定需求
    2. 特征对齐：与音频、视觉模态在相同的情感空间中对齐
    3. 融合准备：为后续的多模态融合提供高质量的文本表示
    4. 知识蒸馏：在融合训练中作为教师模型指导学习

    架构组成：
    - TextEncoder: 基于BERT的文本编码器，输出768维序列表示
    - BaseClassifier: 回归分类器，将文本表示映射到情感预测值

    训练策略：
    - 使用MOSEI数据集的情感标签进行监督学习
    - 通过MSE损失进行回归训练
    - 微调BERT参数以适应情感分析任务
    """

    def __init__(self, config, encoder_fea_dim=None):
        """
        初始化文本预训练模型

        Args:
            config: 配置对象，包含所有超参数设置
            encoder_fea_dim: 编码器特征维度，默认为None时使用配置文件中的值(768)
        """
        super(TextPretrain, self).__init__()

        # 设置编码器特征维度，默认使用配置文件中的768维
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim  # 768

        # ========== 核心组件初始化 ==========
        # 文本编码器：基于预训练BERT，输出768维序列表示
        self.encoder = TextEncoder(config)

        # 回归分类器：将文本表示映射到情感预测值
        # 网络结构：768 → 384 → 96 → 1
        # 注意：这里需要处理序列输出，通常取[CLS]位置或进行池化
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,                                    # 768: 输入维度
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],  # [384, 96]: 隐藏层维度
            output_size=1,                                                 # 1: 输出情感回归值
            name='TextClassifier'                                          # 分类器名称标识
        )

        # ========== 训练相关配置 ==========
        self.device = config.DEVICE                    # 设备配置(CPU/GPU)
        self.criterion = torch.nn.MSELoss()           # MSE损失函数，用于回归任务
        self.config = config                          # 保存配置对象

        # 模型保存路径：根据随机种子创建独立的保存目录
        self.model_path = config.MOSEI.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)  # 确保目录存在

    def forward(self, text, label, return_loss=True, device=None):
        """
        前向传播

        Args:
            text: 输入文本列表，例如：
                 ["I love this movie", "This is terrible", ...]
            label: 情感标签 [batch_size] 或 [batch_size, 1]
                  MOSEI数据集的回归标签，范围通常在[-3, 3]
            return_loss: 是否返回损失值，训练时为True，推理时为False
            device: 计算设备，默认使用初始化时的设备配置

        Returns:
            如果return_loss=True: (pred, loss)
                pred: 情感预测值 [batch_size]
                loss: MSE损失标量值
            如果return_loss=False: pred
                pred: 情感预测值 [batch_size]

        处理流程：
        1. 文本编码：通过BERT获得序列的上下文表示
        2. 序列池化：将序列表示转换为句子级表示（通常取[CLS]位置）
        3. 情感预测：通过分类器将句子表示映射到情感值
        4. 损失计算：如果需要，计算预测值与真实标签的MSE损失

        注意：这里的实现可能需要处理BERT输出的序列维度问题
        """
        # 设备配置检查
        if device is None:
            device = self.device

        # 步骤1: 文本编码 - 通过BERT获得序列的上下文表示
        # 输入: 文本列表 → 输出: [bs, seq_len, 768]
        x = self.encoder(text)

        # 步骤2: 序列池化 + 情感预测
        # 注意：这里直接将序列输出传给分类器可能有维度问题
        # 通常需要取[CLS]位置 (x[:, 0, :]) 或进行平均池化
        # 当前实现可能需要在分类器内部处理序列维度
        pred = self.classifier(x).squeeze()

        # 步骤3: 损失计算（可选）
        if return_loss:
            # 计算预测值与真实标签之间的MSE损失
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, loss
        else:
            # 推理模式，只返回预测值
            return pred

    def save_model(self, name='best_loss'):
        """
        保存模型权重

        将文本编码器和分类器的权重分别保存，这样设计的优势：
        1. 模块化保存：可以单独加载BERT编码器用于融合模型
        2. 灵活性：可以只加载需要的部分（编码器或分类器）
        3. 复用性：微调后的BERT可以在不同任务中复用
        4. 存储效率：避免保存整个模型的冗余信息

        Args:
            name: 保存文件的前缀名，默认为'best_loss'

        保存文件：
        - 文本编码器权重：{model_path}/{name}_text_encoder.pt (包含微调后的BERT)
        - 分类器权重：{model_path}/{name}_text_decoder.pt
        """
        # 构建保存路径
        encoder_path = self.model_path + name + '_text_encoder.pt'
        decoder_path = self.model_path + name + '_text_decoder.pt'

        # 分别保存编码器和分类器的状态字典
        torch.save(self.encoder.state_dict(), encoder_path)      # 保存微调后的BERT权重
        torch.save(self.classifier.state_dict(), decoder_path)  # 保存回归分类器权重

        # 打印保存路径信息
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        """
        加载模型权重

        支持灵活的模型加载策略，特别适用于文本模态的不同使用场景：
        1. 只加载编码器：在TVA融合模型中使用微调后的BERT
        2. 只加载分类器：在特定分类任务中复用分类器
        3. 加载完整模型：用于继续训练或完整推理

        Args:
            name: 加载文件的前缀名，默认为'best_loss'
            module: 指定加载的模块，可选值：
                   - 'encoder': 只加载文本编码器（微调后的BERT）
                   - 'decoder': 只加载回归分类器
                   - 'all' 或 None: 加载完整模型（默认）

        使用场景：
        - module='encoder': 在TVA融合模型中初始化文本编码器
        - module='decoder': 在其他文本分类任务中复用分类器
        - module='all': 完整的文本预训练模型推理或继续训练
        """
        # 构建加载路径
        encoder_path = self.model_path + name + '_text_encoder.pt'
        decoder_path = self.model_path + name + '_text_decoder.pt'
        print('model loaded from:')

        # 根据module参数选择性加载
        if module == 'encoder':
            # 只加载文本编码器权重（包含微调后的BERT参数）
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)

        elif module == 'decoder':
            # 只加载回归分类器权重
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)

        elif module == 'all' or module is None:
            # 加载完整模型（编码器 + 分类器）
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)


