import os
import sys
# 确保当前目录在 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import config
from train.TVA_train import TVA_train_fusion, TVA_test_fusion
# 删除预训练相关导入 - 不再需要单模态预训练
from utils import Metrics
from data_loader import MOSIDataloader


def main():
    batch_size=config.MOSI.downStream.batch_size
    train_data = MOSIDataloader('train', config.MOSI.path.raw_data_path, batch_size=batch_size)
    valid_data = MOSIDataloader('valid', config.MOSI.path.raw_data_path, batch_size=batch_size, shuffle=False, )
    test_data = MOSIDataloader('test', config.MOSI.path.raw_data_path,batch_size=batch_size, shuffle=False, )
    metrics = Metrics()
    
    # config.seed = 1111  

    # Vtrain(config, metrics, config.seed, train_data, valid_data)
    # Vtest(config, metrics, test_data)

    # Atrain(config, metrics, config.seed, train_data, valid_data)
    # Atest(config, metrics, test_data)
    
    TVA_train_fusion(config, metrics, config.seed, train_data, valid_data, test_data)
    # TVA_test_fusion(config, metrics, test_data)  # 不再需要单独测试，训练中已包含
                 

if __name__ == '__main__':
    main()
