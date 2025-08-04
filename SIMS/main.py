import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)

import config
# 删除预训练相关导入 - 不再需要单模态预训练
from train.TVA_train import TVA_train_fusion, TVA_test_fusion
from utils import Metrics
from data_loader import SIMSDataloader


def main():
    batch_size=config.SIMS.downStream.batch_size
    train_data = SIMSDataloader('train', config.SIMS.path.raw_data_path, batch_size=batch_size)
    valid_data = SIMSDataloader('valid', config.SIMS.path.raw_data_path, batch_size=batch_size, shuffle=False, )
    test_data = SIMSDataloader('test', config.SIMS.path.raw_data_path,batch_size=batch_size, shuffle=False, )
    metrics = Metrics()
    
    config.seed = 1111
    
    #Vtrain(config, metrics, config.seed, train_data, valid_data)
    #Vtest(config, metrics, test_data)

    #Atrain(config, metrics, config.seed, train_data, valid_data)
    #Atest(config, metrics, test_data)

    TVA_train_fusion(config, metrics, config.seed, train_data, valid_data, test_data)
    TVA_test_fusion(config, metrics, test_data)  # 不再需要单独测试，训练中已包含
              

if __name__ == '__main__':
    main()
