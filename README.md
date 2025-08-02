Pytorch implementation for Coling2025 paper: Modal Feature Optimization Network with Prompt for Multimodal Sentiment Analysis.

![Image text](https://github.com/123sprouting/MFON/blob/main/MFON-structure.jpg)



The training and testing of the MOSI, MOSEI, and SIMS datasets are implemented in three files with the same name.

### Setup the environment

We work with a conda environment.

```
conda env create -f environment.yaml
conda activate pytorch
```

### Data Download

- Get datasets from public link:https://github.com/thuiar/Self-MM and  change the raw_data_path  to your local path(In config.py).
### Pretrained model:
链接:https://pan.baidu.com/s/1_EcyiHYXtSFTjG5Ro7E1SQ 
提取码:2xx5

### Running the code

Take MOSI for example:
1. cd MOSI
2. python main.py 
