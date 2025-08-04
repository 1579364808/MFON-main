"""
定制化损失权重搜索脚本 - theory_based

基于理论分析的搜索空间
"""

# 搜索空间配置
SEARCH_SPACE = {
    'delta_va': [2.0, 2.5, 3.0, 3.5, 4.0],
    'delta_nce': [0.03, 0.05, 0.07, 0.1, 0.15]
}

# 使用方法:
# 1. 将此配置复制到 loss_weight_grid_search.py 中的 search_space
# 2. 运行: python loss_weight_grid_search.py

print("🔍 theory_based 搜索配置:")
print(f"delta_va 候选值: {SEARCH_SPACE['delta_va']}")
print(f"delta_nce 候选值: {SEARCH_SPACE['delta_nce']}")
print(f"总组合数: {len(SEARCH_SPACE['delta_va']) * len(SEARCH_SPACE['delta_nce'])}")
