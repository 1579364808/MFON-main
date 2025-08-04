"""
定制化损失权重搜索脚本 - coarse_search

粗粒度搜索，覆盖广泛范围
"""

# 搜索空间配置
SEARCH_SPACE = {
    'delta_va': [0.1, 0.3, 0.5, 1.0, 2.0],
    'delta_nce': [0.001, 0.01, 0.05, 0.1, 0.2]
}

# 使用方法:
# 1. 将此配置复制到 loss_weight_grid_search.py 中的 search_space
# 2. 运行: python loss_weight_grid_search.py

print("🔍 coarse_search 搜索配置:")
print(f"delta_va 候选值: {SEARCH_SPACE['delta_va']}")
print(f"delta_nce 候选值: {SEARCH_SPACE['delta_nce']}")
print(f"总组合数: {len(SEARCH_SPACE['delta_va']) * len(SEARCH_SPACE['delta_nce'])}")
