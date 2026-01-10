# 简单的测试脚本

print("开始简单测试...")

# 导入必要的模块
print("导入模块...")

try:
    import torch
    import numpy as np
    import os
    from model import DRSN_NTF
    
    print("模块导入成功")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    # 测试模型创建
    print("\n测试模型创建...")
    model = DRSN_NTF(num_classes=4)
    print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试数据加载器创建
    print("\n测试数据加载器创建...")
    from data_loader_h5 import SingleH5DASDataset
    dataset = SingleH5DASDataset(
        "G:/gitcode/das_small_dataset_N50_20241230_154821.h5",
        split='train'
    )
    print(f"数据集创建成功，样本数量: {len(dataset)}")
    print(f"第一个样本名称: {dataset.sample_info[0]['sample_name']}")
    
    # 测试单个样本加载
    print("\n测试单个样本加载...")
    signal, label = dataset._load_sample(0)
    print(f"样本加载成功，信号形状: {signal.shape}, 标签: {label}")
    
    print("\n所有测试通过!")
    
except Exception as e:
    import traceback
    print(f"\n测试失败: {e}")
    traceback.print_exc()

print("\n测试完成")
