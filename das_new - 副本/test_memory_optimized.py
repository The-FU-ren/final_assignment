import os
import torch
import numpy as np
from data_loader import DASDataset, get_data_loaders

# 测试内存优化后的DASDataset类
def test_memory_optimized_dataset():
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    
    print("=== 测试内存优化后的DASDataset类 ===")
    
    # 创建数据集，限制每个文件的样本数
    dataset = DASDataset(
        data_dir=data_dir,
        max_samples_per_file=50  # 每个文件最多50个样本，减少内存使用
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别分布: {dataset.class_distribution}")
    
    # 测试样本加载
    print("\n=== 测试样本加载 ===")
    for i in range(5):
        signal, label = dataset[i]
        print(f"样本 {i}: 信号形状={signal.shape}, 标签={label}")
    
    # 测试数据加载器
    print("\n=== 测试数据加载器 ===")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        max_samples_per_file=50,
        val_split=0.1,
        test_split=0.1
    )
    
    print(f"训练加载器批次: {len(train_loader)}")
    print(f"验证加载器批次: {len(val_loader)}")
    print(f"测试加载器批次: {len(test_loader)}")
    
    # 测试训练循环
    print("\n=== 测试训练循环 ===")
    for i, (signals, labels) in enumerate(train_loader):
        print(f"批次 {i}: 信号形状={signals.shape}, 标签形状={labels.shape}")
        if i >= 2:  # 只测试3个批次
            break
    
    print("\n=== 测试完成 ===")
    print("内存优化后的DASDataset类工作正常！")

if __name__ == "__main__":
    test_memory_optimized_dataset()
