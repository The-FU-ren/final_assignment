# 测试修复后的数据加载器

print("测试修复后的数据加载器...")

try:
    import sys
    import os
    
    # 添加当前目录到Python路径
    sys.path.append(os.getcwd())
    
    from data_loader_fixed import get_data_loaders
    
    # 测试数据加载器
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    batch_size = 8
    max_samples_per_file = 5
    
    print(f"\n1. 加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        max_samples_per_file=max_samples_per_file
    )
    
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    
    # 测试第一个批次
    print("\n2. 测试第一个批次...")
    data_iter = iter(train_loader)
    signals, labels = next(data_iter)
    
    print(f"   信号形状: {signals.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   标签值: {labels}")
    
    print("\n" + "="*60)
    print("测试成功!")
    print("修复后的数据加载器正常工作")
    print("="*60)
    
except Exception as e:
    import traceback
    print(f"\n测试失败: {e}")
    traceback.print_exc()
    
print("\n测试完成")
