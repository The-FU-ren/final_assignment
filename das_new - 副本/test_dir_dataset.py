# 测试目录数据集加载

print("测试目录数据集加载...")

try:
    from data_loader import DASDataset, get_data_loaders
    
    # 数据集配置
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    batch_size = 16
    max_samples_per_file = 50
    
    # 测试数据集初始化
    print(f"\n1. 初始化数据集: {data_dir}")
    dataset = DASDataset(
        data_dir=data_dir,
        max_samples_per_file=max_samples_per_file
    )
    
    print(f"   类别: {dataset.label_encoder.classes_}")
    print(f"   类别数量: {len(dataset.label_encoder.classes_)}")
    print(f"   样本数量: {len(dataset)}")
    print(f"   类别分布: {dataset.class_distribution}")
    
    # 测试单个样本加载
    print("\n2. 测试单个样本加载...")
    signal, label = dataset[0]
    print(f"   样本形状: {signal.shape}")
    print(f"   标签: {label}")
    print(f"   类别名称: {dataset.label_encoder.inverse_transform([label])[0]}")
    
    # 测试数据加载器创建
    print("\n3. 测试数据加载器创建...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        max_samples_per_file=max_samples_per_file
    )
    
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    
    # 测试数据加载
    print("\n4. 测试数据加载...")
    data_iter = iter(train_loader)
    signals, labels = next(data_iter)
    print(f"   批次信号形状: {signals.shape}")
    print(f"   批次标签形状: {labels.shape}")
    print(f"   标签值: {labels[:5]}")
    print(f"   类别名称: {dataset.label_encoder.inverse_transform(labels[:5].numpy())}")
    
    print("\n所有测试通过!")
    print("目录数据集加载正常工作")
    
except Exception as e:
    import traceback
    print(f"\n测试失败: {e}")
    traceback.print_exc()

print("\n测试完成")
