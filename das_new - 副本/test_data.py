# 简单的测试脚本，只测试数据加载

print("测试数据加载...")

try:
    from data_loader_h5 import get_data_loaders_from_h5
    import torch
    
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders_from_h5(
        "G:/gitcode/das_small_dataset_N50_20241230_154821.h5",
        batch_size=16
    )
    
    print(f"训练加载器批次: {len(train_loader)}")
    
    # 尝试获取第一个批次
    data_iter = iter(train_loader)
    signals, labels = next(data_iter)
    
    print(f"\n批次信息:")
    print(f"信号形状: {signals.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签值: {labels}")
    print(f"标签范围: [{labels.min()}, {labels.max()}]")
    
    print("\n数据加载测试成功!")
    
except Exception as e:
    import traceback
    print(f"\n数据加载测试失败: {e}")
    traceback.print_exc()

print("\n测试结束")
