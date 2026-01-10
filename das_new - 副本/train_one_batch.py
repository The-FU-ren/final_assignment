# 只训练一个批次的简单脚本

print("="*60)
print("只训练一个批次")
print("="*60)

try:
    import torch
    from model import DRSN_NTF
    from data_loader_fixed import get_data_loaders
    
    # 配置
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    batch_size = 4
    max_samples_per_file = 5
    learning_rate = 0.001
    
    print("1. 加载数据...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        max_samples_per_file=max_samples_per_file
    )
    
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    
    # 初始化模型
    print("\n2. 初始化模型...")
    model = DRSN_NTF(num_classes=9)
    print(f"   模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 初始化优化器和损失函数
    print("\n3. 初始化优化器和损失函数...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 获取第一个批次
    print("\n4. 获取第一个批次...")
    data_iter = iter(train_loader)
    signals, labels = next(data_iter)
    print(f"   信号形状: {signals.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   标签值: {labels}")
    
    # 训练一个批次
    print("\n5. 训练一个批次...")
    
    # 前向传播
    model.train()
    outputs = model(signals)
    print(f"   输出形状: {outputs.shape}")
    
    # 计算损失
    loss = criterion(outputs, labels)
    print(f"   损失值: {loss.item()}")
    
    # 反向传播
    print("   反向传播中...")
    optimizer.zero_grad()
    loss.backward()
    
    # 优化器步进
    optimizer.step()
    print("   优化器步进完成")
    
    print("\n" + "="*60)
    print("训练成功!")
    print("模型能够成功训练一个批次")
    print("="*60)
    
except Exception as e:
    import traceback
    print(f"\n训练失败: {e}")
    traceback.print_exc()
    
print("\n测试完成")
