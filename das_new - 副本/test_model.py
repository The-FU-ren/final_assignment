# 简单的模型测试脚本

import torch
from model import DRSN_NTF

print("测试模型基本功能...")

try:
    # 初始化模型
    model = DRSN_NTF(num_classes=4)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建随机输入
    input_data = torch.randn(2, 1, 8000, 1)  # 批次大小2，1通道，8000宽度，1高度
    print(f"输入形状: {input_data.shape}")
    
    # 前向传播
    outputs = model(input_data)
    print(f"输出形状: {outputs.shape}")
    print(f"输出样本: {outputs}")
    
    # 测试损失计算
    labels = torch.tensor([0, 1])  # 两个样本，标签分别为0和1
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(f"损失值: {loss.item()}")
    
    # 测试反向传播
    loss.backward()
    print("反向传播成功")
    
    print("\n模型测试成功!")
    print("所有基本功能正常工作")
    
except Exception as e:
    import traceback
    print(f"\n模型测试失败: {e}")
    traceback.print_exc()

print("\n测试完成")
