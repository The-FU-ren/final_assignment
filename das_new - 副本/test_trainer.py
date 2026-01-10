import torch
from model import DRSN_NTF
from trainer import Trainer
from data_loader_h5 import get_data_loaders_from_h5

# 测试训练器
print("测试训练器初始化")

# 初始化模型
model = DRSN_NTF(num_classes=4)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 初始化训练器
trainer = Trainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
print("训练器初始化成功")

# 加载数据
train_loader, val_loader, test_loader = get_data_loaders_from_h5(
    "G:/gitcode/das_small_dataset_N50_20241230_154821.h5",
    batch_size=16
)
print("数据加载成功")

# 测试训练一个批次
print("测试训练一个批次")
try:
    # 获取第一个批次
    data_iter = iter(train_loader)
    signals, labels = next(data_iter)
    print(f"批次信号形状: {signals.shape}")
    print(f"批次标签形状: {labels.shape}")
    print(f"标签值: {labels[:5]}")
    print(f"标签最大值: {labels.max()}, 最小值: {labels.min()}")
    
    # 测试前向传播
    print("测试前向传播...")
    outputs = model(signals)
    print(f"输出形状: {outputs.shape}")
    print(f"输出样本: {outputs[:2]}")
    
    # 测试损失计算
    print("测试损失计算...")
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    print(f"损失值: {loss.item()}")
    
    print("批次测试成功")
except Exception as e:
    import traceback
    print(f"批次测试失败: {e}")
    traceback.print_exc()

print("测试完成")
