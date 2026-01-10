# 最小化测试脚本，只使用随机输入

import torch
from model import DRSN_NTF

print("="*60)
print("最小化测试脚本")
print("="*60)

# 创建随机输入
batch_size = 2
input_shape = (batch_size, 1, 8000, 1)  # 匹配模型输入
labels = torch.tensor([0, 1])  # 随机标签

print(f"1. 随机输入形状: {input_shape}")
print(f"2. 随机标签: {labels}")

# 初始化模型
model = DRSN_NTF(num_classes=9)
print(f"3. 模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 前向传播
print("\n4. 测试前向传播...")
model.eval()
with torch.no_grad():
    outputs = model(torch.randn(input_shape))
print(f"   输出形状: {outputs.shape}")
print(f"   输出样本: {outputs}")

# 损失计算
print("\n5. 测试损失计算...")
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(outputs, labels)
print(f"   损失值: {loss.item()}")

# 单批次训练
print("\n6. 测试单批次训练...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 清零梯度
optimizer.zero_grad()

# 前向传播
outputs = model(torch.randn(input_shape))
loss = criterion(outputs, labels)
print(f"   训练损失: {loss.item()}")

# 反向传播
print("   反向传播中...")
loss.backward()
print("   反向传播完成")

# 优化器步进
optimizer.step()
print("   优化器步进完成")

print("\n" + "="*60)
print("最小化测试完成!")
print("所有功能正常工作")
print("="*60)
