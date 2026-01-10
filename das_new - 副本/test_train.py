# 简单的训练测试脚本

import torch
from model import DRSN_NTF
from trainer import Trainer
from data_loader_h5 import SingleH5DASDataset
from torch.utils.data import DataLoader

def main():
    print("开始测试训练...")
    
    # 初始化模型
    model = DRSN_NTF(num_classes=4)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 初始化训练器
    trainer = Trainer(model, device='cpu')
    print("训练器初始化成功")
    
    # 加载数据集
    dataset = SingleH5DASDataset(
        "G:/gitcode/das_small_dataset_N50_20241230_154821.h5",
        split='train'
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    print(f"数据加载器批次: {len(data_loader)}")
    
    # 尝试训练一个批次
    print("\n测试训练一个批次...")
    try:
        # 获取第一个批次
        data_iter = iter(data_loader)
        signals, labels = next(data_iter)
        print(f"批次信号形状: {signals.shape}")
        print(f"批次标签形状: {labels.shape}")
        print(f"标签值: {labels}")
        
        # 移动到设备
        signals = signals.to(trainer.device)
        labels = labels.to(trainer.device)
        
        # 前向传播
        outputs = trainer.model(signals)
        print(f"输出形状: {outputs.shape}")
        
        # 损失计算
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        print(f"损失值: {loss.item()}")
        
        # 反向传播
        loss.backward()
        trainer.optimizer.step()
        print("反向传播和优化成功")
        
        # 清零梯度
        trainer.optimizer.zero_grad()
        
    except Exception as e:
        import traceback
        print(f"批次训练失败: {e}")
        traceback.print_exc()
    
    # 尝试训练一轮
    print("\n测试训练一轮...")
    try:
        train_loss, train_accuracy = trainer.train_one_epoch(data_loader, epoch=0, total_epochs=1)
        print(f"训练一轮成功! 损失: {train_loss:.3f}, 准确率: {train_accuracy:.2f}%")
    except Exception as e:
        import traceback
        print(f"训练一轮失败: {e}")
        traceback.print_exc()
    
    print("\n测试完成")

if __name__ == "__main__":
    main()
