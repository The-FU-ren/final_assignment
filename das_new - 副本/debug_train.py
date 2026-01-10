# 调试训练脚本，只测试单个批次

import torch
import os
from model import DRSN_NTF
from trainer import Trainer
from data_loader import get_data_loaders

def main():
    # 训练配置
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    batch_size = 4
    max_samples_per_file = 10  # 只使用少量样本
    device = "cpu"
    
    print("="*60)
    print("调试训练脚本")
    print("="*60)
    
    try:
        # 加载数据
        print("1. 加载数据...")
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            max_samples_per_file=max_samples_per_file
        )
        
        print(f"   训练批次: {len(train_loader)}")
        print(f"   验证批次: {len(val_loader)}")
        print(f"   测试批次: {len(test_loader)}")
        
        # 获取第一个批次
        print("\n2. 获取第一个批次...")
        data_iter = iter(train_loader)
        signals, labels = next(data_iter)
        
        print(f"   信号形状: {signals.shape}")
        print(f"   标签形状: {labels.shape}")
        print(f"   标签值: {labels}")
        
        # 初始化模型
        print("\n3. 初始化模型...")
        num_classes = 9
        model = DRSN_NTF(num_classes=num_classes)
        print(f"   模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 前向传播
        print("\n4. 测试前向传播...")
        model.eval()
        with torch.no_grad():
            outputs = model(signals)
        print(f"   输出形状: {outputs.shape}")
        print(f"   输出样本: {outputs[:2]}")
        
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
        outputs = model(signals)
        loss = criterion(outputs, labels)
        print(f"   训练损失: {loss.item()}")
        
        # 反向传播
        print("   反向传播中...")
        loss.backward()
        
        # 优化器步进
        optimizer.step()
        print("   优化器步进完成")
        
        print("\n" + "="*60)
        print("调试完成!")
        print("所有基本功能正常工作")
        print("="*60)
        
    except Exception as e:
        import traceback
        print(f"\n调试失败: {e}")
        traceback.print_exc()
    
    print("\n测试结束")

if __name__ == "__main__":
    main()
