# 简单训练脚本，用于调试

import torch
import os
from model import DRSN_NTF
from trainer import Trainer
from data_loader import get_data_loaders

def main():
    # 训练配置
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    batch_size = 8
    num_epochs = 2
    learning_rate = 0.001
    max_samples_per_file = 20  # 减少每个文件的样本数
    device = "cpu"
    save_dir = "./simple_train"
    
    print("="*60)
    print("简单训练脚本")
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
        
        # 初始化模型
        print("\n2. 初始化模型...")
        num_classes = 9  # 目录数据集有9个类别
        model = DRSN_NTF(num_classes=num_classes)
        print(f"   模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 初始化训练器
        print("\n3. 初始化训练器...")
        trainer = Trainer(model, device=device, learning_rate=learning_rate)
        print(f"   设备: {trainer.device}")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练
        print(f"\n4. 开始训练，共 {num_epochs} 轮...")
        for epoch in range(num_epochs):
            print(f"\n轮次 {epoch+1}/{num_epochs}:")
            
            # 训练一轮
            print("   训练中...")
            train_loss, train_accuracy = trainer.train_one_epoch(
                train_loader=train_loader,
                epoch=epoch,
                total_epochs=num_epochs
            )
            print(f"   训练损失: {train_loss:.3f}, 训练准确率: {train_accuracy:.2f}%")
            
            # 验证
            print("   验证中...")
            val_loss, val_accuracy = trainer.validate(val_loader)
            print(f"   验证损失: {val_loss:.3f}, 验证准确率: {val_accuracy:.2f}%")
            
            # 学习率调度器
            trainer.scheduler.step()
            
            # 保存模型
            model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            trainer.save_model(model_path)
            print(f"   模型已保存到: {model_path}")
        
        # 测试
        print("\n5. 测试模型...")
        test_accuracy = trainer.test(test_loader)
        print(f"   测试准确率: {test_accuracy:.2f}%")
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        
    except Exception as e:
        import traceback
        print(f"\n训练失败: {e}")
        traceback.print_exc()
        
    print("\n测试结束")

if __name__ == "__main__":
    main()
