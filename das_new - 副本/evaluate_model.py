# 模型评估脚本

import torch
from model import DRSN_NTF
from trainer import Trainer
from data_loader_h5 import get_data_loaders_from_h5

def main():
    # 模型配置
    h5_file_path = "G:/gitcode/das_small_dataset_N50_20241230_154821.h5"
    best_model_path = "./models_test_7/20260109_170404/best_model.pth"
    device = "cpu"
    batch_size = 16
    
    print("="*60)
    print("DRSN-NTF模型评估")
    print("="*60)
    
    # 加载数据
    print("1. 加载测试数据...")
    train_loader, val_loader, test_loader = get_data_loaders_from_h5(
        h5_file_path=h5_file_path,
        batch_size=batch_size
    )
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   验证样本: {len(val_loader.dataset)}")
    print(f"   测试样本: {len(test_loader.dataset)}")
    
    # 初始化模型
    print("\n2. 初始化模型...")
    model = DRSN_NTF(num_classes=4)
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 初始化训练器
    print("\n3. 初始化训练器...")
    trainer = Trainer(model, device=device)
    print(f"   设备: {trainer.device}")
    
    # 加载最佳模型
    print("\n4. 加载最佳模型...")
    trainer.load_model(best_model_path)
    print(f"   模型加载成功: {best_model_path}")
    
    # 测试模型
    print("\n5. 测试模型性能...")
    
    # 测试验证集
    val_loss, val_accuracy = trainer.validate(val_loader)
    print(f"   验证损失: {val_loss:.3f}, 验证准确率: {val_accuracy:.2f}%")
    
    # 测试测试集
    test_accuracy = trainer.test(test_loader)
    print(f"   测试准确率: {test_accuracy:.2f}%")
    
    # 测试训练集
    train_loss, train_accuracy = trainer.validate(train_loader)
    print(f"   训练损失: {train_loss:.3f}, 训练准确率: {train_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("模型评估完成!")
    print(f"验证准确率: {val_accuracy:.2f}%")
    print(f"测试准确率: {test_accuracy:.2f}%")
    print(f"训练准确率: {train_accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
