import argparse
import torch
import random
import numpy as np
import os
from datetime import datetime
from model import DRSN_NTF
from data_loader_fixed import get_data_loaders  # 使用修复后的数据加载器
from data_loader_h5 import get_data_loaders_from_h5
from trainer import Trainer
from visualization import plot_training_curves, save_training_data

# 固定随机种子，确保实验可复现
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='基于新阈值函数的深度残差收缩网络(DRSN-NTF)用于DAS信号分类')
    parser.add_argument('--data_dir', type=str, default='G:/DAS-dataset_3/DAS-dataset/data',
                        help='DAS数据集目录路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练批次大小')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--load_model', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备 (cuda 或 cpu)')
    parser.add_argument('--max_samples_per_file', type=int, default=100,
                        help='每个文件最多生成的样本数，用于控制内存使用')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='初始学习率')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # 固定随机种子
    set_random_seed()
    
    # Create datetime-named directory for results
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}/{current_time}"
    
    print("="*60)
    print("基于新阈值函数的深度残差收缩网络(DRSN-NTF)")
    print("用于DAS信号分类")
    print("="*60)
    print(f"结果将保存到: {save_dir}")
    print("="*60)
    
    # Step 1: Load data
    print("1. 加载数据中...")
    
    # 判断是目录还是单个HDF5文件
    if os.path.isdir(args.data_dir):
        # 使用目录数据加载器
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_samples_per_file=args.max_samples_per_file
        )
    else:
        # 使用单个HDF5文件数据加载器
        # 直接调用数据加载函数，不使用main保护块
        # 在Windows上使用num_workers=0避免多进程问题
        train_loader, val_loader, test_loader = get_data_loaders_from_h5(
            h5_file_path=args.data_dir,
            batch_size=args.batch_size
        )
    
    print(f"   训练批次数量: {len(train_loader)}")
    print(f"   验证批次数量: {len(val_loader)}")
    print(f"   测试批次数量: {len(test_loader)}")
    
    # Step 2: Initialize model
    print("\n2. 初始化模型中...")
    
    # 根据数据集类型设置类别数量
    # HDF5文件有4个类别，目录数据集有9个类别
    if os.path.isdir(args.data_dir):
        num_classes = 9
    else:
        num_classes = 4
    
    model = DRSN_NTF(num_classes=num_classes)
    print(f"   模型初始化成功! 使用 {num_classes} 个类别")
    
    # Step 3: Initialize trainer
    print("\n3. 初始化训练器中...")
    trainer = Trainer(model, device=args.device, learning_rate=args.learning_rate)
    print(f"   使用设备: {trainer.device}")
    
    # Step 4: Load pre-trained model if specified
    if args.load_model:
        print(f"\n4. 从 {args.load_model} 加载预训练模型中...")
        trainer.load_model(args.load_model)
        print("   模型加载成功!")
    
    # Step 5: Train the model
    print(f"\n5. 开始训练，共 {args.num_epochs} 轮...")
    print(f"   设备: {trainer.device}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   验证样本: {len(val_loader.dataset)}")
    print(f"   测试样本: {len(test_loader.dataset)}")
    
    try:
        best_val_accuracy, training_curves = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            save_dir=save_dir
        )
        print(f"\n   训练完成!")
        print(f"   最佳验证准确率: {best_val_accuracy:.2f}%")
        
        # Step 6: 保存训练曲线和数据
        print("\n6. 保存训练曲线和数据...")
        # 绘制并保存训练曲线
        plot_training_curves(
            training_curves['train_losses'],
            training_curves['train_accuracies'],
            training_curves['val_losses'],
            training_curves['val_accuracies'],
            save_dir=save_dir
        )
        # 保存训练数据
        save_training_data(
            training_curves['train_losses'],
            training_curves['train_accuracies'],
            training_curves['val_losses'],
            training_curves['val_accuracies'],
            save_dir=save_dir
        )
        print(f"   训练数据已保存到: {save_dir}")
        
        # Step 7: Test the model
        print("\n7. 测试模型中...")
        # Load the best model
        best_model_path = f"{save_dir}/best_model.pth"
        trainer.load_model(best_model_path)
        
        test_accuracy = trainer.test(test_loader)
        print(f"   测试准确率: {test_accuracy:.2f}%")
        
        print("\n" + "="*60)
        print("训练和测试已成功完成!")
        print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
        print(f"测试准确率: {test_accuracy:.2f}%")
        print(f"结果已保存到: {save_dir}")
        print("="*60)
        
    except Exception as e:
        import traceback
        print(f"\n训练过程中发生错误: {e}")
        traceback.print_exc()
        print("\n请检查错误信息，调试训练流程")

if __name__ == "__main__":
    main()
