import argparse
import torch
from datetime import datetime
from model import DRSN_NTF
from data_loader import get_data_loaders
from trainer import Trainer
from visualization import plot_training_curves, save_training_data

def parse_args():
    parser = argparse.ArgumentParser(description='基于新阈值函数的深度残差收缩网络(DRSN-NTF)用于DAS信号分类')
    parser.add_argument('--data_dir', type=str, default='G:/DAS-dataset_3/DAS-dataset/data',
                        help='DAS数据集目录路径')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='训练批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--load_model', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备 (cuda 或 cpu)')
    parser.add_argument('--max_samples_per_file', type=int, default=100,
                        help='每个文件最多生成的样本数，用于控制内存使用')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
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
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_samples_per_file=args.max_samples_per_file
    )
    print(f"   训练批次数量: {len(train_loader)}")
    print(f"   验证批次数量: {len(val_loader)}")
    print(f"   测试批次数量: {len(test_loader)}")
    
    # Step 2: Initialize model
    print("\n2. 初始化模型中...")
    model = DRSN_NTF(num_classes=9)
    print("   模型初始化成功!")
    
    # Step 3: Initialize trainer
    print("\n3. 初始化训练器中...")
    trainer = Trainer(model, device=args.device)
    print(f"   使用设备: {trainer.device}")
    
    # Step 4: Load pre-trained model if specified
    if args.load_model:
        print(f"\n4. 从 {args.load_model} 加载预训练模型中...")
        trainer.load_model(args.load_model)
        print("   模型加载成功!")
    
    # Step 5: Train the model
    print(f"\n5. 开始训练，共 {args.num_epochs} 轮...")
    best_val_accuracy, training_curves = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=save_dir
    )
    print(f"   训练完成!")
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

if __name__ == "__main__":
    main()
