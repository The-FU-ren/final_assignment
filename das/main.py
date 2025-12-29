import numpy as np
import torch
import argparse
from train import run_experiments, optimize_hyperparameters
from visualization import (
    plot_training_curves,
    plot_snr_vs_accuracy,
    generate_report,
    create_result_directories
)

def main():
    """主函数，协调整个实验流程"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='DRSN-NTF 模型训练与超参数优化')
    parser.add_argument('--optimize', action='store_true', help='运行超参数优化')
    parser.add_argument('--n-calls', type=int, default=50, help='超参数优化的调用次数')
    parser.add_argument('--batch-size', type=int, default=128, help='批量大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练周期')
    parser.add_argument('--initial-channels', type=int, default=8, help='初始通道数')
    parser.add_argument('--depth', type=int, default=34, help='模型深度')
    parser.add_argument('--kernel-size', type=int, default=3, help='卷积核大小')
    parser.add_argument('--threshold-N', type=float, default=1.0, help='阈值函数参数N')
    parser.add_argument('--dropout', type=float, default=0.2, help='丢弃率')
    parser.add_argument('--use-batchnorm', action='store_true', default=True, help='使用批归一化')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--use-data-augmentation', action='store_true', default=False, help='使用数据增强')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--patience', type=int, default=20, help='早停机制的耐心值')
    
    args = parser.parse_args()
    
    print("开始 DRSN-NTF 模型复现实验...")
    
    # 创建结果文件夹结构
    create_result_directories()
    
    # 创建本次运行的结果目录
    from visualization import create_run_directories, RUN_ID
    run_id = create_run_directories()
    print(f"本次运行ID: {run_id}")
    
    # 设置实验参数
    snr_values = [0, 1, 2, 3, 4, 5]  # 不同信噪比
    
    if args.optimize:
        # 运行超参数优化
        best_params = optimize_hyperparameters(
            n_calls=args.n_calls,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        # 使用最佳参数运行完整实验
        print("\n使用最佳参数运行完整实验...")
        results = run_experiments(
            snr_values=snr_values,
            batch_size=args.batch_size,
            epochs=args.epochs,
            run_id=RUN_ID,
            **best_params
        )
    else:
        # 运行默认实验或使用命令行指定的参数
        hyperparams = {
            'initial_channels': args.initial_channels,
            'depth': args.depth,
            'kernel_size': args.kernel_size,
            'threshold_N': args.threshold_N,
            'dropout': args.dropout,
            'use_batchnorm': args.use_batchnorm,
            'lr': args.lr,
            'use_data_augmentation': args.use_data_augmentation,
            'weight_decay': args.weight_decay,
            'patience': args.patience
        }
        
        # 运行实验
        results = run_experiments(
            snr_values=snr_values,
            batch_size=args.batch_size,
            epochs=args.epochs,
            run_id=RUN_ID,
            **hyperparams
        )
    
    # 可视化结果
    print("\n开始可视化结果...")
    
    # 绘制各信噪比下的训练曲线
    for snr in snr_values:
        plot_training_curves(results[snr]['fold_results'], snr_db=snr)
    
    # 绘制信噪比与准确率关系图
    plot_snr_vs_accuracy(results)
    
    # 生成实验报告
    generate_report(results)
    
    print("\n实验完成！所有结果已保存。")

if __name__ == "__main__":
    main()
