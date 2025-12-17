import numpy as np
import torch
from train import run_experiments
from visualization import (
    plot_training_curves,
    plot_snr_vs_accuracy,
    generate_report,
    create_result_directories
)

def main():
    """主函数，协调整个实验流程"""
    print("开始 DRSN-NTF 模型复现实验...")
    
    # 创建结果文件夹结构
    create_result_directories()
    
    # 设置实验参数
    snr_values = [0, 1, 2, 3, 4, 5]  # 不同信噪比
    batch_size = 128  # 批量大小
    epochs = 100  # 训练周期
    
    # 运行实验
    results = run_experiments(
        snr_values=snr_values,
        batch_size=batch_size,
        epochs=epochs
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
