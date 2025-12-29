import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import time

# 生成唯一的运行ID，用于区分不同的训练结果
def generate_run_id():
    """生成唯一的运行ID"""
    return time.strftime("%Y%m%d_%H%M%S")

# 当前运行ID
RUN_ID = generate_run_id()

# 创建结果文件夹结构
def create_result_directories():
    """创建结果文件夹结构"""
    # 基础目录结构
    directories = [
        '结果',
        '结果/图表',
        '结果/图表/训练曲线',
        '结果/图表/性能对比',
        '结果/图表/混淆矩阵',
        '结果/图表/F1分数',
        '结果/报告',
        '结果/数据'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# 创建本次运行的结果目录
def create_run_directories():
    """创建本次运行的结果目录"""
    # 本次运行的目录结构
    directories = [
        f'结果/图表/训练曲线/run_{RUN_ID}',
        f'结果/图表/性能对比/run_{RUN_ID}',
        f'结果/图表/混淆矩阵/run_{RUN_ID}',
        f'结果/图表/F1分数/run_{RUN_ID}',
        f'结果/报告/run_{RUN_ID}',
        f'结果/数据/run_{RUN_ID}'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return RUN_ID

def plot_training_curves(fold_results, snr_db=0):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    for i, fold_result in enumerate(fold_results):
        plt.plot(fold_result['train_losses'], label=f'Train Fold {i+1}')
        plt.plot(fold_result['test_losses'], label=f'Test Fold {i+1}', linestyle='--')
    plt.title(f'Loss Curves (SNR = {snr_db}dB)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    for i, fold_result in enumerate(fold_results):
        plt.plot(fold_result['train_accs'], label=f'Train Fold {i+1}')
        plt.plot(fold_result['test_accs'], label=f'Test Fold {i+1}', linestyle='--')
    plt.title(f'Accuracy Curves (SNR = {snr_db}dB)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f'结果/图表/训练曲线/run_{RUN_ID}/训练曲线_{snr_db}dB.png')
    plt.close()

def plot_snr_vs_accuracy(all_results):
    """绘制不同信噪比下的准确率对比"""
    snr_values = sorted(all_results.keys())
    avg_best_accs = [all_results[snr]['avg_best_acc'] for snr in snr_values]
    avg_final_accs = [all_results[snr]['avg_final_acc'] for snr in snr_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, avg_best_accs, marker='o', label='Average Best Accuracy')
    plt.plot(snr_values, avg_final_accs, marker='s', label='Average Final Accuracy')
    
    plt.title('Accuracy vs SNR for DRSN-NTF')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xticks(snr_values)
    plt.ylim(0, 1)
    
    plt.savefig(f'结果/图表/性能对比/run_{RUN_ID}/信噪比_准确率.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, snr_db=0):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (SNR = {snr_db}dB)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    plt.savefig(f'结果/图表/混淆矩阵/run_{RUN_ID}/混淆矩阵_{snr_db}dB.png')
    plt.close()

def plot_f1_scores(f1_scores, classes, snr_db=0):
    """绘制各事件类型的F1分数"""
    plt.figure(figsize=(10, 6))
    plt.bar(classes, f1_scores)
    plt.title(f'F1-scores for Each Event Type (SNR = {snr_db}dB)')
    plt.xlabel('Event Type')
    plt.ylabel('F1-score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 在柱状图上添加数值
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'结果/图表/F1分数/run_{RUN_ID}/F1分数_{snr_db}dB.png')
    plt.close()

def plot_model_comparison(results_dict, snr_values):
    """绘制不同模型在不同信噪比下的性能对比"""
    plt.figure(figsize=(12, 8))
    
    for model_name, results in results_dict.items():
        accs = [results[snr]['avg_final_acc'] for snr in snr_values]
        plt.plot(snr_values, accs, marker='o', label=model_name)
    
    plt.title('Model Comparison Across Different SNRs')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xticks(snr_values)
    plt.ylim(0, 1)
    
    plt.savefig(f'结果/图表/性能对比/run_{RUN_ID}/模型对比.png')
    plt.close()

def generate_report(all_results):
    """生成实验报告"""
    report_path = f'结果/报告/run_{RUN_ID}/实验报告.txt'
    with open(report_path, 'w') as f:
        f.write(f'DRSN-NTF 实验报告\n')
        f.write(f'运行ID: {RUN_ID}\n')
        f.write('='*50 + '\n\n')
        
        for snr in sorted(all_results.keys()):
            f.write(f'SNR = {snr}dB\n')
            f.write('-'*30 + '\n')
            f.write(f'平均最佳准确率: {all_results[snr]["avg_best_acc"]:.4f}\n')
            f.write(f'平均最终准确率: {all_results[snr]["avg_final_acc"]:.4f}\n\n')
        
        # 计算平均性能
        avg_overall = np.mean([all_results[snr]['avg_final_acc'] for snr in all_results.keys()])
        f.write(f'所有SNR下的平均准确率: {avg_overall:.4f}\n')
        
    print(f"实验报告已生成: {report_path}")
