import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    
    plt.savefig(f'training_curves_{snr_db}dB.png')
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
    
    plt.savefig('snr_vs_accuracy.png')
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
    
    plt.savefig(f'confusion_matrix_{snr_db}dB.png')
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
    plt.savefig(f'f1_scores_{snr_db}dB.png')
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
    
    plt.savefig('model_comparison.png')
    plt.close()

def generate_report(all_results):
    """生成实验报告"""
    with open('experiment_report.txt', 'w') as f:
        f.write('DRSN-NTF 实验报告\n')
        f.write('='*50 + '\n\n')
        
        for snr in sorted(all_results.keys()):
            f.write(f'SNR = {snr}dB\n')
            f.write('-'*30 + '\n')
            f.write(f'Average Best Accuracy: {all_results[snr]["avg_best_acc"]:.4f}\n')
            f.write(f'Average Final Accuracy: {all_results[snr]["avg_final_acc"]:.4f}\n\n')
        
        # 计算平均性能
        avg_overall = np.mean([all_results[snr]['avg_final_acc'] for snr in all_results.keys()])
        f.write(f'Average Accuracy Across All SNRs: {avg_overall:.4f}\n')
        
    print("Experiment report generated: experiment_report.txt")
