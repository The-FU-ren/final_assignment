import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
from model import DRSN_NTF
from data_processor import load_data, create_kfold_loaders

def get_lr_scheduler(optimizer, total_epochs=100):
    """获取学习率调度器"""
    def lr_lambda(epoch):
        if epoch < 40:
            return 0.1
        elif epoch < 80:
            return 0.01
        else:
            return 0.001
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, test_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def train_model(snr_db=0, batch_size=128, epochs=100, num_classes=6):
    """训练模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    data, labels = load_data()
    
    # 创建十折交叉验证加载器
    kfold_loaders = create_kfold_loaders(data, labels, batch_size=batch_size, snr_db=snr_db)
    
    fold_results = []
    
    # 十折交叉验证
    for fold, (train_loader, test_loader) in enumerate(kfold_loaders):
        print(f"\n=== Fold {fold+1}/10 ===")
        
        # 初始化模型
        model = DRSN_NTF(num_classes=num_classes).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        
        # 获取学习率调度器
        scheduler = get_lr_scheduler(optimizer, epochs)
        
        # 训练记录
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # 验证
            test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
            
            # 更新学习率
            scheduler.step()
            
            # 记录
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            # 打印日志
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs:3d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                      f"F1: {test_f1:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # 保存fold结果
        fold_results.append({
            'fold': fold+1,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'best_test_acc': max(test_accs),
            'final_test_acc': test_accs[-1],
            'training_time': training_time
        })
        
        print(f"Fold {fold+1} completed in {training_time:.2f} seconds")
        print(f"Best Test Acc: {max(test_accs):.4f} | Final Test Acc: {test_accs[-1]:.4f}")
    
    # 计算平均结果
    avg_best_acc = np.mean([r['best_test_acc'] for r in fold_results])
    avg_final_acc = np.mean([r['final_test_acc'] for r in fold_results])
    avg_training_time = np.mean([r['training_time'] for r in fold_results])
    
    print(f"\n=== 十折交叉验证平均结果 ===")
    print(f"Average Best Test Acc: {avg_best_acc:.4f}")
    print(f"Average Final Test Acc: {avg_final_acc:.4f}")
    print(f"Average Training Time: {avg_training_time:.2f} seconds")
    
    return fold_results, avg_best_acc, avg_final_acc

def run_experiments(snr_values=[0, 1, 2, 3, 4, 5], batch_size=128, epochs=100):
    """运行不同信噪比下的实验"""
    all_results = {}
    
    for snr in snr_values:
        print(f"\n\n" + "="*60)
        print(f"Running experiment for SNR = {snr}dB")
        print("="*60)
        
        fold_results, avg_best_acc, avg_final_acc = train_model(
            snr_db=snr, 
            batch_size=batch_size, 
            epochs=epochs
        )
        
        all_results[snr] = {
            'fold_results': fold_results,
            'avg_best_acc': avg_best_acc,
            'avg_final_acc': avg_final_acc
        }
    
    return all_results

if __name__ == "__main__":
    # 运行实验
    results = run_experiments()
    
    # 保存结果
    np.save('results/data/experiment_results.npy', results)
    print("\nExperiment results saved to results/data/experiment_results.npy")
