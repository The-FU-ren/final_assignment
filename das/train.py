import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
import os
import signal
import sys
from model import DRSN_NTF
from data_processor import load_data, create_kfold_loaders
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
# 添加混合精度训练支持
from torch.amp import autocast, GradScaler

# 全局变量，用于保存当前训练状态
global_training_state = {
    'model': None,
    'optimizer': None,
    'scheduler': None,
    'scaler': None,
    'current_fold': 0,
    'current_epoch': 0,
    'train_losses': [],
    'train_accs': [],
    'test_losses': [],
    'test_accs': [],
    'fold_results': []
}

# 信号处理函数，捕获Ctrl+C
def signal_handler(sig, frame):
    print('\n\n收到中断信号，正在保存训练结果...')
    
    # 保存当前模型权重
    if global_training_state['model'] is not None:
        model_path = '结果/数据/中断模型.pth'
        torch.save({
            'model_state_dict': global_training_state['model'].state_dict(),
            'optimizer_state_dict': global_training_state['optimizer'].state_dict() if global_training_state['optimizer'] is not None else None,
            'scheduler_state_dict': global_training_state['scheduler'].state_dict() if global_training_state['scheduler'] is not None else None,
            'scaler_state_dict': global_training_state['scaler'].state_dict() if global_training_state['scaler'] is not None else None,
            'current_fold': global_training_state['current_fold'],
            'current_epoch': global_training_state['current_epoch'],
            'train_losses': global_training_state['train_losses'],
            'train_accs': global_training_state['train_accs'],
            'test_losses': global_training_state['test_losses'],
            'test_accs': global_training_state['test_accs'],
            'fold_results': global_training_state['fold_results']
        }, model_path)
        print(f"模型权重已保存到 {model_path}")
    
    # 保存训练记录
    if global_training_state['train_losses']:
        train_records_path = '结果/数据/中断训练记录.npy'
        np.save(train_records_path, {
            'train_losses': global_training_state['train_losses'],
            'train_accs': global_training_state['train_accs'],
            'test_losses': global_training_state['test_losses'],
            'test_accs': global_training_state['test_accs'],
            'fold_results': global_training_state['fold_results']
        })
        print(f"训练记录已保存到 {train_records_path}")
    
    print('训练结果已保存，程序即将终止')
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)

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

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 缩放损失，反向传播
        scaler.scale(loss).backward()
        # 更新参数
        scaler.step(optimizer)
        # 更新缩放器
        scaler.update()
        
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
            
            # 使用混合精度验证
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
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

# 早停机制类
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, delta=0):
        """初始化早停机制"""
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model):
        """早停机制的调用方法"""
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'\n早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
        
        return self.early_stop

def train_model(snr_db=0, batch_size=128, epochs=100, num_classes=6,
                initial_channels=8, depth=34, kernel_size=3,
                threshold_N=1.0, dropout=0.2, use_batchnorm=True,
                lr=0.01, weight_decay=1e-4, patience=20, use_data_augmentation=True, run_id=None):
    """训练模型"""
    # 设置设备 - 只使用GPU
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的GPU设备！请确保已安装CUDA和GPU版PyTorch。")
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # 加载数据
    data, labels = load_data()
    
    # 创建十折交叉验证加载器，使用多线程数据加载和数据增强
    kfold_loaders = create_kfold_loaders(data, labels, batch_size=batch_size, snr_db=snr_db, 
                                        num_workers=4, use_data_augmentation=use_data_augmentation)
    
    # 重置全局训练状态
    global global_training_state
    global_training_state = {
        'model': None,
        'optimizer': None,
        'scheduler': None,
        'scaler': None,
        'current_fold': 0,
        'current_epoch': 0,
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': [],
        'fold_results': []
    }
    
    fold_results = []
    
    # 十折交叉验证
    for fold, (train_loader, test_loader) in enumerate(kfold_loaders):
        print(f"\n=== Fold {fold+1}/10 ===")
        
        # 更新当前fold
        global_training_state['current_fold'] = fold+1
        global_training_state['current_epoch'] = 0
        
        # 初始化模型
        model = DRSN_NTF(num_classes=num_classes, 
                        initial_channels=initial_channels, 
                        depth=depth, 
                        kernel_size=kernel_size,
                        threshold_N=threshold_N,
                        dropout=dropout,
                        use_batchnorm=use_batchnorm).to(device)
        
        # 定义损失函数和优化器，添加L2正则化
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)  # L2正则化
        
        # 获取学习率调度器
        scheduler = get_lr_scheduler(optimizer, epochs)
        
        # 初始化GradScaler用于混合精度训练
        scaler = GradScaler()
        
        # 初始化早停机制
        early_stopping = EarlyStopping(patience=patience)
        
        # 更新全局训练状态
        global_training_state['model'] = model
        global_training_state['optimizer'] = optimizer
        global_training_state['scheduler'] = scheduler
        global_training_state['scaler'] = scaler
        
        # 训练记录
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
        # 更新全局训练记录
        global_training_state['train_losses'] = train_losses
        global_training_state['train_accs'] = train_accs
        global_training_state['test_losses'] = test_losses
        global_training_state['test_accs'] = test_accs
        global_training_state['fold_results'] = fold_results
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(epochs):
            # 更新当前epoch
            global_training_state['current_epoch'] = epoch+1
            
            # 训练
            train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            
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
            
            # 早停检查
            if early_stopping(test_loss, model):
                print(f"\n触发早停机制，在epoch {epoch+1}停止训练")
                break
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # 保存fold结果
        fold_result = {
            'fold': fold+1,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'best_test_acc': max(test_accs),
            'final_test_acc': test_accs[-1],
            'training_time': training_time
        }
        fold_results.append(fold_result)
        
        # 更新全局fold结果
        global_training_state['fold_results'] = fold_results
        
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

def run_experiments(snr_values=[0, 1, 2, 3, 4, 5], batch_size=128, epochs=100, run_id=None, **hyperparams):
    """运行不同信噪比下的实验"""
    all_results = {}
    
    # 如果没有提供run_id，生成一个新的
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    
    for snr in snr_values:
        print(f"\n\n" + "="*60)
        print(f"Running experiment for SNR = {snr}dB")
        print("="*60)
        
        fold_results, avg_best_acc, avg_final_acc = train_model(
            snr_db=snr, 
            batch_size=batch_size, 
            epochs=epochs,
            run_id=run_id,
            **hyperparams
        )
        
        all_results[snr] = {
            'fold_results': fold_results,
            'avg_best_acc': avg_best_acc,
            'avg_final_acc': avg_final_acc
        }
    
    # 保存实验结果
    import os
    run_result_dir = f'结果/数据/run_{run_id}'
    os.makedirs(run_result_dir, exist_ok=True)
    np.save(f'{run_result_dir}/实验结果.npy', all_results)
    print(f"\n实验结果已保存到 {run_result_dir}/实验结果.npy")
    
    return all_results

def optimize_hyperparameters(n_calls=50, batch_size=128, epochs=100):
    """使用贝叶斯优化进行超参数调优"""
    # 确保n_calls至少为10，因为gp_minimize要求n_calls >= 10
    n_calls = max(n_calls, 10)
    
    # 定义超参数空间
    space = [
        Integer(4, 16, name='initial_channels'),
        Integer(18, 54, name='depth'),
        Integer(3, 7, name='kernel_size'),
        Real(0.5, 2.0, name='threshold_N'),
        Real(0.1, 0.5, name='dropout'),
        Categorical([True, False], name='use_batchnorm'),
        Real(0.001, 0.1, prior='log-uniform', name='lr')
    ]
    
    # 加载数据，用于获取数据维度和类别数
    data, labels = load_data()
    num_classes = len(np.unique(labels))
    
    @use_named_args(space)
    def objective(**params):
        """优化目标函数"""
        print(f"\n\n" + "="*80)
        print(f"Running hyperparameter optimization with params: {params}")
        print("="*80)
        
        # 只使用5折交叉验证来加速优化
        # 创建简化的5折交叉验证
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 只使用GPU
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用的GPU设备！请确保已安装CUDA和GPU版PyTorch。")
        device = torch.device('cuda')
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
            print(f"\n--- Fold {fold+1}/5 ---")
            
            # 划分训练集和测试集
            train_data, test_data = data[train_idx], data[test_idx]
            train_labels, test_labels = labels[train_idx], labels[test_idx]
            
            # 预处理
            from data_processor import preprocess_data, create_kfold_loaders
            train_data = preprocess_data(train_data, snr_db=0)
            test_data = preprocess_data(test_data, snr_db=0)
            
            # 创建数据集和数据加载器
            from data_processor import DAADataset
            from torch.utils.data import DataLoader
            train_dataset = DAADataset(train_data, train_labels)
            test_dataset = DAADataset(test_data, test_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # 初始化模型 - 移除lr参数，因为它不应该传递给模型
            model_params = {k: v for k, v in params.items() if k != 'lr'}
            model = DRSN_NTF(num_classes=num_classes, **model_params).to(device)
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=0.0001)
            
            # 获取学习率调度器
            scheduler = get_lr_scheduler(optimizer, epochs)
            
            # 初始化GradScaler用于混合精度训练
            scaler = GradScaler()
            
            # 训练模型
            for epoch in range(epochs):
                train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
                scheduler.step()
            
            # 验证模型
            test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
            fold_results.append(test_acc)
        
        # 计算平均准确率
        avg_acc = np.mean(fold_results)
        print(f"\nAverage accuracy: {avg_acc:.4f}")
        
        # 贝叶斯优化最大化准确率，所以返回负的准确率（因为gp_minimize是最小化函数）
        return -avg_acc
    
    # 运行贝叶斯优化
    print("开始超参数优化...")
    result = gp_minimize(
        objective, 
        space, 
        n_calls=n_calls, 
        random_state=42,
        verbose=True
    )
    
    # 打印最佳参数
    print(f"\n\n" + "="*80)
    print("超参数优化完成！")
    print("最佳参数：")
    best_params = dict(zip([p.name for p in space], result.x))
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print(f"最佳准确率：{-result.fun:.4f}")
    print("="*80)
    
    # 保存最佳参数
    import os
    # 创建本次运行的结果目录
    run_result_dir = f'结果/数据/run_{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(run_result_dir, exist_ok=True)
    np.save(f'{run_result_dir}/最佳超参数.npy', best_params)
    print(f"\n最佳参数已保存到 {run_result_dir}/最佳超参数.npy")
    
    return best_params

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'optimize':
        # 运行超参数优化
        best_params = optimize_hyperparameters()
        print("\n使用最佳参数运行完整实验...")
        results = run_experiments(**best_params)
    else:
        # 运行默认实验
        results = run_experiments()
    
    # 保存结果
    np.save('结果/数据/实验结果.npy', results)
    print("\n实验结果已保存到 结果/数据/实验结果.npy")
