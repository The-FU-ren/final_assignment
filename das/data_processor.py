import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

class DAADataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)  # (1, 8000)
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

def add_noise(data, snr_db):
    """添加指定信噪比的高斯噪声"""
    signal_power = np.mean(data ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def load_data(data_path=None):
    """加载数据集，这里假设数据已经预处理为numpy数组"""
    # 实际使用时，需要替换为真实数据加载逻辑
    # 这里生成模拟数据作为示例
    num_samples = 1000
    signal_length = 8000
    num_classes = 6
    
    # 生成模拟数据
    data = np.random.randn(num_samples, signal_length)
    labels = np.random.randint(0, num_classes, num_samples)
    
    return data, labels

def preprocess_data(data, snr_db=0, normalize=True):
    """数据预处理"""
    # 添加噪声
    noisy_data = add_noise(data, snr_db)
    
    # 归一化
    if normalize:
        scaler = StandardScaler()
        noisy_data = scaler.fit_transform(noisy_data)
    
    return noisy_data

def create_kfold_loaders(data, labels, batch_size=128, snr_db=0, n_splits=10):
    """创建十折交叉验证的数据加载器"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    loaders = []
    
    for train_idx, test_idx in kfold.split(data):
        # 划分训练集和测试集
        train_data, test_data = data[train_idx], data[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        
        # 预处理
        train_data = preprocess_data(train_data, snr_db)
        test_data = preprocess_data(test_data, snr_db)
        
        # 创建数据集
        train_dataset = DAADataset(train_data, train_labels)
        test_dataset = DAADataset(test_data, test_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        loaders.append((train_loader, test_loader))
    
    return loaders
