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

def load_data(data_path=r"G:\gitcode\das_small_dataset_N50_20241230_154821.h5"):
    """加载H5数据集"""
    import h5py
    import numpy as np
    
    print(f"Loading dataset from: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # 查看数据集结构
        print("Dataset structure:")
        def print_attrs(name, obj):
            print(f"{name}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
        f.visititems(print_attrs)
        
        # 检查是否包含train组
        if 'train' in f:
            print("\nLoading train data...")
            train_group = f['train']
            
            # 读取训练数据
            int_data = train_group['int_data']
            labels = train_group['labels'][:]
            
            # 转换数据格式
            num_samples = len(int_data)
            print(f"Found {num_samples} samples in train dataset")
            
            # 初始化数据数组
            # 假设每个样本长度为8000，根据实际情况调整
            signal_length = 8000
            data = np.zeros((num_samples, signal_length))
            
            # 读取每个样本
            for i, key in enumerate(int_data.keys()):
                sample_data = int_data[key][:]
                # 如果样本长度超过8000，截断；如果不足，补零
                if len(sample_data) >= signal_length:
                    data[i] = sample_data[:signal_length]
                else:
                    data[i, :len(sample_data)] = sample_data
            
            print(f"Loaded data shape: {data.shape}")
            print(f"Loaded labels shape: {labels.shape}")
            
            return data, labels
        else:
            # 直接读取根级别的数据（如果数据集没有train组）
            print("\nLoading data from root level...")
            data = f['int_data'][:]
            labels = f['labels'][:]
            
            # 确保数据形状符合要求 (样本数, 8000)
            if data.ndim == 2 and data.shape[1] > 8000:
                # 如果每个样本长度超过8000，截断
                data = data[:, :8000]
            elif data.ndim == 1:
                # 如果是一维数据，重塑为 (1, len(data))
                data = data.reshape(1, -1)
                if data.shape[1] > 8000:
                    data = data[:, :8000]
            
            print(f"Loaded data shape: {data.shape}")
            print(f"Loaded labels shape: {labels.shape}")
            
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
