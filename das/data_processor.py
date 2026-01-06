import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy import signal
import random

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
        
        # 注意：transform会在GPU上应用，此处只返回原始数据
        return sample, label

def add_noise(data, snr_db):
    """添加指定信噪比的高斯噪声"""
    signal_power = np.mean(data ** 2, axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise

def time_stretch(signal, rate=1.0):
    """时间拉伸"""
    original_length = signal.shape[-1]
    if rate == 1.0:
        return signal
    
    # 计算新的长度
    new_length = int(original_length * rate)
    stretched_signal = np.zeros_like(signal)
    
    for i in range(signal.shape[0]):
        # 使用线性插值进行时间拉伸
        if new_length > original_length:
            # 放大
            x_old = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, new_length)
            stretched = np.interp(x_new, x_old, signal[i])
            stretched_signal[i] = stretched[:original_length]  # 截断到原始长度
        else:
            # 缩小
            x_old = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, new_length)
            stretched = np.interp(x_new, x_old, signal[i])
            stretched_signal[i, :new_length] = stretched
    
    return stretched_signal

def amplitude_scale(signal, scale_factor=1.0):
    """幅度缩放"""
    return signal * scale_factor

def time_shift(signal, shift_ratio=0.0):
    """时间偏移"""
    if shift_ratio == 0.0:
        return signal
    
    signal_length = signal.shape[-1]
    shift_samples = int(signal_length * shift_ratio)
    shifted_signal = np.zeros_like(signal)
    
    if shift_samples > 0:
        # 向右偏移
        shifted_signal[:, shift_samples:] = signal[:, :-shift_samples]
    else:
        # 向左偏移
        shifted_signal[:, :shift_samples] = signal[:, -shift_samples:]
    
    return shifted_signal

class DataAugmentation:
    """数据增强类"""
    def __init__(self, 
                 time_stretch_rates=[0.8, 0.9, 1.0, 1.1, 1.2],
                 amplitude_scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                 snr_dbs=[5, 10, 15, 20],
                 time_shifts=[-0.1, -0.05, 0.0, 0.05, 0.1]):
        self.time_stretch_rates = time_stretch_rates
        self.amplitude_scales = amplitude_scales
        self.snr_dbs = snr_dbs
        self.time_shifts = time_shifts
    
    def __call__(self, signal):
        """应用数据增强"""
        import random
        
        # 确保输入是2D数组 (batch_size, signal_length)
        if signal.ndim == 3:
            # 如果是3D数组 (1, batch_size, signal_length)，转换为2D
            signal = signal.squeeze(0)
        elif signal.ndim == 1:
            # 如果是1D数组 (signal_length)，转换为2D
            signal = signal.reshape(1, -1)
        
        # 随机选择时间拉伸率
        stretch_rate = random.choice(self.time_stretch_rates)
        signal = time_stretch(signal, rate=stretch_rate)
        
        # 随机选择幅度缩放因子
        scale_factor = random.choice(self.amplitude_scales)
        signal = amplitude_scale(signal, scale_factor=scale_factor)
        
        # 随机选择SNR添加高斯噪声
        snr_db = random.choice(self.snr_dbs)
        signal = add_noise(signal, snr_db=snr_db)
        
        # 随机选择时间偏移
        shift_ratio = random.choice(self.time_shifts)
        signal = time_shift(signal, shift_ratio=shift_ratio)
        
        # 恢复原始形状
        return signal

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

def create_kfold_loaders(data, labels, batch_size=128, snr_db=0, n_splits=10, num_workers=8, pin_memory=True, use_data_augmentation=True):
    """创建十折交叉验证的数据加载器"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    loaders = []
    
    # 初始化数据增强器
    data_augmentation = DataAugmentation()
    
    for train_idx, test_idx in kfold.split(data):
        # 划分训练集和测试集
        train_data, test_data = data[train_idx], data[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        
        # 预处理
        train_data = preprocess_data(train_data, snr_db)
        test_data = preprocess_data(test_data, snr_db)
        
        # 创建数据集
        if use_data_augmentation:
            train_dataset = DAADataset(train_data, train_labels, transform=data_augmentation)
        else:
            train_dataset = DAADataset(train_data, train_labels)
        
        test_dataset = DAADataset(test_data, test_labels)  # 测试集不使用数据增强
        
        # 创建数据加载器，使用num_workers启用多线程和pin_memory加速GPU传输
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        
        loaders.append((train_loader, test_loader))
    
    return loaders
