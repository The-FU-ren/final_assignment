import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from glob import glob
from scipy.signal import medfilt
import random

class FixedDASDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_length=8000, window_step=4000, max_samples_per_file=50):
        self.data_dir = data_dir
        self.transform = transform
        self.sample_length = sample_length
        self.window_step = window_step
        self.max_samples_per_file = max_samples_per_file
        self.sample_metadata = []
        
        # 收集样本元数据
        self._collect_sample_metadata()
        
        # 初始化标签编码器
        self._initialize_label_encoder()
        
        # 计算类别分布
        self._compute_class_distribution()
    
    def _collect_sample_metadata(self):
        import h5py
        print(f"从 {self.data_dir} 收集样本元数据")
        
        # 获取所有类别目录
        categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        categories.sort()
        print(f"找到 {len(categories)} 个类别: {categories}")
        
        for category in categories:
            category_dir = os.path.join(self.data_dir, category)
            h5_files = glob(os.path.join(category_dir, "*.h5"))
            print(f"类别 {category} 有 {len(h5_files)} 个.h5文件")
            
            for h5_file in h5_files[:2]:  # 限制每个类别只处理前2个文件
                try:
                    with h5py.File(h5_file, 'r') as f:
                        if 'Acquisition/Raw[0]/RawData' in f:
                            data_shape = f["Acquisition/Raw[0]/RawData"].shape
                        else:
                            print(f"警告: 跳过文件 {h5_file}，缺少预期的数据路径")
                            continue
                    
                    # 计算可以生成的样本数量
                    num_samples = min(self.max_samples_per_file, (data_shape[0] - self.sample_length) // self.window_step + 1)
                    
                    # 存储元数据
                    for i in range(num_samples):
                        start_time = i * self.window_step
                        end_time = start_time + self.sample_length
                        if end_time <= data_shape[0]:
                            self.sample_metadata.append({
                                'h5_file': h5_file,
                                'category': category,
                                'start_time': start_time,
                                'end_time': end_time
                            })
                except Exception as e:
                    print(f"处理文件 {h5_file} 时出错: {e}")
                    continue
        
        print(f"成功收集 {len(self.sample_metadata)} 个样本的元数据")
    
    def _initialize_label_encoder(self):
        categories = sorted(set(meta['category'] for meta in self.sample_metadata))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(categories)
        print(f"类别编码: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
    
    def _compute_class_distribution(self):
        labels = [self.label_encoder.transform([meta['category']])[0] for meta in self.sample_metadata]
        label_counts = np.bincount(labels, minlength=len(self.label_encoder.classes_))
        self.class_distribution = dict(zip(self.label_encoder.classes_, label_counts))
        print(f"类别分布: {self.class_distribution}")
    
    def _denoise_signal(self, signal):
        """使用中值滤波去除信号噪声"""
        return medfilt(signal, kernel_size=3)
    
    def _normalize_signal(self, signal):
        """对信号进行Z-score归一化"""
        mean = np.mean(signal)
        std = np.std(signal)
        std = max(std, 1e-6)  # 避免除以零
        return (signal - mean) / std
    
    def _add_random_noise(self, signal, noise_level=0.01):
        """添加随机高斯噪声"""
        noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
        return signal + noise
    
    def _random_crop(self, signal, crop_length=8000):
        """随机裁剪信号，保持长度不变"""
        if signal.shape[0] <= crop_length:
            return signal
        start = np.random.randint(0, signal.shape[0] - crop_length + 1)
        return signal[start:start + crop_length]
    
    def _time_reversal(self, signal):
        """信号时间翻转"""
        # 使用copy()消除负步长，避免PyTorch错误
        return signal[::-1].copy()
    
    def _frequency_shift(self, signal, shift_factor=0.1):
        """频率域移位增强"""
        # 简单实现：信号的线性移位
        shift = int(shift_factor * signal.shape[0])
        return np.roll(signal, shift)
    
    def _apply_data_augmentation(self, signal):
        """应用数据增强，增加随机性和强度"""
        # 随机选择数据增强方法，提高增强概率
        if random.random() < 0.7:  # 提高噪声概率到70%
            signal = self._add_random_noise(signal, noise_level=0.02)  # 增加噪声水平
        if random.random() < 0.7:  # 提高时间翻转概率到70%
            signal = self._time_reversal(signal)
        if random.random() < 0.6:  # 提高频率移位概率到60%
            signal = self._frequency_shift(signal, shift_factor=0.15)  # 增加移位幅度
        if random.random() < 0.6:  # 提高随机裁剪概率到60%
            signal = self._random_crop(signal, self.sample_length)
        return signal
    
    def _load_sample(self, idx):
        import h5py
        meta = self.sample_metadata[idx]
        
        with h5py.File(meta['h5_file'], 'r') as f:
            data = f["Acquisition/Raw[0]/RawData"][meta['start_time']:meta['end_time'], 0]
        
        # 确保数据形状正确
        if data.shape[0] < self.sample_length:
            pad_length = self.sample_length - data.shape[0]
            data = np.pad(data, (0, pad_length), 'constant')
        
        # 预处理
        data = self._denoise_signal(data)
        data = self._normalize_signal(data)
        
        label = self.label_encoder.transform([meta['category']])[0]
        return data.astype(np.float32), label
    
    def __len__(self):
        return len(self.sample_metadata)
    
    def __getitem__(self, idx):
        signal, label = self._load_sample(idx)
        
        # 应用数据增强（仅在训练时）
        signal = self._apply_data_augmentation(signal)
        
        signal = signal.reshape(1, self.sample_length, 1)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义collate_fn函数
def collate_fn(batch):
    signals, labels = zip(*batch)
    signals = torch.stack(signals)
    labels = torch.stack(labels)
    
    # 归一化处理
    mean = signals.mean(dim=[0, 2, 3], keepdim=True)
    std = signals.std(dim=[0, 2, 3], keepdim=True)
    std = torch.clamp(std, min=1e-6)
    signals = (signals - mean) / std
    
    return signals, labels

# 定义get_data_loaders函数
def get_data_loaders(data_dir, batch_size=16, val_split=0.1, test_split=0.1, max_samples_per_file=20):
    # 创建完整数据集
    full_dataset = FixedDASDataset(data_dir, max_samples_per_file=max_samples_per_file)
    
    # 计算划分大小
    total_size = len(full_dataset)
    test_size = max(1, int(total_size * test_split))
    val_size = max(1, int(total_size * val_split))
    train_size = max(1, total_size - test_size - val_size)
    
    print(f"\n数据集划分: {train_size} 训练样本, {val_size} 验证样本, {test_size} 测试样本")
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

# 示例用法
def test_data_loader():
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=8, max_samples_per_file=5)
    
    print(f"训练加载器: {len(train_loader)} 批次")
    print(f"验证加载器: {len(val_loader)} 批次")
    print(f"测试加载器: {len(test_loader)} 批次")
    
    # 测试第一个批次
    for signals, labels in train_loader:
        print(f"信号形状: {signals.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels[:5]}")
        break

if __name__ == "__main__":
    test_data_loader()
