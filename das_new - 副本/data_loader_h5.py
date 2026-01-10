import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

class SingleH5DASDataset(Dataset):
    def __init__(self, h5_file_path, split='train', transform=None, sample_length=8000):
        """
        初始化DAS数据集（单个HDF5文件版本）
        
        参数:
            h5_file_path: HDF5文件路径
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据变换
            sample_length: 单个样本的长度
        """
        self.h5_file_path = h5_file_path
        self.split = split
        self.transform = transform
        self.sample_length = sample_length
        
        # 存储样本信息
        self.sample_info = []
        self.labels = []
        
        # 收集样本信息
        self._collect_sample_info()
        
        # 类别映射
        self.classes = ['climb', 'normal', 'shaking', 'tap']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
    def _collect_sample_info(self):
        """
        从HDF5文件中收集样本信息
        """
        import h5py
        
        print(f"从 {self.h5_file_path} 收集 {self.split} 样本信息")
        
        with h5py.File(self.h5_file_path, 'r') as f:
            # 获取当前划分的组
            split_group = f[self.split]
            
            # 获取数据和标签
            data_group = split_group['int_data']
            self.labels = split_group['labels'][:]
            
            # 遍历所有样本
            for sample_name in sorted(data_group.keys()):
                self.sample_info.append({
                    'sample_name': sample_name,
                    'data_path': f"{self.split}/int_data/{sample_name}"
                })
        
        print(f"成功收集 {len(self.sample_info)} 个 {self.split} 样本")
        print(f"标签数量: {len(self.labels)}")
    
    def _load_sample(self, idx):
        """
        按需加载单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            加载并处理好的样本数据
        """
        import h5py
        
        # 获取样本信息
        sample_info = self.sample_info[idx]
        
        # 打开h5文件并读取数据
        with h5py.File(self.h5_file_path, 'r') as f:
            data = f[sample_info['data_path']][:]
        
        # 截取或填充到指定长度
        if len(data) >= self.sample_length:
            # 从中间截取样本长度
            start = (len(data) - self.sample_length) // 2
            signal = data[start:start + self.sample_length]
        else:
            signal = np.zeros(self.sample_length, dtype=np.float32)
            signal[:len(data)] = data
        
        # 获取标签
        label = self.labels[idx]
        
        return signal.astype(np.float32), label
    
    def __len__(self):
        return len(self.sample_info)
    
    def __getitem__(self, idx):
        # 按需加载样本
        signal, label = self._load_sample(idx)
        
        # 重塑信号以匹配模型输入 (1, 8000, 1)
        signal = signal.reshape(1, self.sample_length, 1)
        
        # 转换为torch张量
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor

# 添加数据预处理的collate_fn，用于在batch级别进行归一化
def collate_fn(batch):
    """
    自定义collate函数，用于batch级别的数据处理
    
    参数:
        batch: 包含(signal, label)元组的列表
        
    返回:
        处理后的batch数据
    """
    # 分离信号和标签
    signals, labels = zip(*batch)
    
    # 堆叠信号和标签
    signals = torch.stack(signals)
    labels = torch.stack(labels)
    
    # 归一化处理（在GPU上进行，减少CPU内存使用）
    # 计算均值和标准差
    mean = signals.mean(dim=[0, 2, 3], keepdim=True)
    std = signals.std(dim=[0, 2, 3], keepdim=True)
    
    # 避免除以零
    std = torch.clamp(std, min=1e-6)
    
    # 归一化
    signals = (signals - mean) / std
    
    return signals, labels

def get_data_loaders_from_h5(h5_file_path, batch_size=128):
    """
    从单个HDF5文件获取训练、验证和测试的数据加载器
    
    参数:
        h5_file_path: HDF5文件路径
        batch_size: 训练批次大小
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 创建数据集
    train_dataset = SingleH5DASDataset(h5_file_path, split='train')
    val_dataset = SingleH5DASDataset(h5_file_path, split='val')
    test_dataset = SingleH5DASDataset(h5_file_path, split='test')
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print(f"测试样本: {len(test_dataset)}")
    
    # 创建数据加载器，使用自定义collate_fn
    # 在Windows上使用num_workers=0避免多进程问题
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
    data_dir = "G:/gitcode/das_small_dataset_N50_20241230_154821.h5"
    train_loader, val_loader, test_loader = get_data_loaders_from_h5(data_dir, batch_size=16)
    
    print(f"训练加载器: {len(train_loader)} 批次")
    print(f"验证加载器: {len(val_loader)} 批次")
    print(f"测试加载器: {len(test_loader)} 批次")
    
    # 检查第一个批次
    for signals, labels in train_loader:
        print(f"信号形状: {signals.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels[:5]}")
        break

if __name__ == "__main__":
    test_data_loader()
