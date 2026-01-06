import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from glob import glob

class DASDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_length=8000, window_step=4000, max_samples_per_file=100):
        """
        初始化DAS数据集（内存优化版本）
        
        参数:
            data_dir: 数据集目录路径，包含各个类别子目录
            transform: 数据变换
            sample_length: 单个样本的长度
            window_step: 滑动窗口的步长
            max_samples_per_file: 每个文件最多生成的样本数，用于控制内存使用
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sample_length = sample_length
        self.window_step = window_step
        self.max_samples_per_file = max_samples_per_file
        self.label_encoder = LabelEncoder()
        
        # 存储样本元数据，而不是直接加载所有数据
        self.sample_metadata = []
        
        # 预处理阶段：收集样本元数据
        self._collect_sample_metadata()
        
        # 计算并存储类别分布
        self._compute_class_distribution()
        
        # 初始化归一化器（在第一个batch加载时拟合）
        self.scaler = None
    
    def _collect_sample_metadata(self):
        """
        收集样本元数据，不加载实际数据到内存
        """
        import h5py
        from tqdm import tqdm
        
        print(f"从 {self.data_dir} 收集样本元数据")
        
        # 获取所有类别目录
        categories = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        print(f"找到 {len(categories)} 个类别: {categories}")
        
        # 为每个类别编码标签
        self.label_encoder.fit(categories)
        
        # 遍历每个类别目录
        for category in categories:
            category_dir = os.path.join(self.data_dir, category)
            label = self.label_encoder.transform([category])[0]
            
            # 获取该类别下的所有.h5文件
            h5_files = glob(os.path.join(category_dir, "*.h5"))
            print(f"类别 {category} 有 {len(h5_files)} 个.h5文件")
            
            # 处理每个.h5文件
            for h5_file in tqdm(h5_files, desc=f"处理 {category}", unit="文件"):
                try:
                    # 检查对应的.npy文件是否存在
                    npy_file = h5_file[:-2] + "npy"
                    if not os.path.exists(npy_file):
                        print(f"警告: 未找到对应的.npy文件 {npy_file}，跳过该文件")
                        continue
                    
                    # 只读取文件的基本信息，不加载所有数据
                    with h5py.File(h5_file, 'r') as f:
                        data_shape = f["Acquisition/Raw[0]/RawData"].shape
                    
                    # 加载位图文件，获取事件位置
                    bitmap = np.load(npy_file)
                    event_positions = np.transpose(np.where(bitmap))
                    
                    # 如果没有事件位置，生成一些位置
                    if len(event_positions) == 0:
                        num_windows = (data_shape[0] - self.sample_length) // self.window_step + 1
                        event_positions = np.array([(i, 0) for i in range(num_windows)])
                    
                    # 限制每个文件的样本数
                    event_positions = event_positions[:self.max_samples_per_file]
                    
                    # 存储元数据：(h5_file, event_positions, label)
                    for pos_idx, channel in event_positions:
                        # 计算窗口起始位置
                        start_time = pos_idx * self.window_step
                        end_time = start_time + self.sample_length
                        
                        # 确保窗口不超出数据范围
                        if end_time <= data_shape[0]:
                            self.sample_metadata.append({
                                'h5_file': h5_file,
                                'start_time': start_time,
                                'end_time': end_time,
                                'channel': channel,
                                'label': label
                            })
                    
                except Exception as e:
                    print(f"处理文件 {h5_file} 时出错: {e}")
                    continue
        
        print(f"成功收集 {len(self.sample_metadata)} 个样本的元数据，{len(np.unique([meta['label'] for meta in self.sample_metadata]))} 个类别")
    
    def _compute_class_distribution(self):
        """
        计算类别分布
        """
        labels = [meta['label'] for meta in self.sample_metadata]
        label_counts = np.bincount(labels, minlength=len(self.label_encoder.classes_))
        self.class_distribution = dict(zip(self.label_encoder.classes_, label_counts))
        print(f"类别分布: {self.class_distribution}")
    
    def _load_sample(self, idx):
        """
        按需加载单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            加载并处理好的样本数据
        """
        import h5py
        
        # 获取样本元数据
        meta = self.sample_metadata[idx]
        
        # 打开h5文件并读取指定窗口的数据
        with h5py.File(meta['h5_file'], 'r') as f:
            data = f["Acquisition/Raw[0]/RawData"][meta['start_time']:meta['end_time'], meta['channel']]
        
        return data.astype(np.float32), meta['label']
    
    def __len__(self):
        return len(self.sample_metadata)
    
    def __getitem__(self, idx):
        # 按需加载样本
        signal, label = self._load_sample(idx)
        
        # 重塑信号以匹配模型输入 (1, 8000, 1)
        signal = signal.reshape(1, self.sample_length, 1)
        
        # 确保标签值在有效范围内
        max_label = len(self.label_encoder.classes_) - 1
        if label < 0 or label > max_label:
            print(f"警告: 标签值 {label} 超出范围 [0, {max_label}]，将其截断")
            label = max(0, min(label, max_label))
        
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

def get_data_loaders(data_dir, batch_size=128, val_split=0.1, test_split=0.2, max_samples_per_file=100):
    """
    获取训练、验证和测试的数据加载器
    
    参数:
        data_dir: 数据集目录路径，包含各个类别子目录
        batch_size: 训练批次大小
        val_split: 验证集划分比例
        test_split: 测试集划分比例
        max_samples_per_file: 每个文件最多生成的样本数，用于控制内存使用
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 创建完整数据集（使用内存优化版本）
    full_dataset = DASDataset(data_dir, max_samples_per_file=max_samples_per_file)
    
    # 计算划分大小
    total_size = len(full_dataset)
    
    # 确保每个集合至少有1个样本
    test_size = max(1, int(total_size * test_split))
    val_size = max(1, int(total_size * val_split))
    train_size = max(1, total_size - test_size - val_size)
    
    # 如果大小不匹配，进行调整
    if train_size + val_size + test_size != total_size:
        diff = total_size - (train_size + val_size + test_size)
        train_size += diff
    
    print(f"\n数据集划分: {train_size} 训练样本, {val_size} 验证样本, {test_size} 测试样本")
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器，使用自定义collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

# 示例用法
def test_data_loader():
    data_dir = "G:/DAS-dataset_3/DAS-dataset/data"
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=16)
    
    print(f"训练加载器: {len(train_loader)} 批次")
    print(f"验证加载器: {len(val_loader)} 批次")
    print(f"测试加载器: {len(test_loader)} 批次")
    
    # 检查第一个批次
    for signals, labels in train_loader:
        print(f"信号形状: {signals.shape}")
        print(f"标签形状: {labels.shape}")
        break

if __name__ == "__main__":
    test_data_loader()
