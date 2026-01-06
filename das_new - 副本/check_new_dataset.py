import h5py
import numpy as np

# 新数据集路径
new_dataset_path = r"G:\gitcode\das_small_dataset_N50_20241230_154821.h5"

try:
    with h5py.File(new_dataset_path, 'r') as f:
        print("=== HDF5文件结构 ===")
        def print_structure(name, obj):
            print(name)
            if isinstance(obj, h5py.Dataset):
                print(f"   类型: Dataset")
                print(f"   形状: {obj.shape}")
                print(f"   数据类型: {obj.dtype}")
                print(f"   属性: {list(obj.attrs.keys())}")
        f.visititems(print_structure)
        
        # 尝试读取一些数据样本
        print("\n=== 数据样本信息 ===")
        # 假设数据存储在某个路径下，先查看根目录下的所有键
        print(f"根目录键: {list(f.keys())}")
        
except Exception as e:
    print(f"读取文件出错: {e}")
