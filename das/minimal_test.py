# 最小化测试脚本
print('Testing basic Python functionality...')

# 测试1: 基本导入
import sys
print(f'Python version: {sys.version}')

# 测试2: 尝试导入numpy（如果已安装）
try:
    import numpy as np
    print(f'NumPy version: {np.__version__}')
except ImportError:
    print('NumPy not installed')

# 测试3: 尝试导入torch
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    
    # 测试4: 检查torch模块内容
    print('Torch attributes:', [attr for attr in dir(torch) if not attr.startswith('_')])
    
    # 测试5: 检查CUDA是否可用
    if hasattr(torch, 'cuda'):
        print('CUDA module available')
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'Device count: {torch.cuda.device_count()}')
            print(f'Device name: {torch.cuda.get_device_name(0)}')
    else:
        print('CUDA module not available')
except Exception as e:
    print(f'Error with PyTorch: {e}')
    import traceback
    traceback.print_exc()
