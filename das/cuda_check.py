# 最简单的CUDA检查脚本
try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU设备: {torch.cuda.get_device_name(0)}')
        print(f'GPU数量: {torch.cuda.device_count()}')
except Exception as e:
    print(f'错误: {e}')
