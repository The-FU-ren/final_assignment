import torch
import torch.nn as nn
from model import DRSN_NTF

# 检查CUDA是否可用
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # 创建一个简单的模型并移动到GPU
    model = DRSN_NTF(num_classes=6)
    model = model.to('cuda')
    
    # 创建一个输入张量并移动到GPU
    input_tensor = torch.randn(1, 1, 8000).to('cuda')
    
    # 前向传播
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("Model ran successfully on GPU!")
else:
    print("CUDA is not available, model will run on CPU.")
    
    # 创建一个简单的模型并在CPU上运行
    model = DRSN_NTF(num_classes=6)
    
    # 创建一个输入张量
    input_tensor = torch.randn(1, 1, 8000)
    
    # 前向传播
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("Model ran successfully on CPU!")
