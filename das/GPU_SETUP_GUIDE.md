# GPU 加速配置指南

## 问题分析

当前PyTorch CUDA版本安装遇到了DLL加载问题，这通常是由于CUDA版本不匹配或缺少必要的系统依赖导致的。

## 系统环境检查

从 `nvidia-smi` 输出可以看到：
- GPU: NVIDIA GeForce RTX 4060
- 驱动版本: 576.52
- CUDA版本: 12.9

## 解决方案

### 方法1：使用Anaconda（推荐）

1. **安装Anaconda**
   - 下载并安装Anaconda：https://www.anaconda.com/download
   - 选择与您系统匹配的版本（Windows 64-bit）

2. **创建虚拟环境**
   ```bash
   conda create -n drsn-ntf python=3.11
   conda activate drsn-ntf
   ```

3. **安装PyTorch CUDA版本**
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. **验证安装**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### 方法2：使用pip（需要手动配置CUDA）

1. **安装CUDA Toolkit**
   - 下载并安装CUDA Toolkit 12.1：https://developer.nvidia.com/cuda-12-1-0-download-archive
   - 选择自定义安装，仅安装必要组件

2. **安装cuDNN**
   - 下载cuDNN 8.9.x for CUDA 12.x：https://developer.nvidia.com/cudnn-download
   - 解压并将文件复制到CUDA安装目录

3. **安装PyTorch**
   ```bash
   pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **设置环境变量**
   - 将CUDA安装目录添加到系统PATH
   - 例如：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`

### 方法3：使用Docker（高级用户）

1. **安装Docker**：https://www.docker.com/products/docker-desktop
2. **拉取PyTorch CUDA镜像**：
   ```bash
   docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime
   ```
3. **运行容器**：
   ```bash
   docker run --gpus all -v g:/gitcode/final_assignment/das:/workspace -it pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime
   ```

## 代码验证

当PyTorch CUDA版本安装成功后，运行以下命令验证：

```bash
python cuda_check.py
```

预期输出：
```
PyTorch版本: 2.4.1+cu121
CUDA可用: True
GPU设备: NVIDIA GeForce RTX 4060
GPU数量: 1
```

## 模型运行

### 1. 使用GPU训练

代码已经默认支持GPU加速，当CUDA可用时会自动使用GPU。

### 2. 手动指定设备

在 `train.py` 中，设备选择逻辑如下：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3. 验证模型使用GPU

运行模型测试：
```bash
python model.py
```

预期输出：
```
Input shape: torch.Size([1, 1, 8000])
Output shape: torch.Size([1, 6])
Model created successfully!
```

## 性能优化建议

1. **使用更大的批量大小**：在GPU上可以使用更大的批量大小，如256或512
2. **启用混合精度训练**：在 `train.py` 中添加：
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   # 在训练循环中
   with autocast():
       outputs = model(inputs)
       loss = criterion(outputs, labels)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```
3. **使用DataLoader多进程**：在 `data_processor.py` 中设置 `num_workers > 0`

## 常见问题排查

### 1. DLL加载失败
- 确保CUDA版本与PyTorch版本匹配
- 重新安装CUDA Toolkit和cuDNN
- 检查系统PATH中是否包含CUDA bin目录

### 2. CUDA不可用
- 运行 `nvidia-smi` 检查GPU驱动是否正常
- 确保PyTorch安装了CUDA版本
- 重启计算机后再次尝试

### 3. 内存不足
- 减少批量大小
- 使用梯度累积
- 启用梯度检查点

## 替代方案

如果GPU加速配置困难，可以先使用CPU版本进行开发和测试，然后在具有正确CUDA环境的服务器上进行正式训练。

## 联系方式

如果遇到无法解决的问题，可以参考PyTorch官方文档：
- https://pytorch.org/get-started/locally/
- https://pytorch.org/docs/stable/cuda.html
