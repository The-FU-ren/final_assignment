@echo off

echo 正在创建DRSN-NTF环境...

rem 1. 检查Anaconda是否已安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo 请先安装Anaconda: https://www.anaconda.com/download
    pause
    exit /b 1
)

rem 2. 创建并激活虚拟环境
conda create -n drsn-ntf python=3.11 -y
conda activate drsn-ntf

rem 3. 安装PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

rem 4. 安装其他依赖
pip install numpy scikit-learn matplotlib seaborn scipy

rem 5. 验证安装
echo 正在验证安装...
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

if %errorlevel% eq 0 (
    echo 环境配置成功！
    echo 使用以下命令激活环境:
    echo conda activate drsn-ntf
    echo 然后运行:
    echo python main.py
) else (
    echo 环境配置失败，请检查错误信息
)

pause
