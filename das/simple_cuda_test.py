import torch
print('Testing PyTorch CUDA availability...')
cuda_available = torch.cuda.is_available()
print(f'CUDA available: {cuda_available}')
if cuda_available:
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device: {torch.cuda.current_device()}')
else:
    print('CUDA is not available. Using CPU.')
