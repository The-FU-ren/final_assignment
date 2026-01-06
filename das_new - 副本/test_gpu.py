import torch
import numpy as np

# Test GPU availability
print("=== GPU Test ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"GPU matrix multiplication successful: {z.shape}")
else:
    print("CUDA not available, using CPU")

print("\n=== Data Loading Test ===")
from data_loader import DASDataset

# Test loading a single file
try:
    dataset = DASDataset('G:/DAS-dataset_3/DAS-dataset')
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test getting a sample
        signal, label = dataset[0]
        print(f"Sample signal shape: {signal.shape}")
        print(f"Sample label: {label}")
        print("Data loading test passed!")
    else:
        print("Dataset is empty")
except Exception as e:
    print(f"Data loading error: {e}")

print("\n=== Model Forward Test ===")
from model import DRSN_NTF

# Test model forward pass
try:
    model = DRSN_NTF(num_classes=6)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create a dummy input
    dummy_input = torch.randn(2, 1, 8000, 1).to(device)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print("Model forward test passed!")
except Exception as e:
    print(f"Model forward error: {e}")
