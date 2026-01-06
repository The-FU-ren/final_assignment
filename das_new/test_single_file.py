import os
import h5py
import numpy as np

# Test loading a single HDF5 file
def test_single_file():
    # Use the first file we found earlier
    file_path = 'G:/DAS-dataset_3/DAS-dataset/data/car/auto2_2023-04-17T124510+0100.h5'
    
    print(f"Testing file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check the data path exists
            data_path = 'Acquisition/Raw[0]/RawData'
            if data_path in f:
                print(f"✓ Found data at {data_path}")
                
                # Read the data
                raw_data = f[data_path][:]
                print(f"✓ Raw data shape: {raw_data.shape}")
                print(f"✓ Raw data dtype: {raw_data.dtype}")
                
                # Process to get 8000-sample signal
                if raw_data.shape[0] >= 8000:
                    signal = raw_data[:8000, 0]
                else:
                    signal = np.zeros(8000, dtype=np.float32)
                    signal[:raw_data.shape[0]] = raw_data[:, 0]
                
                print(f"✓ Processed signal shape: {signal.shape}")
                print(f"✓ Signal min: {signal.min()}, max: {signal.max()}, mean: {signal.mean()}")
                
                return True
            else:
                print(f"✗ Data path {data_path} not found in file")
                return False
                
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return False

if __name__ == "__main__":
    success = test_single_file()
    if success:
        print("\n✓ Single file test passed!")
    else:
        print("\n✗ Single file test failed!")
