import os
import h5py

# Directory containing the DAS dataset
data_dir = 'G:/DAS-dataset_3/DAS-dataset'

# Find all HDF5 files
print(f"Searching for HDF5 files in: {data_dir}")
h5_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.h5') or file.endswith('.hdf5'):
            h5_files.append(os.path.join(root, file))

print(f"Found {len(h5_files)} HDF5 files")

# Check the structure of the first few files
for i, file_path in enumerate(h5_files[:3]):  # Check first 3 files
    print(f"\n=== File {i+1}: {os.path.basename(file_path)} ===")
    print(f"Path: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("\nFile structure:")
            
            def print_group(group, indent=0):
                """Recursively print all groups and datasets"""
                for key in group.keys():
                    item = group[key]
                    prefix = "  " * indent
                    print(f"{prefix}- {key}")
                    
                    if isinstance(item, h5py.Group):
                        print(f"{prefix}  (Group)")
                        print_group(item, indent + 2)
                    else:
                        # It's a dataset
                        print(f"{prefix}  (Dataset)")
                        print(f"{prefix}  Shape: {item.shape}")
                        print(f"{prefix}  Dtype: {item.dtype}")
                        print(f"{prefix}  Size: {item.size} elements")
            
            print_group(f)
            
    except Exception as e:
        print(f"Error reading file: {e}")
