import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    print("\nTesting tensor creation on GPU:")
    try:
        # Create a tensor directly on GPU
        x = torch.tensor([1, 2, 3], device='cuda')
        print(f"Tensor created on {x.device}")
        print(f"Tensor content: {x}")
        print("GPU check successful!")
    except Exception as e:
        print(f"Error creating tensor on GPU: {e}")
else:
    print("CUDA is not available. PyTorch will use CPU only.")
    print("Possible issues:")
    print("1. NVIDIA GPU drivers not installed or outdated")
    print("2. CUDA toolkit not installed or not compatible with PyTorch")
    print("3. PyTorch installed without CUDA support") 