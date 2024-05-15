import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA is not available")
