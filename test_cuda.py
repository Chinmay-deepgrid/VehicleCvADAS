import torch
import onnxruntime

print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("ORT providers:", onnxruntime.get_available_providers())