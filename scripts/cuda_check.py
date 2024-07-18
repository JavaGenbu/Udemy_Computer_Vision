import torch
import os

# Verificación de CUDA
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("CUDA Device Name:", torch.cuda.get_device_name(0))

# Verificación de cuDNN
print("cuDNN Version:", torch.backends.cudnn.version())

# Verificación de TensorRT
try:
    import tensorrt as trt
    print("TensorRT Version:", trt.__version__)
    print("TensorRT Library Path:", os.environ['PATH'])
except ImportError as e:
    print("Error importing TensorRT:", e)
except FileNotFoundError as e:
    print("TensorRT library not found:", e)
