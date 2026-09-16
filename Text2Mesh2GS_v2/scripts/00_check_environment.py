import sys
import torch
import subprocess

def check_python():
    print("Python:", sys.version)

def check_torch():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("Capability:", torch.cuda.get_device_capability(0))

def check_blender():
    try:
        result = subprocess.run(
            ["blender", "--version"],
            capture_output=True,
            text=True
        )
        print(result.stdout.split("\n")[0])
    except FileNotFoundError:
        print("Blender command not found.")

if __name__ == "__main__":
    check_python()
    check_torch()
    check_blender()