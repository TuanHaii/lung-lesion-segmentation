import sys
import torch
import cv2
import albumentations
import streamlit
import numpy as np

def verify_environment():
    print("=" * 50)
    print("KIEM TRA TINH TOAN VEN MOI TRUONG DU AN (ENV-01)")
    print("=" * 50)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version   : {sys.version.split()[0]}")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    print(f"OpenCV Version   : {cv2.__version__}")
    print(f"Albumentations   : {albumentations.__version__}")
    print(f"Streamlit        : {streamlit.__version__}")
    
    # Sanity check tao dummy tensor theo dac ta de tai [Batch, Channel, H, W]
    x = torch.randn(2, 1, 256, 256)
    print(f"Tensor Test Shape: {x.shape} -> Khoi tao thanh cong!")
    print("=" * 50)
    print(">> MOI TRUONG HOAN TAT VA SAN SANG CHO TASK MODEL <<")

if __name__ == "__main__":
    verify_environment()