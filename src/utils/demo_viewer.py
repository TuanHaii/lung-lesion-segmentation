"""
Task ID: DEMO-W6 (Sprint W6 - Web Demo Integration)
Mục đích: Module hỗ trợ trực quan hóa kết quả phân đoạn trên giao diện Web (Streamlit).
Tác giả: Hậu (Baseline & Experiment) - Phối hợp bàn giao cho Hải (Tech Lead).
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.utils.visualization import apply_windowing

def render_segmentation_results(ct_slice: np.ndarray, pred_mask: np.ndarray, window_name: str = "lung"):
    """
    Hiển thị kết quả 3 cột trên Web Demo: CT Input -> Mask Dự đoán -> Overlay.
    """
    st.markdown("### Kết quả Phân đoạn Tổn thương (Segmentation Results)")
    
    # Tiền xử lý cửa sổ HU chuẩn
    ct_norm = apply_windowing(ct_slice, window_name=window_name)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(ct_norm, caption=f"1. Ảnh CT gốc ({window_name.upper()} Window)", use_column_width=True, clamp=True)
        
    with col2:
        st.image(pred_mask * 255, caption="2. Mặt nạ Dự đoán (Prediction)", use_column_width=True, clamp=True)
        
    with col3:
        fig, ax = plt.subplots()
        ax.imshow(ct_norm, cmap="gray")
        masked = np.ma.masked_where(pred_mask == 0, pred_mask)
        ax.imshow(masked, cmap="autumn", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)
        st.caption("3. Lớp phủ (Overlay Mask)")
