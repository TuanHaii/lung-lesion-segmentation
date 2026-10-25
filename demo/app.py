"""
Task ID: DEMO-02 (Interactive Streamlit Web Interface)
Mô tả: Giao diện chẩn đoán tương tác phân đoạn tổn thương CT phổi:
- Chẩn đoán lát cắt CT (Single Slice Diagnosis).
- So sánh định tính đa mô hình (Attention U-Net vs U-Net vs SegNet).
- Báo cáo định lượng y khoa: diện tích tổn thương, tỷ lệ gánh nặng phổi (Burden %), mức độ nghiêm trọng.
- Trực quan hóa dữ liệu thực nghiệm (Learning curves, Comparison benchmarks).
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import streamlit as st
except ImportError:
    st = None

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from demo.inference_engine import (
    load_model,
    preprocess_image,
    run_inference,
    calculate_clinical_metrics,
    CT_WINDOWS,
)
from demo.visualization_tools import (
    create_overlay,
    create_probability_heatmap,
    create_quad_comparison_figure,
    create_dual_model_figure,
)
from src.utils.visualization import generate_mock_ct_and_mask

# Tùy chọn đọc NIfTI nếu có
try:
    import nibabel as nib
except ImportError:
    nib = None

# Tùy chọn đọc file Excel/CSV với pandas
try:
    import pandas as pd  # noqa: F811
except ImportError:
    pass


def inject_custom_css():
    """Tạo giao diện hiện đại, chuyên nghiệp theo phong cách AI Y tế."""
    st.markdown("""
        <style>
            .main-header {
                font-size: 2.2rem;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 0.2rem;
            }
            .sub-header {
                font-size: 1.05rem;
                color: #475569;
                margin-bottom: 1.5rem;
            }
            .metric-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .metric-val {
                font-size: 1.8rem;
                font-weight: 700;
                color: #1e293b;
            }
            .metric-lbl {
                font-size: 0.85rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .badge-proposed {
                background-color: #fee2e2;
                color: #b91c1c;
                padding: 4px 10px;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.85rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 20px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 48px;
                font-weight: 600;
                font-size: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)


def get_default_checkpoint(model_name: str) -> Optional[str]:
    """Tìm đường dẫn checkpoint tương ứng với mô hình."""
    mapping = {
        "Attention U-Net (Đề xuất)": [
            "checkpoints/EXP-ATTUNET-001_best.pth",
            "checkpoints/EXP-ATTUNET-001_last.pth",
            "checkpoints/best_model.pth"
        ],
        "Standard U-Net (Baseline)": [
            "checkpoints/EXP-UNET-001_best.pth",
            "checkpoints/EXP-UNET-001_last.pth",
        ],
        "SegNet (Baseline)": [
            "checkpoints/EXP-SEGNET-001_best.pth",
            "checkpoints/EXP-SEGNET-001_last.pth",
        ]
    }
    candidates = mapping.get(model_name, [])
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    if st is None:
        print("[!] Vui lòng cài đặt streamlit: pip install streamlit")
        return

    st.set_page_config(
        page_title="AI Phân đoạn Tổn thương CT Phổi",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()

    # ------------------ SIDEBAR ------------------
    st.sidebar.image("https://img.icons8.com/fluency/96/lungs.png", width=70)
    st.sidebar.markdown("### **Hệ Thống Phân Đoạn CT Phổi**")
    st.sidebar.caption("Đề tài: Phân đoạn tổn thương COVID-19 / Viêm phổi bằng Attention U-Net")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### **1. Cấu hình Mô hình**")

    model_options = [
        "Attention U-Net (Đề xuất)",
        "Standard U-Net (Baseline)",
        "SegNet (Baseline)",
    ]
    selected_model_name = st.sidebar.selectbox("Mô hình suy luận:", model_options, index=0)

    # Tự động nhận diện checkpoint
    default_ckpt = get_default_checkpoint(selected_model_name)
    ckpt_path = st.sidebar.text_input("Đường dẫn Checkpoint (.pth):", value=default_ckpt or "")

    has_ckpt = os.path.exists(ckpt_path) if ckpt_path else False
    if has_ckpt:
        st.sidebar.success(f"✓ Đã nạp checkpoint: {Path(ckpt_path).name}")
    else:
        st.sidebar.warning("⚠ Chưa tìm thấy file checkpoint. Hệ thống sẽ khởi tạo kiến trúc với trọng số mặc định.")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Phần cứng: **{device.upper()}** ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### **2. Tham số Y khoa (Clinical Tuning)**")

    window_name = st.sidebar.selectbox(
        "Cửa sổ hiển thị CT (HU Window):",
        options=["lung", "mediastinum", "bone", "raw"],
        format_func=lambda x: {
            "lung": "Cửa sổ Phổi [-1000, 400] HU",
            "mediastinum": "Trung thất [-160, 240] HU",
            "bone": "Xương [-200, 1000] HU",
            "raw": "Nguyên bản (Min - Max)"
        }.get(x, x),
        index=0
    )

    threshold = st.sidebar.slider(
        "Ngưỡng nhị phân (Probability Threshold):",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Xác suất từ ngưỡng này trở lên sẽ được phân loại là vùng tổn thương"
    )

    overlay_alpha = st.sidebar.slider(
        "Độ mờ lớp phủ màu (Overlay Alpha):",
        min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )

    # ------------------ MAIN CONTENT ------------------
    st.markdown('<div class="main-header">🫁 Hệ Thống Hỗ Trợ Chẩn Đoán Phân Đoạn Tổn Thương CT Phổi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mô hình đề xuất: <b>Attention U-Net</b> đối chuẩn với Standard U-Net & SegNet trên dữ liệu CT lồng ngực.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🔬 Chẩn đoán Lát cắt Đơn (Single Slice)",
        "⚖️ So sánh Đa mô hình (Model Comparison)",
        "📊 Bảng số liệu Nghiên cứu & Benchmark"
    ])

    # ================= TAB 1 =================
    with tab1:
        c_src1, c_src2 = st.columns([1, 2])
        with c_src1:
            input_source = st.radio(
                "Nguồn dữ liệu CT đầu vào:",
                ["Ca bệnh mẫu (Synthetic Clinical Case)", "Tải lên file (Upload .nii / .png / .jpg)"],
                index=0
            )

        ct_slice = None
        gt_slice = None

        if input_source == "Ca bệnh mẫu (Synthetic Clinical Case)":
            if "mock_vol" not in st.session_state:
                with st.spinner("Đang tạo thể tích CT mẫu 3D..."):
                    m_ct, m_mask = generate_mock_ct_and_mask(shape=(256, 256, 40))
                    st.session_state.mock_ct = m_ct
                    st.session_state.mock_mask = m_mask

            total_z = st.session_state.mock_ct.shape[2]
            # Tìm lát cắt có tổn thương nhiều nhất làm mặc định
            lesion_per_slice = [np.sum(st.session_state.mock_mask[:, :, z] > 0) for z in range(total_z)]
            best_z = int(np.argmax(lesion_per_slice))

            slice_idx = st.slider(f"Chọn lát cắt theo trục dọc (Z-axis, tổng số: {total_z} slices):", 0, total_z - 1, value=best_z)
            ct_slice = st.session_state.mock_ct[:, :, slice_idx]
            gt_slice = st.session_state.mock_mask[:, :, slice_idx]

        else:
            uploaded_file = st.file_uploader("Chọn file lát cắt CT (.nii, .nii.gz, .png, .jpg):", type=["nii", "gz", "png", "jpg", "jpeg"])
            if uploaded_file is not None:
                fname = uploaded_file.name.lower()
                if "nii" in fname:
                    if nib is None:
                        st.error("Cần cài đặt thư viện 'nibabel' để đọc file NIfTI: pip install nibabel")
                    else:
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        nii_obj = nib.load(tmp_path)
                        vol_data = nii_obj.get_fdata(dtype=np.float32)
                        os.unlink(tmp_path)

                        if vol_data.ndim == 3:
                            z_max = vol_data.shape[2]
                            z_idx = st.slider("Chọn lát cắt Z:", 0, z_max - 1, z_max // 2)
                            ct_slice = vol_data[:, :, z_idx]
                        elif vol_data.ndim == 2:
                            ct_slice = vol_data
                else:
                    pil_img = Image.open(uploaded_file).convert("L")
                    ct_slice = np.array(pil_img, dtype=np.float32)

        if ct_slice is not None:
            # Tiền xử lý
            input_tensor, ct_norm = preprocess_image(
                ct_slice,
                target_size=(256, 256),
                window_name=window_name
            )

            # Khởi tạo mô hình
            with st.spinner(f"Đang nạp mô hình {selected_model_name}..."):
                model, _ = load_model(selected_model_name, checkpoint_path=ckpt_path, device=device)

            # Suy luận
            with st.spinner("Đang thực hiện phân đoạn tổn thương (Inference)..."):
                result = run_inference(model, input_tensor, threshold=threshold, device=device)

            prob_map = result["probability_map"]
            pred_mask = result["binary_mask"]
            latency_ms = result["latency_ms"]

            # Tính toán chỉ số lâm sàng
            stats = calculate_clinical_metrics(pred_mask, ct_norm)

            st.markdown("#### **Kết Quả Phân Đoạn Trực Quan**")
            fig = create_quad_comparison_figure(
                ct_norm=ct_norm,
                prob_map=prob_map,
                pred_mask=pred_mask,
                gt_mask=gt_slice,
                alpha=overlay_alpha,
                title=f"Chẩn đoán: {selected_model_name} | Ngưỡng: {threshold:.2f}"
            )
            st.pyplot(fig)
            plt.close(fig)

            # Thẻ số liệu định lượng lâm sàng
            st.markdown("#### **Chỉ Số Định Lượng Lâm Sàng (Clinical Quantification)**")
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Diện Tích Tổn Thương</div>
                    <div class="metric-val" style="color: #ef4444;">{stats['lesion_pixels']} <span style="font-size: 1rem;">px</span></div>
                    <div style="font-size: 0.8rem; color: #64748b;">~ {stats['lesion_area_cm2']:.2f} cm²</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Tỷ Lệ Gánh Nặng Phổi (Burden)</div>
                    <div class="metric-val" style="color: #f97316;">{stats['lung_burden_percent']:.1f}%</div>
                    <div style="font-size: 0.8rem; color: #64748b;">(Tổn thương / Nhu mô phổi)</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                sev_color = "#22c55e" if "Nhẹ" in stats['severity'] else ("#f59e0b" if "Trung bình" in stats['severity'] else "#ef4444")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Mức Độ Nghiêm Trọng</div>
                    <div class="metric-val" style="color: {sev_color}; font-size: 1.3rem;">{stats['severity']}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">Theo phân loại lâm sàng</div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Thời Gian Suy Luận</div>
                    <div class="metric-val" style="color: #3b82f6;">{latency_ms:.1f} <span style="font-size: 1rem;">ms</span></div>
                    <div style="font-size: 0.8rem; color: #64748b;">Xử lý 1 slice trên {device.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

    # ================= TAB 2 =================
    with tab2:
        st.markdown("#### **So Sánh Trực Tiếp Định Tính (Qualitative Comparison)**")
        st.caption("Quan sát khả năng ức chế cảnh báo nhầm (False Positives) của Attention Gate so với mạng U-Net thông thường.")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            m_a = st.selectbox("Mô hình A (Thường là Đề xuất):", model_options, index=0)
        with col_m2:
            m_b = st.selectbox("Mô hình B (Baseline đối chuẩn):", model_options, index=1)

        if ct_slice is not None:
            ckpt_a = get_default_checkpoint(m_a)
            ckpt_b = get_default_checkpoint(m_b)

            with st.spinner("Đang chạy dự đoán song song cả 2 mô hình..."):
                model_a, _ = load_model(m_a, checkpoint_path=ckpt_a, device=device)
                model_b, _ = load_model(m_b, checkpoint_path=ckpt_b, device=device)

                res_a = run_inference(model_a, input_tensor, threshold=threshold, device=device)
                res_b = run_inference(model_b, input_tensor, threshold=threshold, device=device)

            fig_dual = create_dual_model_figure(
                ct_norm=ct_norm,
                mask_a=res_a["binary_mask"],
                mask_b=res_b["binary_mask"],
                name_a=m_a,
                name_b=m_b,
                gt_mask=gt_slice
            )
            st.pyplot(fig_dual)
            plt.close(fig_dual)

            c_info1, c_info2 = st.columns(2)
            with c_info1:
                st.info(f"**{m_a}**: Số điểm ảnh phát hiện: **{int(np.sum(res_a['binary_mask']))} px** | Độ trễ: **{res_a['latency_ms']:.1f} ms**")
            with c_info2:
                st.info(f"**{m_b}**: Số điểm ảnh phát hiện: **{int(np.sum(res_b['binary_mask']))} px** | Độ trễ: **{res_b['latency_ms']:.1f} ms**")

    # ================= TAB 3 =================
    with tab3:
        st.markdown("#### **Bảng Số Liệu Đánh Giá Định Lượng (Sprint W5 Benchmarks)**")

        summary_csv = "reports/model_comparison_summary.csv"
        if os.path.exists(summary_csv):
            import pandas as pd
            df = pd.read_csv(summary_csv)
            st.dataframe(df, use_container_width=True)
            st.caption("Bảng dữ liệu trích xuất từ các báo cáo đánh giá trên tập Test độc lập.")
        else:
            st.warning("Chưa có file 'reports/model_comparison_summary.csv'. Bạn hãy chạy script `python scripts/compare_models.py` để sinh bảng số liệu tự động.")

        # Hiển thị đồ thị huấn luyện nếu có
        st.markdown("---")
        st.markdown("#### **Đồ Thị Tiến Trình Huấn Luyện (Learning Curves)**")
        log_files = {
            "Attention U-Net": "logs/EXP-ATTUNET-001_training.log",
            "Standard U-Net": "logs/EXP-UNET-001_training.log",
            "SegNet": "logs/EXP-SEGNET-001_training.log",
        }

        fig_curves, ax_c = plt.subplots(1, 2, figsize=(14, 5))
        has_logs = False
        import pandas as pd
        for name, lpath in log_files.items():
            if os.path.exists(lpath):
                try:
                    ldf = pd.read_csv(lpath)
                    ax_c[0].plot(ldf["epoch"], ldf["val_loss"], label=f"{name} Val Loss", lw=2)
                    ax_c[1].plot(ldf["epoch"], ldf["val_dice"], label=f"{name} Val Dice", lw=2)
                    has_logs = True
                except Exception:
                    pass

        if has_logs:
            ax_c[0].set_title("Validation Loss Progression", fontweight="bold")
            ax_c[0].set_xlabel("Epoch")
            ax_c[0].grid(True, linestyle="--", alpha=0.6)
            ax_c[0].legend()

            ax_c[1].set_title("Validation Dice Metric Progression", fontweight="bold")
            ax_c[1].set_xlabel("Epoch")
            ax_c[1].grid(True, linestyle="--", alpha=0.6)
            ax_c[1].legend()

            plt.tight_layout()
            st.pyplot(fig_curves)
            plt.close(fig_curves)
        else:
            st.info("Chưa tìm thấy file logs trong thư mục `logs/`. Khi bạn tải thư mục `logs/` từ Kaggle về, các đường cong hội tụ Loss và Dice sẽ tự động hiển thị ở đây.")


if __name__ == "__main__":
    main()
