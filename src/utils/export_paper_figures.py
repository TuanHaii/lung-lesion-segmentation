"""
Task ID: PAPER-W7 (Sprint W7 - Scientific Paper)
Mục đích: Tập hợp và chuẩn hóa tên các biểu đồ/hình ảnh thực nghiệm đưa vào bài báo.
Tác giả: Hậu (Baseline & Experiment)
"""
import os
import shutil

def export_figures():
    os.makedirs("paper_figures", exist_ok=True)
    
    # Mapping các file ảnh gốc sang tên chuẩn phục vụ chèn vào bản thảo Paper
    figures_map = {
        "plots/learning_curves_comparison.png": "paper_figures/Fig1_Learning_Curves.png",
        "plots/qualitative_comparison_grid.png": "paper_figures/Fig2_Qualitative_Comparison.png",
        "plots/rq2_boundary_comparison.png": "paper_figures/Fig3_RQ2_Boundary_Analysis.png",
        "evidence/W2_DATA-11_single_inspection.png": "paper_figures/Fig4_CT_Preprocessing_Windowing.png"
    }
    
    print("[INFO] Đang đóng gói hình ảnh chuẩn bị cho bài báo khoa học (W7)...")
    for src, dst in figures_map.items():
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f" [OK] Đã trích xuất: {dst}")
        else:
            print(f" [WARN] Không tìm thấy ảnh gốc: {src}")

if __name__ == "__main__":
    export_figures()
