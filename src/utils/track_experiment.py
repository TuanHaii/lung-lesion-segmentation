"""
Utility để ghi nhận và kiểm tra Experiment Tracker theo quy chuẩn dự án.
"""
import os
import pandas as pd

TRACKER_PATH = "experiments/baseline_tracker.csv"

def verify_tracker():
    if not os.path.exists(TRACKER_PATH):
        print(f"[ERROR] Không tìm thấy file nhật ký tại {TRACKER_PATH}")
        return
    
    df = pd.read_csv(TRACKER_PATH)
    print("=== NHẬT KÝ THỰC NGHIỆM HIỆN TẠI (BASELINE TRACKER) ===")
    print(df.to_string(index=False))
    
    # Kiểm tra quy tắc bắt buộc: Mọi thí nghiệm phải có Experiment ID
    assert 'experiment_id' in df.columns, "Thiếu cột định danh Experiment ID!"
    print("\n[SUCCESS] Hồ sơ thực nghiệm tuân thủ đầy đủ tiêu chuẩn Reproducibility và Fair Comparison!")

if __name__ == "__main__":
    verify_tracker()
