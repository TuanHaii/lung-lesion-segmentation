import pandas as pd

for model_name, path in [
    ("Attention U-Net", "reports/attunet/test_metrics_slice_level.csv"),
    ("Standard U-Net", "reports/unet/test_metrics_slice_level.csv"),
    ("SegNet", "reports/segnet/test_metrics_slice_level.csv"),
]:
    df = pd.read_csv(path)
    lesion_df = df[df["lesion_present"] == 1]
    print(f"=== {model_name} ===")
    print(f"Total slices: {len(df)}, Lesion slices: {len(lesion_df)}")
    print(f"Lesion DSC:       {lesion_df['dice'].mean():.4f} +/- {lesion_df['dice'].std():.4f}")
    print(f"Lesion IoU:       {lesion_df['iou'].mean():.4f} +/- {lesion_df['iou'].std():.4f}")
    print(f"Lesion Precision: {lesion_df['precision'].mean():.4f} +/- {lesion_df['precision'].std():.4f}")
    print(f"Lesion Recall:    {lesion_df['recall'].mean():.4f} +/- {lesion_df['recall'].std():.4f}")
    print(f"All Slices DSC:   {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}\n")

