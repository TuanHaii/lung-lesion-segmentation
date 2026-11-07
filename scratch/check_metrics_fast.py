import csv
import math

def analyze(model_name, path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    lesion_rows = [r for r in rows if int(r["lesion_present"]) == 1]
    
    def mean_std(values):
        m = sum(values) / len(values)
        v = sum((x - m) ** 2 for x in values) / len(values)
        return m, math.sqrt(v)
    
    dice_l = [float(r["dice"]) for r in lesion_rows]
    iou_l = [float(r["iou"]) for r in lesion_rows]
    prec_l = [float(r["precision"]) for r in lesion_rows]
    rec_l = [float(r["recall"]) for r in lesion_rows]
    all_dice = [float(r["dice"]) for r in rows]
    
    m_d, s_d = mean_std(dice_l)
    m_i, s_i = mean_std(iou_l)
    m_p, s_p = mean_std(prec_l)
    m_r, s_r = mean_std(rec_l)
    m_ad, s_ad = mean_std(all_dice)
    
    print(f"=== {model_name} ===")
    print(f"Total: {len(rows)} | Lesion slices (GT > 0): {len(lesion_rows)}")
    print(f"Lesion DSC:       {m_d:.4f} +/- {s_d:.4f}")
    print(f"Lesion IoU:       {m_i:.4f} +/- {s_i:.4f}")
    print(f"Lesion Precision: {m_p:.4f} +/- {s_p:.4f}")
    print(f"Lesion Recall:    {m_r:.4f} +/- {s_r:.4f}")
    print(f"All Slices DSC:   {m_ad:.4f} +/- {s_ad:.4f}\n")

analyze("Attention U-Net", "reports/attunet/test_metrics_slice_level.csv")
analyze("Standard U-Net", "reports/unet/test_metrics_slice_level.csv")
analyze("SegNet", "reports/segnet/test_metrics_slice_level.csv")

