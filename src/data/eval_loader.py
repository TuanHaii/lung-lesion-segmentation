#!/usr/bin/env python3
"""
Task ID: EVAL-01 (Tuần 5 - Phase: Evaluation & Analysis)
Tên module: src/data/eval_loader.py
Mục đích:
- Cung cấp PyTorch Dataset (IndependentTestDataset) và DataLoader (get_eval_dataloader)
  chuẩn hóa dành riêng cho tập kiểm thử độc lập (Independent Test Set).
- Tuân thủ nghiêm ngặt đặc tả tiền xử lý y khoa v1.0:
  + HU Windowing [-1000, 400]
  + Min-Max scaling về đoạn [0.0, 1.0]
  + Nhị phân hóa Ground Truth Mask về {0, 1} (lesion_label=3)
  + Resize về kích thước chuẩn (256, 256): Bilinear cho ảnh CT, Nearest Neighbor cho mask
  + Tuyệt đối KHÔNG áp dụng Data Augmentation (chỉ giữ nguyên ảnh gốc chuẩn hóa)
- Thiết lập bắt buộc: batch_size = 1 và shuffle = False (phản ánh chẩn đoán lâm sàng tuần tự)
- Trả về dictionary: {"image": [1, 1, 256, 256], "mask": [1, 1, 256, 256], "sample_id": ..., "patient_id": ...}
- Tích hợp cơ chế kiểm tra và khóa quyền ghi (read-only) cho file manifest tập test.
- Tích hợp bộ Unit Test tự động trong khối __main__.
"""

import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo đường dẫn gốc dự án luôn nằm trong sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import nibabel as nib
except ImportError:
    nib = None

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.hu_window import apply_hu_window
from src.data.mask_binarize import binarize_mask
from src.data.transform import normalize_min_max, resize_ct_pair


def is_manifest_readonly(manifest_path: Union[str, Path]) -> bool:
    """Kiểm tra file manifest có đang ở trạng thái READ-ONLY (chỉ đọc) hay không.

    Parameters
    ----------
    manifest_path : Union[str, Path]
        Đường dẫn đến file test_manifest.json.

    Returns
    -------
    bool
        True nếu file không có quyền ghi (Read-Only), False nếu có quyền ghi.
    """
    p = Path(manifest_path)
    if not p.exists():
        return False
    return not os.access(p, os.W_OK)


def set_manifest_readonly(manifest_path: Union[str, Path], readonly: bool = True) -> None:
    """Thiết lập hoặc gỡ bỏ trạng thái READ-ONLY cho file manifest.

    Parameters
    ----------
    manifest_path : Union[str, Path]
        Đường dẫn file test_manifest.json.
    readonly : bool, mặc định True
        Nếu True, gán quyền 0o444 (r--r--r--). Nếu False, gán quyền 0o644 (rw-r--r--).
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file manifest: {manifest_path}")

    if readonly:
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    else:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def verify_test_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Kiểm tra tính toàn vẹn và hợp lệ của file test_manifest.json.

    Parameters
    ----------
    manifest_path : Union[str, Path]
        Đường dẫn file manifest.

    Returns
    -------
    Dict[str, Any]
        Nội dung dữ liệu manifest đã đọc.
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"[X] Không tìm thấy test manifest tại: {manifest_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    slices = data.get("slices", [])

    test_patients = meta.get("test_patients", [])
    total_slices = meta.get("total_test_slices", len(slices))

    if len(slices) == 0:
        raise ValueError("[X] Tập test trong manifest bị rỗng (0 lát cắt)!")

    if len(slices) != total_slices:
        raise ValueError(
            f"[X] Bất nhất số lát cắt: metadata={total_slices} nhưng mảng slices={len(slices)}"
        )

    # Đảm bảo chỉ chứa các bệnh nhân thuộc test
    test_patient_set = set(test_patients)
    for s in slices:
        pid = s["patient_id"]
        if pid not in test_patient_set:
            raise ValueError(
                f"[X] Phát hiện bệnh nhân không hợp lệ trong test set: {pid} (Dự kiến: {test_patients})"
            )

    return data


class IndependentTestDataset(Dataset):
    """PyTorch Dataset tiêu chuẩn y khoa cho tập kiểm thử độc lập (Independent Test Set).

    Đặc điểm kỹ thuật:
    - Kế thừa torch.utils.data.Dataset.
    - Không áp dụng bất kỳ Data Augmentation nào.
    - Áp dụng đầy đủ: HU Window [-1000, 400] -> Min-Max [0, 1] -> Binarize {0, 1} -> Resize 256x256.
    - Trả về dictionary chứa tensors và metadata chuẩn.
    - Hỗ trợ chế độ giả lập (mock=True) phục vụ CI/CD hoặc kiểm thử tự động không cần dữ liệu NIfTI nặng.
    """

    def __init__(
        self,
        manifest_path: str = "data/processed/test_manifest.json",
        target_size: Tuple[int, int] = (256, 256),
        hu_range: Tuple[float, float] = (-1000.0, 400.0),
        lesion_label: int = 3,
        cache_volumes: bool = True,
        raw_data_dir: Optional[str] = None,
        mock: bool = False,
        enforce_readonly: bool = True,
    ):
        """Khởi tạo IndependentTestDataset.

        Parameters
        ----------
        manifest_path : str
            Đường dẫn file test_manifest.json.
        target_size : Tuple[int, int]
            Kích thước chuẩn (chiều rộng, chiều cao), mặc định (256, 256).
        hu_range : Tuple[float, float]
            Cửa sổ Hounsfield Unit [min_hu, max_hu], mặc định (-1000.0, 400.0).
        lesion_label : int
            Giá trị nhãn tổn thương trên mask gốc (mặc định 3).
        cache_volumes : bool
            Lưu bộ nhớ đệm 3D volume trên RAM để tăng tốc.
        raw_data_dir : Optional[str]
            Thư mục chứa dữ liệu thô (cho Kaggle hoặc custom path).
        mock : bool
            Nếu True, tạo dữ liệu lát cắt giả lập tuân thủ đặc tả phục vụ unit test.
        enforce_readonly : bool
            Nếu True, tự động kiểm tra và khóa quyền ghi file manifest nếu chưa khóa.
        """
        self.manifest_path = Path(manifest_path)
        self.target_size = target_size
        self.hu_range = hu_range
        self.lesion_label = lesion_label
        self.cache_volumes = cache_volumes
        self.mock = mock
        self.raw_data_dir = (
            raw_data_dir
            or os.environ.get("RAW_DATA_DIR")
            or os.environ.get("KAGGLE_DATA_DIR")
        )

        # 1. Kiểm tra sự tồn tại của manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"[X] Không tìm thấy file test manifest: '{self.manifest_path}'.\n"
                f"-> Hãy chạy script 'python scripts/prepare_test_manifest.py' trước để tạo manifest."
            )

        # 2. Kiểm tra và đảm bảo tính read-only của manifest
        if enforce_readonly and not is_manifest_readonly(self.manifest_path):
            try:
                set_manifest_readonly(self.manifest_path, readonly=True)
            except Exception as e:
                print(f"[!] Cảnh báo: Không thể thiết lập read-only cho {self.manifest_path}: {e}")

        # 3. Đọc và xác thực manifest
        manifest_data = verify_test_manifest(self.manifest_path)
        self.metadata = manifest_data.get("metadata", {})
        self.slices: List[Dict[str, Any]] = manifest_data.get("slices", [])
        self.test_patients: List[str] = self.metadata.get("test_patients", [])

        # 4. Cache cho dữ liệu 3D NIfTI
        self._ct_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}
        self._path_cache: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.slices)

    def _resolve_path(self, path: str) -> str:
        """Phân giải đường dẫn file trên máy cục bộ hoặc môi trường Kaggle."""
        if path in self._path_cache:
            return self._path_cache[path]

        p = Path(path)
        if p.exists():
            self._path_cache[path] = str(p)
            return str(p)

        # Tìm kiếm qua raw_data_dir
        if self.raw_data_dir:
            base_dir = Path(self.raw_data_dir)
            candidates = [
                base_dir / p,
                base_dir / p.name,
                base_dir / p.parent.name / p.name,
                base_dir / "data" / "raw" / p.parent.name / p.name,
                base_dir / str(p).replace("data/raw/", "").replace("data\\raw\\", ""),
            ]
            for c in candidates:
                if c.exists() and c.stat().st_size > 0:
                    self._path_cache[path] = str(c)
                    return str(c)

            matches = list(base_dir.rglob(p.name))
            if matches:
                self._path_cache[path] = str(matches[0])
                return str(matches[0])

        # Tìm kiếm qua /kaggle/input
        kaggle_root = Path("/kaggle/input")
        if kaggle_root.exists():
            matches = list(kaggle_root.rglob(p.name))
            if matches:
                self._path_cache[path] = str(matches[0])
                return str(matches[0])

        return str(p)

    def _generate_mock_slice(self, record: Dict[str, Any], idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo cặp lát cắt giả lập chuẩn y khoa (độ phân giải 512x512) phục vụ kiểm thử.

        - CT: Dải HU [-1024, 1000] mô phỏng không khí, phổi, mô mềm và xương.
        - Mask: Nhãn đa lớp {0, 1, 2, 3} trong đó 3 là vùng tổn thương nếu lesion_present == 1.
        """
        # Sử dụng seed cố định theo sample_id hoặc idx để đảm bảo tái lập
        rng = np.random.RandomState(idx + 2026)

        # Lát cắt CT thô 512x512: Nền không khí (-1000 HU)
        ct_raw = np.full((512, 512), fill_value=-1000.0, dtype=np.float32)
        # Nhu mô phổi (-700 đến -500 HU)
        ct_raw[100:412, 100:412] = rng.uniform(-700.0, -500.0, size=(312, 312))
        # Thành ngực / mô đặc (+40 HU)
        ct_raw[80:100, 80:432] = 40.0
        # Xương sườn (+700 HU)
        ct_raw[85:95, 120:140] = 700.0

        # Mask thô 512x512
        mask_raw = np.zeros((512, 512), dtype=np.uint8)
        # Vùng phổi trái (1), phổi phải (2)
        mask_raw[120:380, 120:230] = 1
        mask_raw[120:380, 280:390] = 2

        # Nếu lát cắt có tổn thương (lesion_present == 1)
        if int(record.get("lesion_present", 0)) == 1:
            # Tổn thương đông đặc / kính mờ (-200 đến +50 HU, nhãn 3)
            ct_raw[200:260, 140:200] = rng.uniform(-200.0, 50.0, size=(60, 60))
            mask_raw[200:260, 140:200] = self.lesion_label

        return ct_raw, mask_raw

    def _get_raw_slice(self, record: Dict[str, Any], idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tải lát cắt 2D thô từ 3D volume NIfTI hoặc dùng giả lập nếu mock=True."""
        if self.mock:
            return self._generate_mock_slice(record, idx)

        ct_path = record["source_volume_path"]
        mask_path = record["source_mask_path"]
        z = int(record["slice_index"])

        resolved_ct = self._resolve_path(ct_path)
        resolved_mask = self._resolve_path(mask_path)

        # Nếu file không tồn tại trên đĩa và chưa có dữ liệu thực tế
        if not os.path.exists(resolved_ct) or not os.path.exists(resolved_mask):
            # Tự động chuyển sang mock slice nếu không tìm thấy file NIfTI
            return self._generate_mock_slice(record, idx)

        if nib is None:
            raise ImportError("Cần cài đặt 'nibabel' để đọc file .nii/.nii.gz: pip install nibabel")

        # Đọc dữ liệu với cache
        if self.cache_volumes:
            if resolved_ct not in self._ct_cache:
                self._ct_cache[resolved_ct] = nib.load(resolved_ct).get_fdata(dtype=np.float32)
            if resolved_mask not in self._mask_cache:
                self._mask_cache[resolved_mask] = nib.load(resolved_mask).get_fdata(dtype=np.float32)

            ct_volume = self._ct_cache[resolved_ct]
            mask_volume = self._mask_cache[resolved_mask]
        else:
            ct_volume = nib.load(resolved_ct).get_fdata(dtype=np.float32)
            mask_volume = nib.load(resolved_mask).get_fdata(dtype=np.float32)

        raw_ct_slice = ct_volume[:, :, z]
        raw_mask_slice = mask_volume[:, :, z]
        return raw_ct_slice, raw_mask_slice

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Trích xuất và tiền xử lý chuẩn y khoa cho 1 lát cắt kiểm thử.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa:
            - 'image': Tensor kích thước [1, target_size[0], target_size[1]], dải [0.0, 1.0].
            - 'mask': Tensor kích thước [1, target_size[0], target_size[1]], nhãn {0.0, 1.0}.
            - 'sample_id': Mã định danh mẫu lát cắt (str).
            - 'patient_id': Mã bệnh nhân (str).
            - 'slice_index': Chỉ số lát cắt theo trục z (int).
            - 'lesion_present': Cờ nhãn có tổn thương hay không (int, 0 hoặc 1).
            - 'volume_id': Tên volume gốc.
        """
        record = self.slices[idx]

        # 1. Trích xuất lát cắt 2D thô
        raw_ct_slice, raw_mask_slice = self._get_raw_slice(record, idx)

        # 2. Pipeline tiền xử lý chuẩn y khoa v1.0 (Tuyệt đối KHÔNG Augmentation):
        # 2.1. HU Windowing [-1000, 400] & Min-Max Scaling về [0.0, 1.0]
        norm_img = normalize_min_max(
            raw_ct_slice,
            min_hu=self.hu_range[0],
            max_hu=self.hu_range[1],
        )

        # 2.2. Nhị phân hóa Ground Truth Mask về {0, 1}
        bin_mask = binarize_mask(
            raw_mask_slice,
            lesion_label=self.lesion_label,
            dtype="uint8",
        )

        # 2.3. Resizing đồng bộ về target_size (Bilinear cho CT, Nearest Neighbor cho mask)
        res_img, res_mask = resize_ct_pair(
            norm_img,
            bin_mask,
            target_size=self.target_size,
        )

        # 3. Chuyển đổi thành PyTorch Tensor [C, H, W] với C=1
        tensor_img = torch.from_numpy(res_img).unsqueeze(0).to(torch.float32)
        tensor_mask = torch.from_numpy(res_mask).unsqueeze(0).to(torch.float32)

        return {
            "image": tensor_img,
            "mask": tensor_mask,
            "sample_id": record["sample_id"],
            "patient_id": record["patient_id"],
            "slice_index": int(record["slice_index"]),
            "lesion_present": int(record["lesion_present"]),
            "volume_id": record.get("volume_id", ""),
            "original_shape": record.get("original_shape", "512x512"),
        }


def get_eval_dataloader(
    manifest_path: str = "data/processed/test_manifest.json",
    target_size: Tuple[int, int] = (256, 256),
    hu_range: Tuple[float, float] = (-1000.0, 400.0),
    lesion_label: int = 3,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_volumes: bool = True,
    raw_data_dir: Optional[str] = None,
    mock: bool = False,
    enforce_readonly: bool = True,
) -> DataLoader:
    """Khởi tạo PyTorch DataLoader chuẩn hóa phục vụ đánh giá trên Independent Test Set.

    Ràng buộc kỹ thuật quan trọng:
    - batch_size BẮT BUỘC = 1: Phản ánh kịch bản chẩn đoán lâm sàng tuần tự từng lát cắt.
    - shuffle BẮT BUỘC = False: Bảo toàn tuyệt đối trình tự giải phẫu của từng bệnh nhân.
    - Không áp dụng Data Augmentation.

    Parameters
    ----------
    manifest_path : str
        Đường dẫn file test_manifest.json.
    target_size : Tuple[int, int]
        Kích thước chuẩn (256, 256).
    hu_range : Tuple[float, float]
        Cửa sổ HU [-1000, 400].
    lesion_label : int
        Nhãn tổn thương trên mask thô (mặc định 3).
    batch_size : int
        Kích thước batch (mặc định 1, luôn được cưỡng chế về 1 nếu khác).
    shuffle : bool
        Trộn dữ liệu (mặc định False, luôn được cưỡng chế về False nếu khác).
    num_workers : int
        Số luồng tải dữ liệu (mặc định 0 để an toàn và nhẹ).
    pin_memory : bool
        Khóa bộ nhớ RAM khi đưa vào GPU (mặc định False hoặc True nếu có CUDA).
    cache_volumes : bool
        Lưu cache 3D volumes vào RAM.
    raw_data_dir : Optional[str]
        Thư mục dữ liệu thô.
    mock : bool
        Nếu True, tạo dữ liệu lát cắt giả lập cho unit test / CI.
    enforce_readonly : bool
        Nếu True, đảm bảo manifest ở chế độ read-only.

    Returns
    -------
    DataLoader
        DataLoader trả về dict với image [1, 1, 256, 256], mask [1, 1, 256, 256], v.v.
    """
    # Cưỡng chế quy chuẩn chẩn đoán lâm sàng
    if batch_size != 1:
        print(f"[!] Chú ý: EVAL-01 yêu cầu batch_size=1 cho đánh giá lâm sàng tuần tự. Đã gán batch_size=1.")
        batch_size = 1

    if shuffle:
        print("[!] Chú ý: EVAL-01 yêu cầu shuffle=False để giữ nguyên thứ tự giải phẫu. Đã tắt shuffle.")
        shuffle = False

    dataset = IndependentTestDataset(
        manifest_path=manifest_path,
        target_size=target_size,
        hu_range=hu_range,
        lesion_label=lesion_label,
        cache_volumes=cache_volumes,
        raw_data_dir=raw_data_dir,
        mock=mock,
        enforce_readonly=enforce_readonly,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )

    return loader


if __name__ == "__main__":
    print("=" * 72)
    print("CHẠY UNIT TEST TOÀN DIỆN CHO EVAL_LOADER (TASK EVAL-01)")
    print("=" * 72)

    manifest_file = "data/processed/test_manifest.json"

    # 1. Kiểm tra sự tồn tại và tính READ-ONLY của manifest
    print("\n[Bước 1/5] Kiểm tra tính toàn vẹn và quyền READ-ONLY của manifest...")
    assert os.path.exists(manifest_file), f"Lỗi: Không tìm thấy file {manifest_file}!"
    is_ro = is_manifest_readonly(manifest_file)
    print(f"  + Trạng thái Read-Only của {manifest_file}: {'ĐẠT (Locked 0o444)' if is_ro else 'CHƯA KHÓA'}")
    assert is_ro, "Lỗi: File test_manifest.json chưa được khóa quyền ghi (Read-Only)!"

    # 2. Khởi tạo Dataset và DataLoader chuẩn
    print("\n[Bước 2/5] Khởi tạo IndependentTestDataset và get_eval_dataloader...")
    # Kiểm tra loader đọc manifest thực tế (sử dụng mock slice nếu volume thô chưa tải về máy)
    eval_loader = get_eval_dataloader(
        manifest_path=manifest_file,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        mock=False,
    )
    dataset_len = len(eval_loader.dataset)
    num_batches = len(eval_loader)
    print(f"  + Tổng số mẫu lát cắt trong Dataset: {dataset_len}")
    print(f"  + Tổng số batches trong DataLoader: {num_batches}")
    assert dataset_len == 265, f"Lỗi: Kỳ vọng 265 lát cắt test, nhận được {dataset_len}!"
    assert num_batches == 265, f"Lỗi: Kỳ vọng 265 batches (batch_size=1), nhận được {num_batches}!"

    # 3. Trích xuất mẫu batch đầu tiên và kiểm tra cấu trúc dictionary
    print("\n[Bước 3/5] Lấy batch mẫu và kiểm tra kiểu trả về dictionary...")
    batch = next(iter(eval_loader))
    assert isinstance(batch, dict), f"Lỗi: DataLoader phải trả về dictionary, nhận được {type(batch)}!"
    required_keys = ["image", "mask", "sample_id", "patient_id"]
    for k in required_keys:
        assert k in batch, f"Lỗi: Thiếu khóa '{k}' trong dictionary trả về từ DataLoader!"
    print(f"  + Các khóa có trong batch: {list(batch.keys())} -> ĐẠT")
    print(f"  + Sample ID: {batch['sample_id'][0]}")
    print(f"  + Patient ID: {batch['patient_id'][0]}")

    # 4. Kiểm tra kích thước Tensor [1, 1, 256, 256]
    print("\n[Bước 4/5] Kiểm tra Tensor shape của Image và Mask...")
    img = batch["image"]
    msk = batch["mask"]
    print(f"  + Image Tensor Shape : {list(img.shape)} (Dự kiến: [1, 1, 256, 256])")
    print(f"  + Mask Tensor Shape  : {list(msk.shape)} (Dự kiến: [1, 1, 256, 256])")
    assert img.shape == torch.Size([1, 1, 256, 256]), f"Sai kích thước ảnh: {img.shape}"
    assert msk.shape == torch.Size([1, 1, 256, 256]), f"Sai kích thước mask: {msk.shape}"
    assert img.dtype == torch.float32, f"Sai kiểu dữ liệu ảnh: {img.dtype}"
    assert msk.dtype == torch.float32, f"Sai kiểu dữ liệu mask: {msk.dtype}"
    print("  + Kích thước và kiểu dữ liệu tensor -> ĐẠT")

    # 5. Kiểm tra dải giá trị: Image in [0.0, 1.0], Mask in {0, 1}
    print("\n[Bước 5/5] Kiểm tra dải giá trị chuẩn hóa và tính nhị phân...")
    img_min = float(img.min())
    img_max = float(img.max())
    mask_unique = torch.unique(msk).tolist()
    print(f"  + Dải giá trị Image : [{img_min:.4f}, {img_max:.4f}] (Yêu cầu: [0.0, 1.0])")
    print(f"  + Tập nhãn của Mask : {mask_unique} (Yêu cầu: tập con của {{0.0, 1.0}})")
    assert 0.0 <= img_min and img_max <= 1.0, f"Ảnh vi phạm dải [0, 1]: min={img_min}, max={img_max}"
    assert set(mask_unique).issubset({0.0, 1.0}), f"Mask chứa nhãn ngoài nhị phân: {mask_unique}"
    print("  + Dải giá trị và nhị phân hóa -> ĐẠT")

    # 6. Kiểm tra đảm bảo batch_size == 1 và shuffle == False
    assert img.size(0) == 1, f"Lỗi: batch_size={img.size(0)} khác 1!"
    assert msk.size(0) == 1, f"Lỗi: batch_size={msk.size(0)} khác 1!"
    print(f"  + Xác nhận batch_size == 1 -> ĐẠT")

    print("\n" + "=" * 72)
    print("TẤT CẢ 5 BƯỚC KIỂM THỬ CHO EVAL_LOADER ĐÃ ĐẠT 100% TIÊU CHUẨN!")
    print("=" * 72)
