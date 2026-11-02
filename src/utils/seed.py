"""
Module cố định hạt giống ngẫu nhiên (Random Seed) đảm bảo tính tái lập 100% (Reproducibility).
Tuân thủ đặc tả nghiên cứu y khoa trong docs/05_OPERATIONAL_MANUAL.md và docs/04_TECH_STACK_AND_DATASET.md.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 2026, deterministic: bool = True) -> None:
    """Cố định toàn bộ các nguồn ngẫu nhiên trong hệ thống.

    Parameters
    ----------
    seed : int, mặc định 2026
        Hạt giống ngẫu nhiên theo chuẩn đề tài (seed = 2026).
    deterministic : bool, mặc định True
        Bật chế độ tất định của cuDNN (chống sai lệch giữa các lượt chạy).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    print(f">> [REPRODUCIBILITY] Random Seed locked to: {seed} (cuDNN deterministic={deterministic})")


def seed_worker(worker_id: int) -> None:
    """Hàm khởi tạo seed cho từng worker của PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

