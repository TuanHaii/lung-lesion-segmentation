import hashlib
from pathlib import Path


def compute_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main():
    raw_dir = Path("data/raw")
    output_file = raw_dir / "md5_checksums.txt"

    if not raw_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {raw_dir}")
        return

    # Quét tất cả file dữ liệu y tế trong data/raw/
    raw_files = sorted([
        f
        for f in raw_dir.rglob("*")
        if f.is_file()
        and f.name != "md5_checksums.txt"
        and not f.name.startswith(".")
    ])

    if not raw_files:
        print(
            "Cảnh báo: data/raw/ hiện chưa có file dữ liệu nào để tính checksum!"
        )
        return

    print(f"Đang tính mã băm MD5 cho {len(raw_files)} files...")
    with open(output_file, "w", encoding="utf-8") as out:
        for f in raw_files:
            file_hash = compute_md5(f)
            rel_path = f.relative_to(raw_dir)
            out.write(f"{file_hash}  {rel_path}\n")
            print(f"OK: {file_hash}  {rel_path}")

    print(f"\n=> Hoàn tất DATA-02! Checksum đã lưu tại: {output_file}")


if __name__ == "__main__":
    main()