"""Safe Path Resolver for Pipeline V2.

Guarantees:
1. Strict type isolation: CT paths cannot resolve to mask files; Mask paths cannot resolve to CT files.
2. Non-identity enforcement: CT path and Mask path can NEVER be identical.
3. No unsafe fallbacks: If a matching file of the exact requested type is not found,
   raises FileNotFoundError rather than returning an arbitrary NIfTI file.
4. Validates existence and non-zero file size.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple


class PathResolverError(RuntimeError):
    """Raised when path resolution fails or violates data integrity."""
    pass


class SafePathResolver:
    """Safe resolver for CT and Mask NIfTI files across local and Kaggle environments."""

    CT_POSITIVE_KEYWORDS = ["ct_volume", "ct_volumes", "ct_vol", "images", "ct"]
    CT_NEGATIVE_KEYWORDS = ["mask", "infection", "ground_truth", "label"]

    MASK_POSITIVE_KEYWORDS = [
        "mask", "masks", "infection", "infection_mask",
        "ground_truth_masks", "ground_truth", "label", "lung_and_infection"
    ]
    MASK_NEGATIVE_KEYWORDS = ["ct_volume", "ct_volumes", "ct_vol", "images"]

    def __init__(self, raw_data_dir: Optional[str] = None):
        self.raw_data_dir = (
            raw_data_dir
            or os.environ.get("RAW_DATA_DIR")
            or os.environ.get("KAGGLE_DATA_DIR")
        )
        self._path_cache: dict[str, Path] = {}

    def _get_search_roots(self) -> List[Path]:
        roots: List[Path] = []
        if self.raw_data_dir:
            p = Path(self.raw_data_dir)
            if p.exists():
                roots.append(p)

        # Standard environment checks
        for env_var in ["RAW_DATA_DIR", "KAGGLE_DATA_DIR"]:
            val = os.environ.get(env_var)
            if val:
                p = Path(val)
                if p.exists() and p not in roots:
                    roots.append(p)

        # Kaggle environment
        kaggle_root = Path("/kaggle/input")
        if kaggle_root.exists():
            roots.append(kaggle_root)
            for child in kaggle_root.iterdir():
                if child.is_dir() and child not in roots:
                    roots.append(child)

        # Local workspace defaults
        for default_rel in [Path("data/raw"), Path(".")]:
            if default_rel.exists() and default_rel.resolve() not in [r.resolve() for r in roots]:
                roots.append(default_rel)

        return roots

    def _is_valid_type(self, path: Path, file_type: str) -> bool:
        """Check if a path legitimately matches the expected file type."""
        path_str = str(path).lower().replace("\\", "/")

        if file_type == "ct":
            # CT file must NOT contain mask keywords
            for neg in self.CT_NEGATIVE_KEYWORDS:
                if neg in path_str:
                    return False
            return True
        elif file_type == "mask":
            # Mask file must contain at least one mask keyword
            has_pos = any(pos in path_str for pos in self.MASK_POSITIVE_KEYWORDS)
            # Mask file must NOT be in a pure CT image folder without mask keyword in filename
            is_ct_folder = any(neg in str(path.parent).lower() for neg in self.MASK_NEGATIVE_KEYWORDS)
            if is_ct_folder and not any(pos in path.name.lower() for pos in self.MASK_POSITIVE_KEYWORDS):
                return False
            return has_pos
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

    def resolve_file(self, target_path: str, file_type: str) -> Path:
        """Resolve a single file path safely.

        Parameters
        ----------
        target_path : str
            Requested relative or absolute file path.
        file_type : str
            Either 'ct' or 'mask'.

        Returns
        -------
        Path
            Verified path to the existing file.

        Raises
        ------
        FileNotFoundError
            If file cannot be found.
        PathResolverError
            If file fails type verification or is 0-byte.
        """
        assert file_type in ["ct", "mask"], f"Invalid file_type: {file_type}"
        cache_key = f"{file_type}:{target_path}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        p = Path(target_path)

        # 1. Direct path check
        if p.exists() and p.is_file():
            if p.stat().st_size == 0:
                raise PathResolverError(f"File '{p}' exists but is 0 bytes (corrupted).")
            if not self._is_valid_type(p, file_type):
                raise PathResolverError(
                    f"File '{p}' does not match expected type '{file_type}'! "
                    f"Possible CT/Mask misplacement."
                )
            self._path_cache[cache_key] = p
            return p

        # 2. Search roots
        clean_stem = p.name.replace(".nii.gz", "").replace(".nii", "")
        exts = [".nii.gz", ".nii"]

        candidate_subdirs = (
            ["Infection_Mask", "infection_mask", "ground_truth_masks",
             "Lung_and_Infection_Mask", "masks", "mask"]
            if file_type == "mask"
            else ["ct_volumes", "CT_Volumes", "images", "ct"]
        )

        search_roots = self._get_search_roots()

        # Step 2a: Check candidate subdirectories
        for root in search_roots:
            for sdir in candidate_subdirs:
                for ext in exts:
                    cand = root / sdir / f"{clean_stem}{ext}"
                    if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                        if self._is_valid_type(cand, file_type):
                            self._path_cache[cache_key] = cand
                            return cand

        # Step 2b: Recursive search with strict type filtering
        for root in search_roots:
            matches = list(root.rglob(f"{clean_stem}.nii*"))
            for m in matches:
                if m.is_file() and m.stat().st_size > 0:
                    if self._is_valid_type(m, file_type):
                        self._path_cache[cache_key] = m
                        return m

        # NEVER fall back to arbitrary matches!
        raise FileNotFoundError(
            f"SafePathResolver: Could not find valid {file_type.upper()} file for '{target_path}'. "
            f"Searched roots: {[str(r) for r in search_roots]} with clean stem '{clean_stem}'."
        )

    def resolve_pair(
        self,
        ct_path: str,
        mask_path: str,
        patient_id: str = "",
        sample_id: str = "",
    ) -> Tuple[Path, Path]:
        """Resolve both CT and Mask paths, enforcing that they are distinct files.

        Returns
        -------
        Tuple[Path, Path]
            (resolved_ct_path, resolved_mask_path)

        Raises
        ------
        PathResolverError
            If CT and Mask resolve to the exact same file.
        """
        res_ct = self.resolve_file(ct_path, file_type="ct")
        res_mask = self.resolve_file(mask_path, file_type="mask")

        # Crucial Safety Rule: CT and Mask can NEVER be the same file
        if res_ct.resolve() == res_mask.resolve():
            raise PathResolverError(
                f"FATAL PAIRING ERROR: CT path and Mask path resolved to the EXACT SAME file: "
                f"'{res_ct}'. Patient: '{patient_id}', Sample: '{sample_id}'."
            )

        return res_ct, res_mask
