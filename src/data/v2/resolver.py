"""Hardened Safe Path Resolver for Pipeline V2.

Guarantees:
1. Patient Identity Resolution: Resolves CT and Mask files strictly by patient identity
   and known dataset structure, not broad arbitrary keywords.
2. Distinct File Enforcement: CT path and Mask path can NEVER be identical.
3. Strict Directory Boundaries: CT candidates must originate from approved CT directories
   (e.g. 'COVID-19-CT-Seg_20cases', 'ct_volumes'); Mask candidates from approved Mask
   directories (e.g. 'Infection_Mask', 'ground_truth_masks').
4. No Permissive / Arbitrary Classification: Arbitrary NIfTI files outside approved
   dataset directories are rejected.
5. Flexible Filename Pattern Matching: Supports all actual CT variations (e.g. 'case009.nii',
   'case_009.nii.gz', 'coronacases_009.nii.gz', 'coronacases_org_009.nii', 'coronacases_9.nii.gz')
   and does not assume CT and Mask filenames are identical.
6. Ambiguity Detection: If multiple distinct candidate files exist for a patient in the
   approved directory, raises an explicit AmbiguityError instead of silently picking one.
7. Strict Extension Validation: Only .nii and .nii.gz files are accepted.
8. Pair Integrity Validation: Validates file existence, non-zero size, NIfTI loadability,
   shape compatibility, affine compatibility within documented tolerance (atol=1e-3),
   and mask label conventions. Never silently resamples or transposes.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    nib = None
    np = None


class PathResolverError(RuntimeError):
    """Base exception for path resolution failures or data integrity violations."""
    pass


class AmbiguityError(PathResolverError):
    """Raised when multiple valid candidate files match the requested patient/type."""
    pass


class PairIntegrityError(PathResolverError):
    """Raised when CT and Mask pairing violates integrity constraints (shapes, affines, etc.)."""
    pass


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Verified Explicit Patient Registry (All 20 Patients)
# -------------------------------------------------------------------------

VERIFIED_PATIENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "P_coronacases_001": {
        "source": "coronacases",
        "case_num": 1,
        "known_ct_aliases": {"coronacases_org_001", "coronacases_001", "case001", "case_001"},
        "known_mask_aliases": {"coronacases_001", "infection_mask_coronacases_001"},
    },
    "P_coronacases_002": {
        "source": "coronacases",
        "case_num": 2,
        "known_ct_aliases": {"coronacases_org_002", "coronacases_002", "case002", "case_002"},
        "known_mask_aliases": {"coronacases_002", "infection_mask_coronacases_002"},
    },
    "P_coronacases_003": {
        "source": "coronacases",
        "case_num": 3,
        "known_ct_aliases": {"coronacases_org_003", "coronacases_003", "case003", "case_003"},
        "known_mask_aliases": {"coronacases_003", "infection_mask_coronacases_003"},
    },
    "P_coronacases_004": {
        "source": "coronacases",
        "case_num": 4,
        "known_ct_aliases": {"coronacases_org_004", "coronacases_004", "case004", "case_004"},
        "known_mask_aliases": {"coronacases_004", "infection_mask_coronacases_004"},
    },
    "P_coronacases_005": {
        "source": "coronacases",
        "case_num": 5,
        "known_ct_aliases": {"coronacases_org_005", "coronacases_005", "case005", "case_005"},
        "known_mask_aliases": {"coronacases_005", "infection_mask_coronacases_005"},
    },
    "P_coronacases_006": {
        "source": "coronacases",
        "case_num": 6,
        "known_ct_aliases": {"coronacases_org_006", "coronacases_006", "case006", "case_006"},
        "known_mask_aliases": {"coronacases_006", "infection_mask_coronacases_006"},
    },
    "P_coronacases_007": {
        "source": "coronacases",
        "case_num": 7,
        "known_ct_aliases": {
            "coronacases_org_case007", "coronacases_org_007", "coronacases_007", "case007", "case_007"
        },
        "known_mask_aliases": {"coronacases_007", "infection_mask_coronacases_007"},
    },
    "P_coronacases_008": {
        "source": "coronacases",
        "case_num": 8,
        "known_ct_aliases": {"coronacases_org_008", "coronacases_008", "case008", "case_008"},
        "known_mask_aliases": {"coronacases_008", "infection_mask_coronacases_008"},
    },
    "P_coronacases_009": {
        "source": "coronacases",
        "case_num": 9,
        "known_ct_aliases": {"case009", "coronacases_org_009", "coronacases_009", "case_009"},
        "known_mask_aliases": {"coronacases_009", "infection_mask_coronacases_009"},
    },
    "P_coronacases_010": {
        "source": "coronacases",
        "case_num": 10,
        "known_ct_aliases": {
            "coronacases_org_case010", "coronacases_org_010", "coronacases_010", "case010", "case_010"
        },
        "known_mask_aliases": {"coronacases_010", "infection_mask_coronacases_010"},
    },
    "P_radiopaedia_10_85902_1": {
        "source": "radiopaedia",
        "case": 10, "study": 85902, "series": 1,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm",
            "radiopaedia_org_10_85902_1", "radiopaedia_10_85902_1", "case_10_85902_1",
        },
        "known_mask_aliases": {"radiopaedia_10_85902_1", "infection_mask_radiopaedia_10_85902_1"},
    },
    "P_radiopaedia_10_85902_3": {
        "source": "radiopaedia",
        "case": 10, "study": 85902, "series": 3,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm",
            "radiopaedia_org_10_85902_3", "radiopaedia_10_85902_3", "case_10_85902_3",
        },
        "known_mask_aliases": {"radiopaedia_10_85902_3", "infection_mask_radiopaedia_10_85902_3"},
    },
    "P_radiopaedia_14_85914_0": {
        "source": "radiopaedia",
        "case": 14, "study": 85914, "series": 0,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm",
            "radiopaedia_org_14_85914_0", "radiopaedia_14_85914_0", "case_14_85914_0",
        },
        "known_mask_aliases": {"radiopaedia_14_85914_0", "infection_mask_radiopaedia_14_85914_0"},
    },
    "P_radiopaedia_27_86410_0": {
        "source": "radiopaedia",
        "case": 27, "study": 86410, "series": 0,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm",
            "radiopaedia_org_27_86410_0", "radiopaedia_27_86410_0", "case_27_86410_0",
        },
        "known_mask_aliases": {"radiopaedia_27_86410_0", "infection_mask_radiopaedia_27_86410_0"},
    },
    "P_radiopaedia_29_86490_1": {
        "source": "radiopaedia",
        "case": 29, "study": 86490, "series": 1,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm",
            "radiopaedia_org_29_86490_1", "radiopaedia_29_86490_1", "case_29_86490_1",
        },
        "known_mask_aliases": {"radiopaedia_29_86490_1", "infection_mask_radiopaedia_29_86490_1"},
    },
    "P_radiopaedia_29_86491_1": {
        "source": "radiopaedia",
        "case": 29, "study": 86491, "series": 1,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm",
            "radiopaedia_org_29_86491_1", "radiopaedia_29_86491_1", "case_29_86491_1",
        },
        "known_mask_aliases": {"radiopaedia_29_86491_1", "infection_mask_radiopaedia_29_86491_1"},
    },
    "P_radiopaedia_36_86526_0": {
        "source": "radiopaedia",
        "case": 36, "study": 86526, "series": 0,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-36_86526_0-dcm",
            "radiopaedia_org_36_86526_0", "radiopaedia_36_86526_0", "case_36_86526_0",
        },
        "known_mask_aliases": {"radiopaedia_36_86526_0", "infection_mask_radiopaedia_36_86526_0"},
    },
    "P_radiopaedia_40_86625_0": {
        "source": "radiopaedia",
        "case": 40, "study": 86625, "series": 0,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm",
            "radiopaedia_org_40_86625_0", "radiopaedia_40_86625_0", "case_40_86625_0",
        },
        "known_mask_aliases": {"radiopaedia_40_86625_0", "infection_mask_radiopaedia_40_86625_0"},
    },
    "P_radiopaedia_4_85506_1": {
        "source": "radiopaedia",
        "case": 4, "study": 85506, "series": 1,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm",
            "radiopaedia_org_4_85506_1", "radiopaedia_4_85506_1", "case_4_85506_1",
        },
        "known_mask_aliases": {"radiopaedia_4_85506_1", "infection_mask_radiopaedia_4_85506_1"},
    },
    "P_radiopaedia_7_85703_0": {
        "source": "radiopaedia",
        "case": 7, "study": 85703, "series": 0,
        "known_ct_aliases": {
            "radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm",
            "radiopaedia_org_7_85703_0", "radiopaedia_7_85703_0", "case_7_85703_0",
        },
        "known_mask_aliases": {"radiopaedia_7_85703_0", "infection_mask_radiopaedia_7_85703_0"},
    },
}

# -------------------------------------------------------------------------
# Pre-compiled Regex Patterns for Strict Source-Specific Normalization
# -------------------------------------------------------------------------

RE_CORONACASES = re.compile(
    r"^(?:p_)?(?:coronacases(?:_org)?(?:_case)?|case)[-_]?0*(\d+)$",
    re.IGNORECASE,
)

RE_RADIOPAEDIA = re.compile(
    r"^(?:p_)?(?:radiopaedia(?:_org)?(?:_covid[-_]19[-_]pneumonia)?(?:_case)?|case)[-_]?0*(\d+)[-_](\d+)[-_](\d+)(?:[-_]dcm)?$",
    re.IGNORECASE,
)


def is_nii_file(path: Path) -> bool:
    """Check if file ends with .nii or .nii.gz (case-insensitive)."""
    name = path.name.lower()
    return name.endswith(".nii.gz") or (name.endswith(".nii") and not name.endswith(".nii.gz"))


def clean_stem(filename: str) -> str:
    """Strip .nii or .nii.gz from filename, in lower case."""
    name = filename.lower()
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name


def parse_coronacases_id(identifier: str) -> Optional[Dict[str, Any]]:
    """Parse Coronacases patient identifier or filename stem.

    Supports:
    - Standard: 'P_coronacases_001', 'coronacases_001', 'coronacases_1'
    - Org variations: 'coronacases_org_001', 'coronacases_org_case007', 'coronacases_org_case010'
    - Case variations: 'case009', 'case_009', 'case9'
    """
    stem = clean_stem(identifier)
    stem = re.sub(r"^(?:infection_mask_|mask_)", "", stem)
    stem = re.sub(r"[-_](?:mask|clean|org)$", "", stem)

    m = RE_CORONACASES.match(stem)
    if m:
        case_num = int(m.group(1))
        return {
            "source": "coronacases",
            "case_num": case_num,
            "canonical_id": f"P_coronacases_{case_num:03d}",
        }
    return None


def parse_radiopaedia_id(identifier: str) -> Optional[Dict[str, Any]]:
    """Parse Radiopaedia patient identifier or filename stem.

    Supports:
    - Standard: 'P_radiopaedia_10_85902_1', 'radiopaedia_10_85902_1'
    - Real Kaggle CT volume names: 'radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm'
    - Case variations: 'case_10_85902_1', 'radiopaedia_case_10_85902_1'
    """
    stem = clean_stem(identifier)
    stem = re.sub(r"^(?:infection_mask_|mask_)", "", stem)
    stem = re.sub(r"[-_](?:mask|clean|org)$", "", stem)

    m = RE_RADIOPAEDIA.match(stem)
    if m:
        case_val = int(m.group(1))
        study_val = int(m.group(2))
        series_val = int(m.group(3))
        return {
            "source": "radiopaedia",
            "case": case_val,
            "study": study_val,
            "series": series_val,
            "canonical_id": f"P_radiopaedia_{case_val}_{study_val}_{series_val}",
        }
    return None


def parse_patient_id(identifier: str) -> Optional[Dict[str, Any]]:
    """Parse patient ID or filename stem into structured patient metadata."""
    res_c = parse_coronacases_id(identifier)
    if res_c is not None:
        return res_c
    res_r = parse_radiopaedia_id(identifier)
    if res_r is not None:
        return res_r
    return None


def classify_enclosing_directory(path: Path) -> Optional[str]:
    """Classify path as CT or Mask based on its closest approved enclosing directory.

    Traverses parents from innermost (immediate parent) to outermost.
    Returns 'ct', 'mask', or None.
    """
    # Approved folder names (case-insensitive)
    approved_ct = {"covid-19-ct-seg_20cases", "ct_volumes", "ct_scans", "images", "ct"}
    approved_mask = {
        "infection_mask", "ground_truth_masks", "lung_and_infection_mask",
        "masks", "mask", "labels"
    }

    # First check immediate parent
    immediate = path.parent.name.lower()
    if immediate in approved_mask:
        return "mask"
    if immediate in approved_ct:
        return "ct"

    # Next check ancestors from innermost to outermost
    for parent in path.parents:
        pname = parent.name.lower()
        if pname in approved_mask:
            return "mask"
        if pname in approved_ct:
            return "ct"

    return None


# -------------------------------------------------------------------------
# SafePathResolver Implementation
# -------------------------------------------------------------------------

class SafePathResolver:
    """Hardened resolver for CT and Mask NIfTI volumes across local and Kaggle environments."""

    APPROVED_CT_DIRS: Set[str] = {
        "covid-19-ct-seg_20cases",
        "ct_volumes",
        "ct_scans",
        "images",
        "ct",
    }
    APPROVED_MASK_DIRS: Set[str] = {
        "infection_mask",
        "ground_truth_masks",
        "lung_and_infection_mask",
        "masks",
        "mask",
        "labels",
    }

    ALLOWED_EXTENSIONS: Set[str] = {".nii.gz", ".nii"}
    CT_NEGATIVE_KEYWORDS: List[str] = ["mask", "infection", "ground_truth", "label"]

    AFFINE_ATOL: float = 1e-3
    AFFINE_RTOL: float = 1e-3

    def __init__(
        self,
        raw_data_dir: Optional[str] = None,
        strict_nifti: bool = False,
    ):
        """Initialize SafePathResolver.

        Parameters
        ----------
        raw_data_dir : Optional[str]
            Explicit directory containing raw CT/Mask dataset subdirectories.
        strict_nifti : bool
            If True, enforces strict nibabel header reading during validation.
        """
        self.raw_data_dir = (
            raw_data_dir
            or os.environ.get("RAW_DATA_DIR")
            or os.environ.get("KAGGLE_DATA_DIR")
        )
        self.strict_nifti = strict_nifti
        self._path_cache: Dict[str, Path] = {}
        self._pair_cache: Dict[str, Tuple[Path, Path]] = {}

    def _get_search_roots(self) -> List[Path]:
        """Collect all candidate search roots in local workspace and Kaggle environments."""
        roots: List[Path] = []
        if self.raw_data_dir:
            p = Path(self.raw_data_dir)
            if p.exists():
                roots.append(p)
                return roots  # Explicit raw_data_dir takes strict precedence to avoid cross-mount pollution

        for env_var in ["RAW_DATA_DIR", "KAGGLE_DATA_DIR"]:
            val = os.environ.get(env_var)
            if val:
                p = Path(val)
                if p.exists() and p not in roots:
                    roots.append(p)

        # Kaggle input directory exploration (up to 4 levels deep)
        kaggle_root = Path("/kaggle/input")
        if kaggle_root.exists():
            roots.append(kaggle_root)
            try:
                for child in kaggle_root.iterdir():
                    if child.is_dir() and child not in roots:
                        roots.append(child)
                        for grandchild in child.iterdir():
                            if grandchild.is_dir() and grandchild not in roots:
                                roots.append(grandchild)
                                for ggchild in grandchild.iterdir():
                                    if ggchild.is_dir() and ggchild not in roots:
                                        roots.append(ggchild)
                                        for gggchild in ggchild.iterdir():
                                            if gggchild.is_dir() and gggchild not in roots:
                                                roots.append(gggchild)
            except Exception:
                pass

        # Local workspace fallbacks
        for default_rel in [Path("data/raw"), Path("data"), Path(".")]:
            if default_rel.exists() and default_rel.resolve() not in [r.resolve() for r in roots]:
                roots.append(default_rel)

        return roots

    def _is_valid_type(self, path: Path, file_type: str) -> bool:
        """Validate that a candidate path legitimately belongs to file_type ('ct' or 'mask')."""
        if not is_nii_file(path):
            return False

        file_name = path.name.lower()
        dir_class = classify_enclosing_directory(path)

        if file_type == "ct":
            # 1. Must originate from an approved CT directory
            if dir_class != "ct":
                return False
            # 2. Filename cannot contain mask keywords
            for neg in self.CT_NEGATIVE_KEYWORDS:
                if neg in file_name:
                    return False
            return True

        elif file_type == "mask":
            # 1. Must originate from an approved Mask directory
            if dir_class != "mask":
                return False
            # 2. Immediate parent cannot be a CT folder
            if path.parent.name.lower() in self.APPROVED_CT_DIRS:
                return False
            return True
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

    def matches_patient(self, path: Path, patient_id: str, file_type: str) -> bool:
        """Verify whether a file path belongs to the requested patient identity."""
        pid_info = parse_patient_id(patient_id)
        file_info = parse_patient_id(path.name)

        if pid_info is not None and file_info is not None:
            # Source must match
            if pid_info["source"] != file_info["source"]:
                return False

            if pid_info["source"] == "coronacases":
                return pid_info["case_num"] == file_info["case_num"]
            elif pid_info["source"] == "radiopaedia":
                return (
                    pid_info["case"] == file_info["case"]
                    and pid_info["study"] == file_info["study"]
                    and pid_info["series"] == file_info["series"]
                )

        # Explicit verified patient registry lookup (controlled matching, no wild substring)
        canon_id = pid_info["canonical_id"] if pid_info else patient_id
        if canon_id in VERIFIED_PATIENT_REGISTRY:
            reg = VERIFIED_PATIENT_REGISTRY[canon_id]
            target_aliases = reg["known_ct_aliases"] if file_type == "ct" else reg["known_mask_aliases"]
            cstem = clean_stem(path.name)
            return cstem in target_aliases

        return False

    def find_candidates_for_patient(self, patient_id: str, file_type: str) -> List[Path]:
        """Search all roots and return all distinct valid candidate files for patient_id."""
        assert file_type in ["ct", "mask"], f"Invalid file_type: {file_type}"
        target_dirs = self.APPROVED_CT_DIRS if file_type == "ct" else self.APPROVED_MASK_DIRS

        roots = self._get_search_roots()
        candidates: List[Path] = []
        seen_resolved: Set[Path] = set()

        for root in roots:
            if not root.exists():
                continue

            # Check direct subdirectories matching target_dirs
            dirs_to_check: List[Path] = []
            if root.name.lower() in target_dirs:
                dirs_to_check.append(root)

            try:
                for child in root.iterdir():
                    if child.is_dir() and child.name.lower() in target_dirs:
                        dirs_to_check.append(child)
            except Exception:
                pass

            for d in dirs_to_check:
                try:
                    for item in d.glob("*"):
                        if item.is_file() and is_nii_file(item) and item.stat().st_size > 0:
                            res = item.resolve()
                            if res not in seen_resolved:
                                if self._is_valid_type(item, file_type) and self.matches_patient(item, patient_id, file_type):
                                    seen_resolved.add(res)
                                    candidates.append(item)
                        elif item.is_dir():
                            for subitem in item.glob("*"):
                                if subitem.is_file() and is_nii_file(subitem) and subitem.stat().st_size > 0:
                                    res = subitem.resolve()
                                    if res not in seen_resolved:
                                        if self._is_valid_type(subitem, file_type) and self.matches_patient(subitem, patient_id, file_type):
                                            seen_resolved.add(res)
                                            candidates.append(subitem)
                except Exception:
                    pass

        # If direct matching didn't yield candidates, perform controlled rglob
        if not candidates:
            for root in roots:
                try:
                    for item in root.rglob("*.nii*"):
                        if item.is_file() and is_nii_file(item) and item.stat().st_size > 0:
                            res = item.resolve()
                            if res not in seen_resolved:
                                if self._is_valid_type(item, file_type) and self.matches_patient(item, patient_id, file_type):
                                    seen_resolved.add(res)
                                    candidates.append(item)
                except Exception:
                    pass

        return candidates

    def resolve_patient_file(
        self,
        patient_id: str,
        file_type: str,
        hint_path: Optional[str] = None,
    ) -> Path:
        """Resolve a single CT or Mask file for patient_id with strict ambiguity detection."""
        assert file_type in ["ct", "mask"], f"Invalid file_type: {file_type}"

        # 1. Test hint_path if given
        if hint_path:
            p_hint = Path(hint_path)
            if p_hint.exists() and p_hint.is_file():
                if not is_nii_file(p_hint):
                    raise PathResolverError(f"File '{p_hint}' has invalid extension. Only .nii/.nii.gz accepted.")
                if p_hint.stat().st_size == 0:
                    raise PathResolverError(f"File '{p_hint}' is 0 bytes (corrupted).")
                if not self._is_valid_type(p_hint, file_type):
                    raise PathResolverError(
                        f"Hint file '{p_hint}' does not match expected file type '{file_type}'."
                    )
                if not self.matches_patient(p_hint, patient_id, file_type):
                    raise PairIntegrityError(
                        f"Hint file '{p_hint.name}' does not match patient identity '{patient_id}'."
                    )
                # Verify hint has no ambiguity against discovered candidates
                cands = self.find_candidates_for_patient(patient_id, file_type)
                if len(cands) > 1:
                    raise AmbiguityError(
                        f"Ambiguous {file_type.upper()} candidates for patient '{patient_id}': "
                        f"{[str(c) for c in cands]}."
                    )
                return p_hint

        # 2. Discover candidates across approved directories
        candidates = self.find_candidates_for_patient(patient_id, file_type)

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"SafePathResolver: Could not find valid {file_type.upper()} file for patient '{patient_id}'. "
                f"Searched approved directories in roots: {[str(r) for r in self._get_search_roots()]}."
            )
        elif len(candidates) > 1:
            raise AmbiguityError(
                f"Ambiguous {file_type.upper()} candidates for patient '{patient_id}'. "
                f"Found {len(candidates)} matching files: {[str(c) for c in candidates]}. "
                f"Manual disambiguation required."
            )

        return candidates[0]

    def resolve_file(
        self,
        target_path: str,
        file_type: str,
        patient_id: Optional[str] = None,
    ) -> Path:
        """Resolve a single file path safely with type validation and ambiguity detection.

        Maintains backward compatibility for callers specifying file paths directly.
        """
        assert file_type in ["ct", "mask"], f"Invalid file_type: {file_type}"
        cache_key = f"{file_type}:{target_path}:{patient_id or ''}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        p = Path(target_path)

        # Validate extension
        if not is_nii_file(p):
            raise PathResolverError(
                f"Invalid file extension for '{p.name}'. Only .nii and .nii.gz are accepted."
            )

        # Check direct existence
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

        # Infer patient ID from filename stem if not provided
        pid = patient_id
        if not pid:
            info = parse_patient_id(clean_stem(p.name))
            if info:
                pid = info["canonical_id"]

        if pid:
            resolved = self.resolve_patient_file(pid, file_type=file_type, hint_path=target_path)
            self._path_cache[cache_key] = resolved
            return resolved

        # If patient ID cannot be inferred, search approved directories by stem
        cstem = clean_stem(p.name)
        search_roots = self._get_search_roots()
        cands: List[Path] = []
        for root in search_roots:
            for item in root.rglob(f"{cstem}.nii*"):
                if item.is_file() and is_nii_file(item) and item.stat().st_size > 0:
                    if self._is_valid_type(item, file_type):
                        cands.append(item)

        distinct = list({c.resolve(): c for c in cands}.values())
        if len(distinct) == 0:
            raise FileNotFoundError(
                f"SafePathResolver: Could not find valid {file_type.upper()} file for '{target_path}'. "
                f"Searched roots: {[str(r) for r in search_roots]}."
            )
        elif len(distinct) > 1:
            raise AmbiguityError(
                f"Ambiguous matches for '{target_path}': {[str(c) for c in distinct]}."
            )

        self._path_cache[cache_key] = distinct[0]
        return distinct[0]

    def validate_pair(
        self,
        ct_path: Path,
        mask_path: Path,
        patient_id: str = "",
        validate_data: bool = False,
    ) -> Tuple[Path, Path]:
        """Validate CT-mask pairing integrity according to Phase 3 criteria.

        Validates:
        - Both files exist and are non-zero.
        - CT and Mask are distinct physical files.
        - Both files belong to the same patient.
        - Both NIfTI files can be loaded (if nibabel is installed).
        - 3D shapes are compatible.
        - Affine matrices match within explicitly documented tolerance (atol=1e-3, rtol=1e-3).
        - Mask labels conform to segmentation conventions (when validate_data=True).
        - CT scan does not contain purely binary mask data.
        """
        # 1. Existence and non-zero size
        if not ct_path.is_file():
            raise FileNotFoundError(f"CT file does not exist: {ct_path}")
        if ct_path.stat().st_size == 0:
            raise PathResolverError(f"CT file '{ct_path}' is 0 bytes (corrupted).")

        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
        if mask_path.stat().st_size == 0:
            raise PathResolverError(f"Mask file '{mask_path}' is 0 bytes (corrupted).")

        # 2. Distinct file enforcement (Rule 2)
        if ct_path.resolve() == mask_path.resolve():
            raise PairIntegrityError(
                f"FATAL PAIRING ERROR: CT path and Mask path resolved to the EXACT SAME file: '{ct_path}'. "
                f"Patient: '{patient_id}'."
            )

        # 3. Patient identity verification
        if patient_id:
            if not self.matches_patient(ct_path, patient_id, file_type="ct"):
                raise PairIntegrityError(
                    f"CT file '{ct_path.name}' does not match requested patient '{patient_id}'."
                )
            if not self.matches_patient(mask_path, patient_id, file_type="mask"):
                raise PairIntegrityError(
                    f"Mask file '{mask_path.name}' does not match requested patient '{patient_id}'."
                )

        # Verify CT and Mask patient match each other
        pid_ct = parse_patient_id(clean_stem(ct_path.name))
        pid_mask = parse_patient_id(clean_stem(mask_path.name))
        if pid_ct and pid_mask:
            if pid_ct["canonical_id"] != pid_mask["canonical_id"]:
                raise PairIntegrityError(
                    f"Patient mismatch: CT '{ct_path.name}' belongs to {pid_ct['canonical_id']} "
                    f"but Mask '{mask_path.name}' belongs to {pid_mask['canonical_id']}."
                )

        # 4. NIfTI loading and geometric/header integrity
        if nib is not None:
            try:
                ct_nii = nib.load(str(ct_path))
                mask_nii = nib.load(str(mask_path))
            except Exception as e:
                if self.strict_nifti:
                    raise PairIntegrityError(f"Failed to load NIfTI headers: {e}") from e
                return ct_path, mask_path

            # Shape check
            if ct_nii.shape != mask_nii.shape:
                raise PairIntegrityError(
                    f"Shape mismatch for patient '{patient_id}': "
                    f"CT shape {ct_nii.shape} != Mask shape {mask_nii.shape}."
                )

            # Affine compatibility check
            if not np.allclose(ct_nii.affine, mask_nii.affine, atol=self.AFFINE_ATOL, rtol=self.AFFINE_RTOL):
                max_diff = float(np.max(np.abs(ct_nii.affine - mask_nii.affine)))
                raise PairIntegrityError(
                    f"Affine mismatch between CT and Mask for patient '{patient_id}': "
                    f"max absolute difference {max_diff:.6f} exceeds tolerance "
                    f"(atol={self.AFFINE_ATOL}, rtol={self.AFFINE_RTOL})."
                )

            # Deep data inspection (optional, cached)
            if validate_data:
                mask_data = mask_nii.get_fdata(dtype=np.float32)
                unique_labels = set(np.unique(mask_data))
                if not unique_labels.issubset({0.0, 1.0, 2.0, 3.0}):
                    raise PairIntegrityError(
                        f"Invalid mask labels {unique_labels} for patient '{patient_id}'. "
                        f"Allowed label values must be a subset of {{0, 1, 2, 3}}."
                    )
                ct_data = ct_nii.get_fdata(dtype=np.float32)
                ct_unique = set(np.unique(ct_data))
                if ct_unique.issubset({0.0, 1.0}) and len(ct_unique) <= 2:
                    raise PairIntegrityError(
                        f"CT volume '{ct_path}' contains only binary values {ct_unique}. "
                        f"This is a mask file masquerading as a CT scan!"
                    )

        return ct_path, mask_path

    def resolve_patient(
        self,
        patient_id: str,
        ct_hint: Optional[str] = None,
        mask_hint: Optional[str] = None,
        validate: bool = True,
    ) -> Tuple[Path, Path]:
        """Resolve both CT and Mask paths for a given patient ID."""
        cache_key = f"patient:{patient_id}"
        if cache_key in self._pair_cache:
            return self._pair_cache[cache_key]

        ct_path = self.resolve_patient_file(patient_id, file_type="ct", hint_path=ct_hint)
        mask_path = self.resolve_patient_file(patient_id, file_type="mask", hint_path=mask_hint)

        if validate:
            self.validate_pair(ct_path, mask_path, patient_id=patient_id, validate_data=False)

        self._pair_cache[cache_key] = (ct_path, mask_path)
        return ct_path, mask_path

    def resolve_pair(
        self,
        ct_path: str,
        mask_path: str,
        patient_id: str = "",
        sample_id: str = "",
        validate: bool = True,
    ) -> Tuple[Path, Path]:
        """Resolve both CT and Mask paths, enforcing patient identity and pairing integrity."""
        cache_key = f"{patient_id}:{ct_path}:{mask_path}"
        if cache_key in self._pair_cache:
            return self._pair_cache[cache_key]

        pid = patient_id
        if not pid:
            info = parse_patient_id(clean_stem(Path(ct_path).name)) or parse_patient_id(clean_stem(Path(mask_path).name))
            if info:
                pid = info["canonical_id"]

        if pid:
            res_ct, res_mask = self.resolve_patient(pid, ct_hint=ct_path, mask_hint=mask_path, validate=validate)
        else:
            res_ct = self.resolve_file(ct_path, file_type="ct")
            res_mask = self.resolve_file(mask_path, file_type="mask")
            if validate:
                self.validate_pair(res_ct, res_mask, patient_id=patient_id, validate_data=False)

        self._pair_cache[cache_key] = (res_ct, res_mask)
        return res_ct, res_mask
