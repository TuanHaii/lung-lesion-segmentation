"""Source Identification Module for Dual-Source CT Dataset (V2).

Identifies whether a CT volume originates from Coronacases or Radiopaedia
using strict rules (metadata, ID prefixes, and verified folder structure).
Never guesses based solely on data types.
"""

from typing import Any, Dict, Optional


class SourceIdentificationError(ValueError):
    """Raised when the dataset source cannot be strictly identified."""
    pass


VALID_SOURCES = {"coronacases", "radiopaedia"}


def detect_source(
    sample_id: str = "",
    patient_id: str = "",
    path: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Identify the CT source as 'coronacases' or 'radiopaedia'.

    Parameters
    ----------
    sample_id : str
        Sample identifier (e.g. 'P_coronacases_001_coronacases_001_z0085').
    patient_id : str
        Patient identifier (e.g. 'P_coronacases_001' or 'P_radiopaedia_10_85902_1').
    path : str
        File or volume path.
    metadata : Optional[Dict[str, Any]]
        Optional dictionary that may contain an explicit 'source' key.

    Returns
    -------
    str
        'coronacases' or 'radiopaedia'

    Raises
    ------
    SourceIdentificationError
        If the source cannot be determined definitively.
    """
    # 1. Explicit source in metadata
    if metadata and "source" in metadata:
        src = str(metadata["source"]).strip().lower()
        if src in VALID_SOURCES:
            return src

    # 2. Check patient_id
    pid = str(patient_id).strip().lower()
    if pid.startswith("p_coronacases") or pid.startswith("coronacases"):
        return "coronacases"
    if pid.startswith("p_radiopaedia") or pid.startswith("radiopaedia"):
        return "radiopaedia"

    # 3. Check sample_id
    sid = str(sample_id).strip().lower()
    if "coronacases" in sid:
        return "coronacases"
    if "radiopaedia" in sid:
        return "radiopaedia"

    # 4. Check path
    p = str(path).strip().lower()
    if "coronacases" in p:
        return "coronacases"
    if "radiopaedia" in p:
        return "radiopaedia"

    raise SourceIdentificationError(
        f"Cannot identify dataset source from metadata. "
        f"sample_id='{sample_id}', patient_id='{patient_id}', path='{path}'. "
        f"Valid sources are: {sorted(list(VALID_SOURCES))}"
    )
