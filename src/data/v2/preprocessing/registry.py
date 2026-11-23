"""Preprocessor Registry for Pipeline V2."""

from typing import Dict
from src.data.v2.preprocessing.base import BasePreprocessor
from src.data.v2.preprocessing.coronacases import CoronacasesPreprocessor
from src.data.v2.preprocessing.radiopaedia import RadiopaediaPreprocessor

PREPROCESSOR_REGISTRY: Dict[str, BasePreprocessor] = {
    "coronacases": CoronacasesPreprocessor(min_hu=-1000.0, max_hu=400.0),
    "radiopaedia": RadiopaediaPreprocessor(max_value=255.0),
}


def get_preprocessor(source: str) -> BasePreprocessor:
    """Retrieve preprocessor instance by source name.

    Parameters
    ----------
    source : str
        'coronacases' or 'radiopaedia'.

    Returns
    -------
    BasePreprocessor

    Raises
    ------
    KeyError
        If source is unregistered.
    """
    s = source.strip().lower()
    if s not in PREPROCESSOR_REGISTRY:
        raise KeyError(
            f"No preprocessor registered for source '{source}'. "
            f"Registered sources: {list(PREPROCESSOR_REGISTRY.keys())}"
        )
    return PREPROCESSOR_REGISTRY[s]
