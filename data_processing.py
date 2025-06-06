# Utility functions for Codeforces tag analysis

from __future__ import annotations

from collections import Counter
from typing import Dict, List
import numpy as np
import pandas as pd


def parse_tags(tag_str: str) -> List[str]:
    """Convert comma separated tag string to list of tags."""
    if isinstance(tag_str, str):
        return [t.strip() for t in tag_str.split(',') if t.strip()]
    return []


def build_tag_list(tags_series: pd.Series) -> List[str]:
    """Extract sorted unique tags from pandas Series."""
    tag_set = set()
    for tags in tags_series:
        tag_set.update(tags)
    tag_list = sorted(tag_set)
    assert len(tag_list) == 37, "Expected 37 unique tags"
    return tag_list


def build_cooccurrence_matrix(tag_lists: List[List[str]], tag_to_idx: Dict[str, int]) -> np.ndarray:
    """Return co-occurrence matrix."""
    size = len(tag_to_idx)
    C = np.zeros((size, size), dtype=int)
    for tags in tag_lists:
        unique_tags = list(dict.fromkeys(tags))
        for i, tag_i in enumerate(unique_tags):
            idx_i = tag_to_idx[tag_i]
            for tag_j in unique_tags[i + 1:]:
                idx_j = tag_to_idx[tag_j]
                C[idx_i, idx_j] += 1
                C[idx_j, idx_i] += 1
    return C


def compute_ppmi_matrix(C: np.ndarray) -> np.ndarray:
    """Compute a min-max scaled PPMI matrix."""
    total = float(C.sum())
    if total == 0:
        return np.zeros_like(C, dtype=float)

    row_sums = C.sum(axis=1).astype(float)
    size = C.shape[0]
    M = np.zeros((size, size), dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(size):
            if row_sums[i] == 0:
                continue
            P_i = row_sums[i] / total
            for j in range(size):
                if C[i, j] == 0 or row_sums[j] == 0:
                    continue
                P_j = row_sums[j] / total
                P_ij = C[i, j] / total
                pmi = np.log(P_ij / (P_i * P_j))
                if pmi > 0:
                    M[i, j] = pmi

    max_val = M.max()
    if max_val > 0:
        M /= max_val
    return M


def compute_tag_frequency(tag_lists: List[List[str]]) -> Counter:
    """Return Counter of tag frequencies."""
    counter = Counter()
    for tags in tag_lists:
        counter.update(tags)
    return counter
