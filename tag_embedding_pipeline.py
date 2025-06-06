from collections import Counter
import math
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import umap


def parse_tags(tag_str: str) -> List[str]:
    """Parse a comma separated tag string into a list of tags."""
    if not isinstance(tag_str, str):
        return []
    return [t.strip() for t in tag_str.split(',') if t.strip()]


def build_tag_list(tags_series: pd.Series) -> List[str]:
    """Build sorted unique tag list from a pandas Series of tag lists."""
    tag_set = set()
    for tags in tags_series:
        tag_set.update(tags)
    return sorted(tag_set)


def build_cooccurrence_matrix(tag_lists: List[List[str]], tag_to_idx: Dict[str, int]) -> np.ndarray:
    """Create a co-occurrence matrix from list of tag lists."""
    size = len(tag_to_idx)
    C = np.zeros((size, size), dtype=int)
    for tags in tag_lists:
        # remove duplicate tags within the same problem if any
        unique_tags = list(dict.fromkeys(tags))
        n = len(unique_tags)
        for i in range(n):
            idx_i = tag_to_idx[unique_tags[i]]
            for j in range(i + 1, n):
                idx_j = tag_to_idx[unique_tags[j]]
                C[idx_i, idx_j] += 1
                C[idx_j, idx_i] += 1
    return C


def compute_ppmi_matrix(C: np.ndarray) -> np.ndarray:
    """Compute PPMI matrix from co-occurrence counts."""
    total = C.sum()
    if total == 0:
        return np.zeros_like(C, dtype=float)
    row_sums = C.sum(axis=1)
    size = C.shape[0]
    M = np.zeros((size, size), dtype=float)
    for i in range(size):
        if row_sums[i] == 0:
            continue
        P_i = row_sums[i] / total
        for j in range(size):
            if C[i, j] == 0 or row_sums[j] == 0:
                continue
            P_j = row_sums[j] / total
            P_ij = C[i, j] / total
            pmi = math.log(P_ij / (P_i * P_j))
            if pmi > 0:
                M[i, j] = pmi
    return M


def main() -> None:
    # 1. Load data
    df = pd.read_csv('filtered_problems.csv')
    df['tags'] = df['tags'].apply(parse_tags)

    # 2. Build unique tag list
    tag_list = build_tag_list(df['tags'])
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}

    # Tag frequency counter
    tag_counter = Counter()
    for tags in df['tags']:
        tag_counter.update(tags)
    n_problems = len(df)

    # Save tag list with frequency
    tag_freq_df = pd.DataFrame({'tag': tag_list, 'freq': [tag_counter[t] for t in tag_list]})
    tag_freq_df.to_csv('tag_list.csv', index=False)

    # 3. Co-occurrence matrix
    C = build_cooccurrence_matrix(df['tags'].tolist(), tag_to_idx)
    coocc_df = pd.DataFrame(C, index=tag_list, columns=tag_list)
    coocc_df.to_csv('cooccurrence_matrix.csv')

    # 4. PPMI matrix
    M = compute_ppmi_matrix(C)
    ppmi_df = pd.DataFrame(M, index=tag_list, columns=tag_list)
    ppmi_df.to_csv('ppmi_matrix.csv')

    # 5. Truncated SVD for embeddings
    # Set the number of components to the smaller of 50 or the number
    # of unique tags to avoid n_components > n_features errors.
    n_components = min(50, len(tag_list))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tag_embeddings = svd.fit_transform(M)

    # 6. Save embeddings
    emb_columns = [f'dim_{i+1}' for i in range(tag_embeddings.shape[1])]
    emb_df = pd.DataFrame(tag_embeddings, columns=emb_columns)
    emb_df.insert(0, 'tag', tag_list)
    emb_df.to_csv('tag_embeddings_50d.csv', index=False)

    # 7. UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=2, random_state=42)
    tag_coords = reducer.fit_transform(tag_embeddings)

    # 8. Frequency, ratio, marker size
    freqs = np.array([tag_counter[t] for t in tag_list])
    ratios = freqs / n_problems
    marker_sizes = ratios * 1000 + 50

    coords_df = pd.DataFrame({
        'tag': tag_list,
        'x': tag_coords[:, 0],
        'y': tag_coords[:, 1],
        'freq': freqs,
        'ratio': ratios,
        'marker_size': marker_sizes,
    })
    coords_df.to_csv('tag_coords_umap.csv', index=False)

    # 9. Edge list from top PPMI values
    upper_indices = np.triu_indices_from(M, k=1)
    weights = M[upper_indices]
    edge_tuples = []
    for (i, j), w in zip(zip(*upper_indices), weights):
        if w > 0:
            edge_tuples.append((tag_list[i], tag_list[j], w))
    edge_tuples.sort(key=lambda x: x[2], reverse=True)
    topk = 100
    top_edges = edge_tuples[:topk]
    edge_df = pd.DataFrame(top_edges, columns=['tag1', 'tag2', 'weight'])
    edge_df.to_csv('edges_ppmi_topk.csv', index=False)


if __name__ == '__main__':
    main()
