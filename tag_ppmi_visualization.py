"""PPMI-based visualizations for Codeforces tags.

This script loads ``filtered_problems.csv`` from ``data/problems`` which
contains Codeforces problem metadata.  It assumes **37** unique tags are
present.  For each difficulty group as well as the entire data set the script
produces

1. a co-occurrence matrix and PPMI matrix (min-max scaled)
2. an improved force-directed graph
3. several improved UMAP projections

Images are stored under ``figures/<group>`` and CSV summaries under ``data/<group>``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Dict

import networkx as nx
from networkx.algorithms import community
import numpy as np
import pandas as pd

from data_processing import (
    parse_tags,
    build_tag_list,
    build_cooccurrence_matrix,
    compute_ppmi_matrix,
    compute_tag_frequency,
)
from vis_force_directed import draw_force_directed
from vis_umap import draw_umap_plotly
from vis_heatmap import draw_heatmap

# Paths
PROBLEMS_PATH = os.path.join("data", "problems", "filtered_problems.csv")
VIS_DIR = os.path.join("figures")  # data/visualization에서 figures로 변경
os.makedirs(VIS_DIR, exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def run_for_group(
    tag_lists: List[List[str]],
    tag_list: List[str],
    tag_to_idx: Dict[str, int],
    prefix: str,
    out_root: str,
) -> None:
    """Compute matrices and visualizations for a subset of problems."""
    start_time = time.time()
    logging.info(f"시작: {prefix} 그룹 처리 중...")

    # 시각화 결과물 저장 경로
    group_dir = os.path.join(out_root, prefix)
    os.makedirs(group_dir, exist_ok=True)

    # 데이터 파일 저장 경로
    data_dir = os.path.join("data", prefix)
    os.makedirs(data_dir, exist_ok=True)

    # 이 그룹에서 실제 사용된 태그만 필터링
    used_tags = set()
    for tags in tag_lists:
        used_tags.update(tags)

    # 전체 태그 리스트에서 이 그룹에 존재하는 태그만 필터링
    filtered_tag_list = [tag for tag in tag_list if tag in used_tags]
    filtered_tag_to_idx = {tag: idx for idx, tag in enumerate(filtered_tag_list)}

    # 존재하는 태그가 없으면 경고 출력 후 리턴
    if not filtered_tag_list:
        logging.warning(f"  {prefix}: 사용된 태그가 없습니다.")
        return

    logging.info(f"  {prefix}: 전체 태그 {len(tag_list)}개 중 {len(filtered_tag_list)}개 사용됨")

    # ----- Step 1: build co-occurrence and PPMI matrices with filtered tags -----
    logging.info(f"  {prefix}: 공출현 행렬 계산 중...")
    C = build_cooccurrence_matrix(tag_lists, filtered_tag_to_idx)

    logging.info(f"  {prefix}: PPMI 행렬 계산 중...")
    M = compute_ppmi_matrix(C)

    # ----- Step 2: tag frequencies and marker size -----
    logging.info(f"  {prefix}: 태그 빈도 계산 중...")
    tag_freq = compute_tag_frequency(tag_lists)
    n_problems = len(tag_lists)
    ratios = {t: (tag_freq.get(t, 0) / n_problems) if n_problems else 0 for t in filtered_tag_list}
    marker_sizes = {t: max(ratios[t] * 1000 + 50, 40) for t in filtered_tag_list}

    # CSV 파일 저장 - data/<group> 디렉토리에 저장
    tag_list_df = pd.DataFrame({
        "tag": filtered_tag_list,
        "freq": [tag_freq.get(t, 0) for t in filtered_tag_list],
        "ratio": [ratios[t] for t in filtered_tag_list],
        "marker_size": [marker_sizes[t] for t in filtered_tag_list],
    })
    tag_list_df.to_csv(os.path.join(data_dir, "tag_list.csv"), index=False)

    pd.DataFrame(C, index=filtered_tag_list, columns=filtered_tag_list).to_csv(
        os.path.join(data_dir, "cooccurrence_matrix.csv")
    )
    pd.DataFrame(M, index=filtered_tag_list, columns=filtered_tag_list).to_csv(
        os.path.join(data_dir, "ppmi_matrix.csv")
    )

    # prepare weighted tag pairs using Top-K method
    logging.info(f"  {prefix}: Top-K 간선 선택 중...")
    ppmi_raw = M.copy()
    n = len(filtered_tag_list)  # 필터링된 태그 리스트의 길이를 사용
    flat = ppmi_raw[np.triu_indices_from(ppmi_raw, k=1)]
    top_k = min(120, len(flat))  # flat이 비어있을 경우 처리
    idx_k = np.argpartition(flat, -top_k)[-top_k:] if len(flat) > top_k else np.arange(len(flat))
    thr = flat[idx_k].min() if len(flat) > 0 else 0

    # 간선 필터링 - 필터링된 태그 리스트를 사용
    pair_weights = [
        (filtered_tag_list[i], filtered_tag_list[j], ppmi_raw[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if ppmi_raw[i, j] >= thr and ppmi_raw[i, j] > 0
    ]

    # 디버깅용 로그
    logging.info(f"  {prefix}: 간선 수: {len(pair_weights)}, 임계값: {thr:.4f}")

    # G 객체 구성 (기존 코드 그대로)
    G = nx.Graph()
    for t_i, t_j, w in pair_weights:
        G.add_edge(t_i, t_j, weight=w)

    # 디버깅용 추가 정보
    logging.info(f"  {prefix}: 고립 노드 수: {sum(1 for _, d in G.degree() if d == 0)}")

    # ----- Build graph and Louvain clusters -----
    G = nx.Graph()
    for t_i, t_j, w in pair_weights:
        G.add_edge(t_i, t_j, weight=w)

    deg_cent = nx.degree_centrality(G)
    communities = community.louvain_communities(G, weight="weight", seed=42)
    cluster_id = {n: idx for idx, com in enumerate(communities) for n in com}

    pd.DataFrame(pair_weights, columns=["tag_i", "tag_j", "weight"]).to_csv(
        os.path.join(data_dir, "edges_all.csv"), index=False
    )

    # ----- Visualizations -----
    logging.info(f"  {prefix}: Force-directed 그래프 생성 중...")
    draw_force_directed(G, marker_sizes, deg_cent, cluster_id, group_dir, data_dir)

    logging.info(f"  {prefix}: UMAP 시각화 생성 중...")
    try:
        draw_umap_plotly(
            M,
            filtered_tag_list,  # tag_list 대신 filtered_tag_list 사용
            marker_sizes,
            tag_freq,
            ratios,
            deg_cent,
            cluster_id,
            pair_weights,
            group_dir,
            data_dir,
        )
    except Exception as e:
        logging.error(f"  {prefix}: UMAP 시각화 실패: {str(e)}")

    logging.info(f"  {prefix}: 히트맵 생성 중...")
    try:
        draw_heatmap(M, filtered_tag_list, tag_freq, cluster_id, filtered_tag_to_idx, group_dir)
    except Exception as e:
        logging.error(f"  {prefix}: 히트맵 시각화 실패: {str(e)}")
    elapsed = time.time() - start_time
    logging.info(f"완료: {prefix} 그룹 처리 (소요시간: {elapsed:.2f}초)")


# ------------------------- Main Pipeline -------------------------

def main() -> None:
    logging.info("프로그램 시작")
    # 1. CSV 파일 읽기
    df = pd.read_csv(
        PROBLEMS_PATH,
        usecols=[
            'problem_id',
            'name',
            'contestId',
            'index',
            'rating',
            'tags',
            'difficulty_group',
        ],
    )
    df['tags'] = df['tags'].apply(parse_tags)

    # 2. 고유 태그 추출 (총 37개)
    tag_list = build_tag_list(df['tags'])
    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_list)}

    logging.info("전체 데이터 시각화 시작")
    run_for_group(df['tags'].tolist(), tag_list, tag_to_idx, 'All', VIS_DIR)

    logging.info("난이도 그룹별 시각화 시작")
    difficulty_groups = df['difficulty_group'].unique()
    logging.info(f"총 {len(difficulty_groups)}개 난이도 그룹 처리 예정: {difficulty_groups}")

    for i, (group, df_group) in enumerate(df.groupby('difficulty_group')):
        safe_group = str(group).replace(' ', '_').replace('/', '_')
        logging.info(f"난이도 그룹 처리 중 ({i+1}/{len(difficulty_groups)}): {group}")
        run_for_group(
            df_group['tags'].tolist(),
            tag_list,
            tag_to_idx,
            safe_group,
            VIS_DIR,
        )

    logging.info("모든 처리 완료")


if __name__ == '__main__':
    main()
