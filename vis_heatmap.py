"""Heatmap visualization for Codeforces tag PPMI."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd

# 한글 폰트 설정 추가
import platform

# 운영체제별 기본 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우의 경우 '맑은 고딕' 폰트 사용
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')  # macOS의 경우 'AppleGothic' 사용
else:  # Linux 등 기타 운영체제
    # 나눔글꼴이나 다른 한글 폰트가 설치되어 있다면 사용
    try:
        plt.rc('font', family='NanumGothic')
    except:
        pass  # 설치된 한글 폰트가 없으면 기본값 사용

# 음수 표시 문제 해결
mpl.rcParams['axes.unicode_minus'] = False

def draw_heatmap(
    M: np.ndarray,
    tag_list: List[str],
    tag_freq: Dict[str, int],
    cluster_id: Dict[str, int],
    tag_to_idx: Dict[str, int],
    out_dir: str,
) -> None:
    """PPMI heatmap for top 15 tags with cluster color bars."""

    top_tags = [t for t, _ in sorted(tag_freq.items(), key=lambda x: -x[1])[:15]]

    cluster_tags: Dict[int, List[str]] = defaultdict(list)
    for t in top_tags:
        cluster_tags[cluster_id[t]].append(t)

    ordered_tags: List[str] = []
    for cid in sorted(cluster_tags.keys()):
        tags = cluster_tags[cid]
        tags.sort(key=lambda x: -M[tag_to_idx[x]].sum())
        ordered_tags.extend(tags)

    idxs = [tag_to_idx[t] for t in ordered_tags]
    sub = M[np.ix_(idxs, idxs)]

    # 클러스터 색상 맵 생성
    colors = px.colors.qualitative.Dark24

    # 각 태그별 클러스터 ID에 따른 색상 매핑 생성
    cluster_colors = {tag: colors[cluster_id[tag] % len(colors)] for tag in ordered_tags}

    # 데이터프레임 생성 및 클러스터 열 추가
    df_heatmap = pd.DataFrame(sub, index=ordered_tags, columns=ordered_tags)

    # 클러스터 정보 시리즈 생성 (열과 행에 사용)
    cluster_series = pd.Series({tag: cluster_id[tag] for tag in ordered_tags}, name='Cluster')

    # 클러스터 색상 팔레트 생성
    unique_clusters = sorted(set(cluster_id[tag] for tag in ordered_tags))
    cluster_palette = {cid: colors[cid % len(colors)] for cid in unique_clusters}

    # clustermap 생성
    cm = sns.clustermap(
        df_heatmap,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        figsize=(12, 10),
        row_cluster=False,  # 행 클러스터링 끄기 (원래 순서 유지)
        col_cluster=False,  # 열 클러스터링 끄기 (원래 순서 유지)
        row_colors=cluster_series.map(cluster_palette),  # 행에 클러스터 색상 추가
        col_colors=cluster_series.map(cluster_palette),  # 열에 클러스터 색상 추가
        annot_kws={"size": 9},
        cbar_kws={"label": "PPMI 값"},
        xticklabels=1,  # 모든 x 라벨 표시
        yticklabels=1   # 모든 y 라벨 표시
    )

    # X축 라벨 회전
    cm.ax_heatmap.set_xticklabels(
        cm.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor'
    )

    # 제목 추가
    plt.suptitle("PPMI Heatmap (Top 15 Tags)", fontsize=14, y=0.95)

    # 클러스터 범례 추가
    handles = [
        mpl.patches.Patch(color=cluster_palette[cid], label=f'Cluster {cid}')
        for cid in sorted(unique_clusters)
    ]
    plt.legend(
        handles=handles,
        title="Clusters",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # 여백 조정
    plt.tight_layout()

    # 저장 및 종료
    plt.savefig(os.path.join(out_dir, "ppmi_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()
