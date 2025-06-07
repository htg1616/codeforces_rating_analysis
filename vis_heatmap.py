"""Heatmap visualization for Codeforces tag PPMI."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
import os
import logging  # 상단으로 이동

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
    # 유효성 검사 추가
    if len(tag_list) < 2:
        logging.warning("  Heatmap: 태그가 부족하여 히트맵을 생성할 수 없습니다.")
        return

    try:
        # 태그 필터링 - tag_freq와 cluster_id, tag_to_idx에 모두 있는 태그만 사용
        valid_tags = [t for t in tag_list
                    if t in tag_freq and t in cluster_id and t in tag_to_idx
                    and tag_to_idx[t] < M.shape[0]]

        if len(valid_tags) < 2:
            logging.warning("  Heatmap: 유효한 태그가 부족하여 히트맵을 생성할 수 없습니다.")
            return

        # 상위 15개 태그 선택 (또는 가용한 모든 태그)
        top_n = min(15, len(valid_tags))
        top_tags = [t for t, _ in sorted(
            [(t, tag_freq.get(t, 0)) for t in valid_tags],
            key=lambda x: -x[1]
        )[:top_n]]

        if not top_tags:
            logging.warning("  Heatmap: 표시할 태그가 없습니다.")
            return

        # 기존 코드 - 클러스터별로 태그 그룹화
        cluster_tags: Dict[int, List[str]] = defaultdict(list)
        for t in top_tags:
            cluster_tags[cluster_id[t]].append(t)

        # 클러스터별로 정렬된 태그 목록 생성
        ordered_tags: List[str] = []
        for cid in sorted(cluster_tags.keys()):
            tags = cluster_tags[cid]
            # 안전하게 정렬 - 유효한 인덱스만 사용
            tags.sort(key=lambda x: -M[tag_to_idx[x]].sum() if x in tag_to_idx and tag_to_idx[x] < M.shape[0] else 0)
            ordered_tags.extend(tags)

        # 유효한 인덱스만 추출
        idxs = [tag_to_idx[t] for t in ordered_tags if t in tag_to_idx and tag_to_idx[t] < M.shape[0]]
        if len(idxs) < 2:
            logging.warning("  Heatmap: 유효한 인덱스가 부족하여 히트맵을 생성할 수 없습니다.")
            return

        # PPMI 부분행렬 추출
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
        plt.suptitle(f"PPMI Heatmap (Top {len(ordered_tags)} Tags)", fontsize=14, y=0.95)

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
        out_path = os.path.join(out_dir, "ppmi_heatmap.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"  Heatmap: 저장 완료: {out_path}")

    except Exception as e:
        logging.error(f"  Heatmap: 예상치 못한 오류 발생: {str(e)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Draw heatmap for a difficulty group")
    parser.add_argument("group", help="Difficulty group name, e.g. All or Pupil")
    args = parser.parse_args()

    in_dir = os.path.join("data", args.group)
    out_dir = os.path.join("figures", args.group)
    os.makedirs(out_dir, exist_ok=True)

    tag_df = pd.read_csv(os.path.join(in_dir, "tag_list.csv"))
    tag_list = tag_df["tag"].tolist()
    tag_freq = dict(zip(tag_df["tag"], tag_df["freq"]))
    cluster_id = dict(zip(tag_df["tag"], tag_df["cluster_id"]))
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}

    M = pd.read_csv(os.path.join(in_dir, "ppmi_matrix.csv"), index_col=0).values

    draw_heatmap(M, tag_list, tag_freq, cluster_id, tag_to_idx, out_dir)


if __name__ == "__main__":
    main()
