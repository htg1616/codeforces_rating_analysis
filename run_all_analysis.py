import logging
import os

from fetch_codeforces_problems import main as fetch_main
from tag_ppmi_visualization import main as viz_main


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def main() -> None:
    logging.info("시작: Codeforces API에서 문제 데이터 가져오기")
    fetch_main()

    logging.info("시작: 전체 분석 및 시각화 실행")
    viz_main()

    # 시각화 결과 확인을 위한 정보 출력
    figures_dir = os.path.join("figures")
    if os.path.exists(figures_dir) and os.listdir(figures_dir):
        logging.info(f"시각화 완료: 결과물이 '{figures_dir}' 디렉토리에 저장되었습니다.")
        for group_dir in os.listdir(figures_dir):
            full_path = os.path.join(figures_dir, group_dir)
            if os.path.isdir(full_path):
                files = os.listdir(full_path)
                logging.info(f"  - {group_dir}: {len(files)}개 파일 ({', '.join(files[:3])}{'...' if len(files) > 3 else ''})")
    else:
        logging.warning(f"경고: '{figures_dir}' 디렉토리가 비어있거나 존재하지 않습니다.")


if __name__ == "__main__":
    main()
