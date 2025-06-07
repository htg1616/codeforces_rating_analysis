import logging

from fetch_codeforces_problems import main as fetch_main
from tag_ppmi_visualization import main as viz_main


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def main() -> None:
    logging.info("Fetching problems from Codeforces API")
    fetch_main()
    logging.info("Running full analysis and visualization")
    viz_main()


if __name__ == "__main__":
    main()
