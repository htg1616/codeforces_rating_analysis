import csv
import hashlib
import os
import random
import time
from typing import List, Dict

import requests
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
api_key = os.getenv("CODEFORCES_API_KEY")
api_secret = os.getenv("CODEFORCES_API_SECRET")

ENDPOINT = "https://codeforces.com/api/problemset.problems"

DIFFICULTY_GROUPS = [
    (1199, "Newbie"),
    (1399, "Pupil"),
    (1599, "Specialist"),
    (1899, "Expert"),
    (2099, "Candidate Master"),
    (2299, "Master"),
    (2399, "International Master"),
]


def difficulty_group(rating: int) -> str:
    for limit, group in DIFFICULTY_GROUPS:
        if rating <= limit:
            return group
    return "Grandmaster+"


def build_params() -> Dict[str, str]:
    if not api_key or not api_secret:
        return {}
    params = {"apiKey": api_key, "time": int(time.time())}
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    rand = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    data = f"{rand}/problemset.problems?{query}#{api_secret}"
    digest = hashlib.sha512(data.encode("utf-8")).hexdigest()
    params["apiSig"] = rand + digest
    return params


def fetch_problems() -> List[Dict]:
    params = build_params()
    resp = requests.get(ENDPOINT, params=params, timeout=15)
    resp.raise_for_status()
    result = resp.json().get("result", {})
    return result.get("problems", [])


def main() -> None:
    problems = fetch_problems()
    filtered = [
        p
        for p in problems
        if p.get("rating") is not None and p.get("tags")
    ]

    sorted_problems = sorted(
        filtered,
        key=lambda x: (-int(x["contestId"]), x["index"]),
    )

    top_problems = sorted_problems[:20000]

    with open("filtered_problems.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "problem_id",
            "name",
            "contestId",
            "index",
            "rating",
            "tags",
            "difficulty_group",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for p in top_problems:
            writer.writerow(
                {
                    "problem_id": f"{p['contestId']}{p['index']}",
                    "name": p.get("name", ""),
                    "contestId": int(p["contestId"]),
                    "index": p.get("index", ""),
                    "rating": int(p["rating"]),
                    "tags": ",".join(p.get("tags", [])),
                    "difficulty_group": difficulty_group(int(p["rating"])),
                }
            )


if __name__ == "__main__":
    main()
