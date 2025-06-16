import json
from collections import defaultdict
from pathlib import Path

import yaml


def extract_profiles(github_data):
    # (extract_profiles関数の中身は変更なし)
    profiles = defaultdict(
        lambda: {
            "code": 0.1,
            "review": 0.1,
            "lang_emb": [1.0, 0.0, 0.0],
            "task_types": [0, 0, 0],
        }
    )

    for pr in github_data.get("prs", []):
        author = pr.get("author", {}).get("login")
        if author and "bot" not in author.lower():
            profiles[author]["code"] += 0.1

        for review in pr.get("reviews", {}).get("nodes", []):
            reviewer = review.get("author", {}).get("login")
            if reviewer and "bot" not in reviewer.lower():
                profiles[reviewer]["review"] += 0.05

        for commit_node in pr.get("commits", {}).get("nodes", []):
            user_info = commit_node.get("commit", {}).get("author", {}).get("user")
            login = user_info.get("login") if user_info else None
            if login and "bot" not in login.lower():
                profiles[login]["code"] += 0.05

    for profile in profiles.values():
        profile["code"] = min(1.0, round(profile["code"], 2))
        profile["review"] = min(1.0, round(profile["review"], 2))

    return profiles


def main():
    input_path = Path("data/github_data.json")
    output_path = Path("configs/dev_profiles.yaml")

    with input_path.open() as f:
        github_data = json.load(f)

    profiles = extract_profiles(github_data)

    # 出力先のディレクトリが存在しない場合に作成する
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        yaml.dump(dict(profiles), f)


if __name__ == "__main__":
    main()
