import json
import os
import time
from datetime import datetime, timedelta, timezone  # timezone をインポート

import requests
from dotenv import load_dotenv

# === 設定 ===
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError(
        "GITHUB_TOKENが設定されていません。.envファイルを確認してください。"
    )

HEADERS = {"Authorization": f"bearer {GITHUB_TOKEN}"}
API_URL = "https://api.github.com/graphql"

REPO_OWNER = "numpy"  # リポジトリの所有者
REPO_NAME = "numpy"  # リポジトリ名
DAYS_AGO = 5  # 収集する期間（日数）

# --- GraphQL クエリ ---
GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $since: DateTime!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: 50, after: $cursor, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo {
        endCursor
        hasNextPage
      }
      nodes {
        number
        title
        author { login }
        createdAt
        updatedAt
        additions
        deletions
        merged
        mergedAt
        mergedBy { login }
        
        timelineItems(first: 100, since: $since, itemTypes: [PULL_REQUEST_REVIEW, MERGED_EVENT]) {
          nodes {
            __typename
            ... on PullRequestReview {
              author { login }
              state
              submittedAt
            }
            ... on MergedEvent {
              actor { login }
              createdAt
            }
          }
        }
      }
    }
  }
}
"""


def run_query(query, variables):
    """GraphQLクエリを実行し、レート制限を考慮する関数"""
    for _ in range(5):  # 5回までリトライ
        response = requests.post(
            API_URL, json={"query": query, "variables": variables}, headers=HEADERS
        )
        if response.status_code == 200 and "data" in response.json():
            rate_limit = response.headers.get("x-ratelimit-remaining")
            if rate_limit:
                print(f"Rate limit remaining: {rate_limit}")
            return response.json()

        print(
            f"Query failed to run by returning code of {response.status_code}. Retrying..."
        )
        time.sleep(5)
    raise Exception(f"GraphQL query failed multiple times. Response: {response.text}")


print(
    f"Fetching data from {REPO_OWNER}/{REPO_NAME} for the last {DAYS_AGO} days using GraphQL..."
)

# --- 変更点 1 ---
# タイムゾーンをUTCに指定して日付を生成
since_date_iso = (datetime.now(timezone.utc) - timedelta(days=DAYS_AGO)).isoformat()
all_events = []
has_next_page = True
cursor = None

# --- ページネーションループ ---
while has_next_page:
    variables = {
        "owner": REPO_OWNER,
        "name": REPO_NAME,
        "since": since_date_iso,
        "cursor": cursor,
    }

    result = run_query(GRAPHQL_QUERY, variables)
    data = result["data"]["repository"]["pullRequests"]

    for pr in data["nodes"]:
        # --- 変更点 2 ---
        # 比較する両方の日付をタイムゾーンアウェア(UTC)に統一
        if datetime.fromisoformat(pr["updatedAt"].replace("Z", "+00:00")) < (
            datetime.now(timezone.utc) - timedelta(days=DAYS_AGO)
        ):
            has_next_page = False
            break

        print(f"  - Processing PR #{pr['number']}")

        # PRオープンイベント
        all_events.append(
            {
                "type": "pr_opened",
                "number": pr["number"],
                "actor": pr["author"]["login"] if pr["author"] else None,
                "created_at": pr["createdAt"],
                "data": {
                    "title": pr["title"],
                    "additions": pr["additions"],
                    "deletions": pr.get("deletions", 0),
                },
            }
        )

        # PRのマージイベント
        if pr["merged"]:
            all_events.append(
                {
                    "type": "pr_merged",
                    "number": pr["number"],
                    "actor": pr["mergedBy"]["login"] if pr["mergedBy"] else None,
                    "created_at": pr["mergedAt"],
                }
            )

        # タイムラインアイテム（レビューなど）をパース
        for item in pr["timelineItems"]["nodes"]:
            event_type = item["__typename"]
            if event_type == "PullRequestReview":
                all_events.append(
                    {
                        "type": f"pr_review_{item['state'].lower()}",
                        "number": pr["number"],
                        "actor": item["author"]["login"] if item["author"] else None,
                        "created_at": item["submittedAt"],
                    }
                )

    if has_next_page:
        page_info = data["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        cursor = page_info["endCursor"]

    if not has_next_page:
        print("No more pages.")

all_events.sort(key=lambda x: x["created_at"])

output_path = "data/expert_events.json"
with open(output_path, "w") as f:
    json.dump(all_events, f, indent=2)
print(f"\nDetailed expert events saved to {output_path}")
