import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# === 設定 ===
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"bearer {GITHUB_TOKEN}"}

OWNER = "pytest-dev"
REPO = "pytest"
START_MONTH = "2023-05"
END_MONTH = "2024-04"

# === GraphQLクエリ ===
PR_QUERY = """
query($cursor: String) {
  repository(owner: \"%s\", name: \"%s\") {
    pullRequests(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        title
        body
        state
        createdAt
        updatedAt
        closedAt
        mergedAt
        isDraft
        isCrossRepository
        mergeable
        merged
        mergedBy { login }
        baseRefName
        headRefName
        author { login }
        assignees(first: 10) { nodes { login } }
        labels(first: 10) { nodes { name } }
        milestone { title dueOn }
        comments { totalCount }
        commits(last: 1) {
          nodes {
            commit {
              message
              committedDate
              author { name email user { login } }
            }
          }
        }
        reviewRequests(first: 10) {
          nodes {
            requestedReviewer { ... on User { login } }
          }
        }
        reviews(first: 20) {
          nodes {
            author { login }
            state
            submittedAt
            body
          }
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
""" % (OWNER, REPO)

ISSUE_QUERY = """
query($cursor: String) {
  repository(owner: \"%s\", name: \"%s\") {
    issues(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        title
        body
        state
        createdAt
        updatedAt
        closedAt
        author { login }
        assignees(first: 10) { nodes { login } }
        labels(first: 10) { nodes { name } }
        milestone { title dueOn }
        comments { totalCount }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
""" % (OWNER, REPO)

# === データ収集関数 ===
def fetch_all_items(query, item_key):
    items = []
    cursor = None
    page = 1

    while True:
        print(f"🔄 Fetching {item_key} page {page}...")

        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://api.github.com/graphql",
                    headers=HEADERS,
                    json={"query": query, "variables": {"cursor": cursor}},
                )
                if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers and resp.headers["X-RateLimit-Remaining"] == "0":
                    reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait_seconds = max(0, reset_time - int(time.time()))
                    print(f"⛔ GitHub API レート制限に達しました。{wait_seconds} 秒待機します...")
                    time.sleep(wait_seconds + 5)
                    continue
                if resp.status_code in [502, 503, 504]:
                    raise Exception(f"Temporary error (status {resp.status_code})")
                data = resp.json()
                break
            except Exception as e:
                print(f"⏳ Retry {attempt + 1}/3 after error: {e}")
                time.sleep(3)
        else:
            print("❌ 最大リトライ回数を超えました")
            break

        if "errors" in data:
            print(f"❌ GraphQL error in {item_key}:", data["errors"])
            break

        nodes = data["data"]["repository"][item_key]["nodes"]
        items += nodes

        page_info = data["data"]["repository"][item_key]["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
        page += 1

    print(f"✅ {item_key} total: {len(items)}")
    return items

# === 年間フィルター ===
def filter_by_year_range(items, start_month, end_month):
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")
    filtered = []
    for item in items:
        created = item.get("createdAt")
        if not created:
            continue
        dt = datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ")
        if start_dt <= dt <= end_dt:
            filtered.append(item)
    print(f"📅 {start_month}〜{end_month} に該当する件数: {len(filtered)}")
    return filtered

# === 実行 ===
if __name__ == "__main__":
    print(f"🔎 {OWNER}/{REPO} の PR と Issue を取得中...")
    all_prs = fetch_all_items(PR_QUERY, "pullRequests")
    all_issues = fetch_all_items(ISSUE_QUERY, "issues")

    year_prs = filter_by_year_range(all_prs, START_MONTH, END_MONTH)
    year_issues = filter_by_year_range(all_issues, START_MONTH, END_MONTH)

    save_path = Path(f"data/github_data_{START_MONTH}_to_{END_MONTH}.json")
    save_path.parent.mkdir(exist_ok=True)
    json.dump({"prs": year_prs, "issues": year_issues}, save_path.open("w"), indent=2)
    print(f"✅ {START_MONTH}〜{END_MONTH} の PR + Issue データを保存しました → {save_path}")
