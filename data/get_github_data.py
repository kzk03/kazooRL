# ✅ get_github_data.py
# GitHub GraphQL API を使って、PRとレビュー情報を取得

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# === .env から GITHUB_TOKEN を読み込む
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise RuntimeError("❌ GITHUB_TOKEN が .env に見つかりません")

headers = {"Authorization": f"Bearer {token}"}

# === 対象 OSS リポジトリ（例: pallets/flask）
owner = "pallets"
name = "flask"

# === GraphQL クエリ
query = """
query($owner: String!, $name: String!, $n: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequests(first: $n, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        title
        author {
          login
        }
        mergedAt
        reviews(first: 10) {
          nodes {
            author {
              login
            }
          }
        }
      }
    }
  }
}
"""

variables = {
    "owner": owner,
    "name": name,
    "n": 100  # 最新100件のPRを取得
}

# === API リクエスト
response = requests.post(
    "https://api.github.com/graphql",
    headers=headers,
    json={"query": query, "variables": variables}
)

if response.status_code != 200:
    raise RuntimeError(f"❌ APIエラー: {response.status_code} → {response.text}")

result = response.json()

# === 保存
out_path = Path("data/github_data.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"✅ データ保存完了 → {out_path}")
