import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# .env をプロジェクトルートから読み込む
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

# トークン取得
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

token = os.getenv("GITHUB_TOKEN")
print("✅ GITHUB_TOKEN:", token)
print("✅ ASCII only?:", all(ord(c) < 128 for c in token))
if not GITHUB_TOKEN:
    raise ValueError("❌ .env に GITHUB_TOKEN が設定されていません")

# GraphQLヘッダーとクエリ
headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

query = """
query {
  repository(owner: "pallets", name: "flask") {
    pullRequests(first: 50, states: MERGED) {
      nodes {
        number
        title
        author { login }
        mergedAt
        reviews(first: 10) {
          nodes {
            author { login }
            submittedAt
          }
        }
      }
    }
  }
}
"""

# リクエスト送信
response = requests.post(
    "https://api.github.com/graphql",
    json={"query": query},
    headers=headers
)

# レスポンス確認と保存
if response.status_code == 200:
    data = response.json()
    save_path = Path(__file__).resolve().parents[1] / "data/github_data.json"
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ データ取得成功！→ {save_path}")
else:
    print(f"❌ APIエラー {response.status_code}")
    print(response.text)
