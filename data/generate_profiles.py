import json
import random
from pathlib import Path
from collections import defaultdict
import yaml

# === パス設定 ===
root = Path(__file__).resolve().parents[1]
github_path = root / "data/github_data.json"
output_path = root / "configs/dev_profiles.yaml"

# === 設定
LANGS = ["python", "javascript", "java"]
TASK_LABELS = ["bug", "feature", "docs"]

# === GitHubデータ読み込み ===
with open(github_path) as f:
    data = json.load(f)
    prs = data.get("prs", [])

# === スキル・言語・ラベルカウント用辞書 ===
dev_stats = defaultdict(lambda: {
    "code": 0, "review": 0,
    "lang_idx": None,
    "task_types": [0] * len(TASK_LABELS)
})

# === PR解析
for pr in prs:
    author = pr.get("author", {}).get("login")
    if not author:
        continue

    # スキル
    if pr.get("state") == "MERGED":
        dev_stats[author]["code"] += 1

    # 言語検出
    labels = [l["name"].lower() for l in pr.get("labels", {}).get("nodes", [])]
    title = pr.get("title", "").lower()
    text = " ".join(labels + [title])
    for i, lang in enumerate(LANGS):
        if lang in text:
            dev_stats[author]["lang_idx"] = i
            break

    # ラベルカウント
    for i, key in enumerate(TASK_LABELS):
        if key in labels:
            dev_stats[author]["task_types"][i] += 1

    # レビュアー
    for r in pr.get("reviews", {}).get("nodes", []):
        reviewer = r.get("author", {}).get("login")
        if reviewer and r.get("state") == "APPROVED":
            dev_stats[reviewer]["review"] += 1

# === 正規化関数（0.3〜1.0）
def normalize_score(x):
    return min(1.0, max(0.3, 0.3 + 0.15 * (x ** 0.5)))

# === 出力形式へ変換
profiles = {}
for dev, stat in dev_stats.items():
    lang_idx = stat["lang_idx"] if stat["lang_idx"] is not None else 0
    lang_emb = [1.0 if i == lang_idx else 0.0 for i in range(len(LANGS))]

    profiles[dev] = {
        "skill": {
            "code": round(normalize_score(stat["code"]), 2),
            "review": round(normalize_score(stat["review"]), 2)
        },
        "lang_emb": lang_emb,
        "task_types": stat["task_types"]
    }

# === 保存
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w") as f:
    yaml.dump(profiles, f, allow_unicode=True)

print(f"✅ {len(profiles)} 件の開発者プロファイルを保存 → {output_path}")
