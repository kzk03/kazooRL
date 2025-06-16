import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import HeteroData

data_path = Path("data/github_data.json")
profile_path = Path("configs/dev_profiles.yaml")
backlog_path = Path("data/backlog.json")
graph_out = Path("data/graph.pt")

# === データ読み込み ===
with data_path.open() as f:
    raw = json.load(f)
prs = raw.get("prs", [])
issues = raw.get("issues", [])

# プロフィール読み込み
with profile_path.open() as f:
    profiles = yaml.safe_load(f)

# バックログ読み込み（task_idのリスト）
if backlog_path.exists():
    with open(backlog_path) as f:
        backlog_data = json.load(f)
        backlog = set(x["id"] for x in backlog_data)
else:
    backlog = set()

# === ノード構築 ===
data = HeteroData()
dev_set = set()
task_ids = []
task_features = []
task_meta = []  # (id, author, kind, object)

# PRノード構築
for pr in prs:
    author = pr.get("author", {}).get("login")
    if not author:
        continue
    task_id = f"pr_{pr['number']}"
    task_ids.append(task_id)
    task_meta.append((task_id, author, "pr", pr))
    dev_set.add(author)
    task_features.append([1.0, 0.0] + [0.0] * 6)  # PR特徴量例

# Issueノード構築
for issue in issues:
    author = issue.get("author", {}).get("login")
    if not author:
        continue
    task_id = f"issue_{issue['number']}"
    task_ids.append(task_id)
    task_meta.append((task_id, author, "issue", issue))
    dev_set.add(author)
    task_features.append([0.0, 1.0] + [0.0] * 6)  # Issue特徴量例

# 開発者ノード
dev_names = sorted(list(dev_set))
dev2idx = {name: i for i, name in enumerate(dev_names)}
skill_features = []

for name in dev_names:
    p = profiles.get(name, {})
    # 'skill' キーが存在しない場合のデフォルト値を設定
    skill_p = p.get("skill", {})
    skill = [
        float(skill_p.get("code", 0.0)),
        float(skill_p.get("review", 0.0)),
    ]
    langs = p.get("lang_emb", [0.0, 0.0, 0.0])
    task_types = p.get("task_types", [0.0, 0.0, 0.0])
    feat = skill + langs + task_types
    skill_features.append(feat)

data["dev"].x = torch.tensor(skill_features, dtype=torch.float)
data["dev"].node_id = dev_names

# タスクノード
x = torch.tensor(task_features, dtype=torch.float)
node_id = task_ids
is_open = torch.tensor(
    [1.0 if tid in backlog else 0.0 for tid in node_id], dtype=torch.float
).unsqueeze(1)
data["task"].x = torch.cat([x, is_open], dim=1)
data["task"].node_id = node_id

# === エッジ構築 ===
pr_edges = []
review_edges = []
issue_edges = []
task2idx = {tid: i for i, tid in enumerate(task_ids)}

for task_id, author, kind, obj in task_meta:
    t_idx = task2idx[task_id]
    if author in dev2idx:
        d_idx = dev2idx[author]
        pr_edges.append((d_idx, t_idx))

    if kind == "pr":
        for review in obj.get("reviews", {}).get("nodes", []):
            r = review.get("author")
            if r and r.get("login") in dev2idx:
                review_edges.append((dev2idx[r["login"]], t_idx))
    elif kind == "issue":
        for a in obj.get("assignees", {}).get("nodes", []):
            if a.get("login") in dev2idx:
                issue_edges.append((dev2idx[a["login"]], t_idx))


# エッジテンソル変換
def to_edge_index(edge_list):
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


data["dev", "writes", "task"].edge_index = to_edge_index(pr_edges)
data["dev", "reviews", "task"].edge_index = to_edge_index(review_edges)
data["dev", "assigned", "task"].edge_index = to_edge_index(issue_edges)

# === 保存 ===
# 出力先のディレクトリが存在しない場合に作成
graph_out.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, graph_out)
print(f"✅ GNNグラフを保存しました → {graph_out}")
