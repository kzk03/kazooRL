import json
import random
from pathlib import Path

import torch

# === ファイル読み込み ===
root = Path(__file__).resolve().parents[1]
github_path = root / "data/github_data.json"
graph_path = root / "data/graph.pt"
label_out = root / "data/labels.pt"

# === GitHubデータ（整形済）を読み込み ===
with open(github_path) as f:
    github = json.load(f)
    prs = github.get("prs", [])

# === GNN グラフ（dev/task）読み込み ===
data = torch.load(graph_path, weights_only=False)
dev_names = list(map(str, data["dev"].node_id))
task_names = list(map(str, data["task"].node_id))

dev2idx = {name: i for i, name in enumerate(dev_names)}
task2idx = {name: i for i, name in enumerate(task_names)}

# === 正例ペア抽出（レビュー関係） ===
positive = set()
for pr in prs:
    task_id = f"pr_{pr['number']}"
    if task_id not in task2idx:
        continue
    t_idx = task2idx[task_id]

    for r in pr.get("reviews", {}).get("nodes", []):
        author = r.get("author", {})
        if not author:
            continue
        dev_name = author["login"]
        if dev_name in dev2idx:
            d_idx = dev2idx[dev_name]
            positive.add((d_idx, t_idx))

print(f"✅ 正例の数: {len(positive)}")
if len(positive) == 0:
    print("❌ 正例がゼロです。dev名が一致していない可能性があります。")
    exit()

# === 負例をランダム生成 ===
n_dev = data["dev"].num_nodes
n_task = data["task"].num_nodes
all_pairs = set((d, t) for d in range(n_dev) for t in range(n_task))
negative = random.sample(list(all_pairs - positive), len(positive))

# === テンソル化・保存 ===
examples = list(positive) + list(negative)
labels = [1] * len(positive) + [0] * len(negative)

pairs_tensor = torch.tensor(examples, dtype=torch.long).reshape(-1, 2)
labels_tensor = torch.tensor(labels, dtype=torch.float)

torch.save((pairs_tensor, labels_tensor), label_out)
print(f"✅ ラベル保存 → {label_out}")
