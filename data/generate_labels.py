import json
import random
from pathlib import Path

import torch

# === パス設定
root = Path(__file__).resolve().parents[1]
graph_path = root / "data/graph.pt"
github_path = root / "data/github_data.json"
label_out = root / "data/labels.pt"

# === データ読み込み
data = torch.load(graph_path, weights_only=False)
with open(github_path) as f:
    prs = json.load(f)["data"]["repository"]["pullRequests"]["nodes"]

# === node_id（GitHubユーザ名）を使った dev2idx を作成
dev_names_raw = data["dev"].node_id
dev_names = [str(name) for name in dev_names_raw]
dev2idx = {name: idx for idx, name in enumerate(dev_names)}
# PR番号に基づいて task2idx を構築
with open("data/github_data.json") as f:
    prs = json.load(f)["data"]["repository"]["pullRequests"]["nodes"]

task2idx = {f"task_{pr['number']}": idx for idx, pr in enumerate(prs)}


# === 正例作成
positive = set()
for pr in prs:
    task_id = f"task_{pr['number']}"
    if task_id not in task2idx:
        continue
    t_idx = task2idx[task_id]
    for r in pr.get("reviews", {}).get("nodes", []):
        if not r.get("author"):
            continue
        dev = r["author"]["login"]
        if dev in dev2idx:
            d_idx = dev2idx[dev]
            positive.add((d_idx, t_idx))

print(f"✅ 正例の数: {len(positive)}")
if len(positive) == 0:
    print("❌ 正例がゼロです。dev名が一致していない可能性があります。")
    exit()

# === 負例サンプリング（ランダム）
n_dev = data["dev"].num_nodes
n_task = data["task"].num_nodes
all_pairs = set((d, t) for d in range(n_dev) for t in range(n_task))
negative = random.sample(list(all_pairs - positive), len(positive))

# === テンソル化
examples = list(positive) + list(negative)
labels = [1] * len(positive) + [0] * len(negative)
pairs = torch.tensor(examples, dtype=torch.long).reshape(-1, 2)
labels = torch.tensor(labels, dtype=torch.float)

# === 保存
torch.save((pairs, labels), label_out)
print(f"✅ ラベル保存 → {label_out}")
