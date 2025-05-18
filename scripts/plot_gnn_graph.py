# scripts/plot_gnn_graph.py

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

# --- グラフ読み込み ---
graph_path = Path("data/graph.pt")
data = torch.load(graph_path, weights_only=False)

# --- NetworkX グラフ構築 ---
G = nx.DiGraph()

# ノード追加
for node_type in data.node_types:
    for i in range(data[node_type].num_nodes):
        G.add_node(f"{node_type}_{i}", node_type=node_type)

# エッジ追加
for edge_type in data.edge_types:
    src_type, rel_type, dst_type = edge_type
    edge_index = data[edge_type].edge_index
    for i in range(edge_index.size(1)):
        src = f"{src_type}_{edge_index[0, i].item()}"
        dst = f"{dst_type}_{edge_index[1, i].item()}"
        G.add_edge(src, dst, rel_type=rel_type)

# --- 部分グラフ抽出（表示負荷軽減） ---
sub_nodes = [
    n for n in G.nodes if
    (n.startswith("dev_") and int(n.split("_")[1]) < 30) or
    (n.startswith("task_") and int(n.split("_")[1]) < 30)
]
sub_G = G.subgraph(sub_nodes)

# --- 可視化 ---
pos = nx.spring_layout(sub_G, seed=42, k=0.8)
node_colors = ["skyblue" if n.startswith("dev_") else "lightgreen" for n in sub_G.nodes]

plt.figure(figsize=(16, 12))
nx.draw(sub_G, pos, node_color=node_colors, node_size=400, with_labels=False, alpha=0.9, arrows=True)

# ノードラベル（名前）を表示
nx.draw_networkx_labels(sub_G, pos, font_size=8, font_color="black")

plt.title("Heterogeneous GNN Graph (with Node Labels)", fontsize=15)
plt.axis("off")
plt.tight_layout()
plt.show()
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

# scripts/plot_gnn_with_labels.py
import torch
from matplotlib.lines import Line2D

# === グラフ読み込み ===
graph_path = Path("data/graph.pt")
data = torch.load(graph_path, weights_only=False)

# === ノード名取得 ===
dev_names = data["dev"].node_id
num_devs = data["dev"].num_nodes
num_tasks = data["task"].num_nodes

# === NetworkX グラフ構築 ===
G = nx.DiGraph()

# ノード追加（GitHub名付き）
for i in range(num_devs):
    G.add_node(f"dev_{i}", label=dev_names[i], node_type="dev")
for i in range(num_tasks):
    G.add_node(f"task_{i}", label=f"task_{i}", node_type="task")

# === エッジ追加（種類で色分けのため記録） ===
edge_colors = []
edge_types = []
color_map = {
    "writes": "cornflowerblue",
    "reviews": "mediumseagreen",
    "written_by": "lightgray",
    "reviewed_by": "orange"
}

for edge_type in data.edge_types:
    src_type, rel_type, dst_type = edge_type
    edge_index = data[edge_type].edge_index
    for i in range(edge_index.size(1)):
        src = f"{src_type}_{edge_index[0, i].item()}"
        dst = f"{dst_type}_{edge_index[1, i].item()}"
        G.add_edge(src, dst, rel_type=rel_type)
        edge_colors.append(color_map.get(rel_type, "gray"))
        edge_types.append(rel_type)

# === 部分グラフ（dev 30人 + task 30件） ===
sub_nodes = [
    n for n in G.nodes if
    (n.startswith("dev_") and int(n.split("_")[1]) < 30) or
    (n.startswith("task_") and int(n.split("_")[1]) < 30)
]
sub_G = G.subgraph(sub_nodes).copy()

# === 可視化 ===
pos = nx.spring_layout(sub_G, seed=42, k=0.8)
node_colors = ["skyblue" if sub_G.nodes[n]["node_type"] == "dev" else "lightgreen" for n in sub_G.nodes]
labels = {n: sub_G.nodes[n]["label"] for n in sub_G.nodes}

plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=600, alpha=0.9)
nx.draw_networkx_labels(sub_G, pos, labels=labels, font_size=8)

# エッジの rel_type に応じて色分け描画
legend_elements = []
for rel_type, color in color_map.items():
    rel_edges = [(u, v) for u, v, d in sub_G.edges(data=True) if d["rel_type"] == rel_type]
    nx.draw_networkx_edges(sub_G, pos, edgelist=rel_edges, edge_color=color, arrows=True, alpha=0.8, width=1.5)
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=rel_type))

# 凡例
plt.legend(handles=legend_elements, title="Edge Types", loc="lower left")

plt.title("GNN Graph with GitHub Labels and Colored Edges", fontsize=15)
plt.axis("off")
plt.tight_layout()

# 保存
output_path = Path("outputs/gnn_graph_labeled.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"✅ グラフ画像を保存しました → {output_path}")

plt.show()
# scripts/plot_gnn_with_labels.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# === グラフ読み込み ===
graph_path = Path("data/graph.pt")
data = torch.load(graph_path, weights_only=False)

# === ノード名取得 ===
dev_names = data["dev"].node_id
num_devs = data["dev"].num_nodes
num_tasks = data["task"].num_nodes

# === NetworkX グラフ構築 ===
G = nx.DiGraph()

# ノード追加（GitHub名付き）
for i in range(num_devs):
    G.add_node(f"dev_{i}", label=dev_names[i], node_type="dev")
for i in range(num_tasks):
    G.add_node(f"task_{i}", label=f"task_{i}", node_type="task")

# === エッジ追加（種類で色分けのため記録） ===
edge_colors = []
edge_types = []
color_map = {
    "writes": "cornflowerblue",
    "reviews": "mediumseagreen",
    "written_by": "lightgray",
    "reviewed_by": "orange"
}

for edge_type in data.edge_types:
    src_type, rel_type, dst_type = edge_type
    edge_index = data[edge_type].edge_index
    for i in range(edge_index.size(1)):
        src = f"{src_type}_{edge_index[0, i].item()}"
        dst = f"{dst_type}_{edge_index[1, i].item()}"
        G.add_edge(src, dst, rel_type=rel_type)
        edge_colors.append(color_map.get(rel_type, "gray"))
        edge_types.append(rel_type)

# === 部分グラフ（dev 30人 + task 30件） ===
sub_nodes = [
    n for n in G.nodes if
    (n.startswith("dev_") and int(n.split("_")[1]) < 30) or
    (n.startswith("task_") and int(n.split("_")[1]) < 30)
]
sub_G = G.subgraph(sub_nodes).copy()

# === 可視化 ===
pos = nx.spring_layout(sub_G, seed=42, k=0.8)
node_colors = ["skyblue" if sub_G.nodes[n]["node_type"] == "dev" else "lightgreen" for n in sub_G.nodes]
labels = {n: sub_G.nodes[n]["label"] for n in sub_G.nodes}

plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=600, alpha=0.9)
nx.draw_networkx_labels(sub_G, pos, labels=labels, font_size=8)

# エッジの rel_type に応じて色分け描画
legend_elements = []
for rel_type, color in color_map.items():
    rel_edges = [(u, v) for u, v, d in sub_G.edges(data=True) if d["rel_type"] == rel_type]
    nx.draw_networkx_edges(sub_G, pos, edgelist=rel_edges, edge_color=color, arrows=True, alpha=0.8, width=1.5)
    legend_elements.append(Line2D([0], [0], color=color, lw=2, label=rel_type))

# 凡例
plt.legend(handles=legend_elements, title="Edge Types", loc="lower left")

plt.title("GNN Graph with GitHub Labels and Colored Edges", fontsize=15)
plt.axis("off")
plt.tight_layout()

# 保存
output_path = Path("outputs/gnn_graph_labeled.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"✅ グラフ画像を保存しました → {output_path}")

plt.show()
