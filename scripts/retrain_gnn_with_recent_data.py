#!/usr/bin/env python3

"""
より最近のデータを使用してGNNを再訓練するスクリプト
- 2020-2022年のGitHubアーカイブデータを使用
- 現在のバックログと開発者プロファイルに基づいてグラフを生成
- GNNモデルを再訓練
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))


def load_recent_github_data():
    """最近のGitHubアーカイブデータを読み込み"""
    print("📚 最近のGitHubデータを読み込み中...")

    data_files = [
        "data/gharchive_docker_compose_events_2020-08.jsonl",
        "data/gharchive_docker_compose_events_2021-01.jsonl",
        "data/gharchive_docker_compose_events_2021-10.jsonl",
        "data/gharchive_docker_compose_events_2022-02.jsonl",
    ]

    all_events = []
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"  📄 読み込み: {file_path}")
            with open(file_path, "r") as f:
                count = 0
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        all_events.append(event)
                        count += 1
                    except json.JSONDecodeError:
                        continue
                print(f"    イベント数: {count}")
        else:
            print(f"  ⚠️  ファイルが見つかりません: {file_path}")

    print(f"📊 総イベント数: {len(all_events)}")
    return all_events


def extract_developer_task_interactions(events):
    """開発者とタスクのインタラクションを抽出"""
    print("🔍 開発者-タスクインタラクションを抽出中...")

    interactions = []
    task_info = {}
    dev_info = {}

    for event in events:
        # イベントタイプを確認
        event_type = event.get("type", "")

        if event_type in ["PullRequestEvent", "IssuesEvent", "PullRequestReviewEvent"]:
            # 開発者情報を取得
            actor = event.get("actor", {})
            dev_name = actor.get("login", "")

            # ボット除外
            if "bot" in dev_name.lower() or not dev_name:
                continue

            # タスク/プルリク情報を取得
            payload = event.get("payload", {})

            if event_type == "PullRequestEvent":
                pr = payload.get("pull_request", {})
                task_id = f"pr_{pr.get('id', '')}"
                task_title = pr.get("title", "")
                task_body = pr.get("body", "") or ""
                task_labels = [label.get("name", "") for label in pr.get("labels", [])]

            elif event_type == "IssuesEvent":
                issue = payload.get("issue", {})
                task_id = f"issue_{issue.get('id', '')}"
                task_title = issue.get("title", "")
                task_body = issue.get("body", "") or ""
                task_labels = [
                    label.get("name", "") for label in issue.get("labels", [])
                ]

            elif event_type == "PullRequestReviewEvent":
                pr = payload.get("pull_request", {})
                task_id = f"pr_{pr.get('id', '')}"
                task_title = pr.get("title", "")
                task_body = pr.get("body", "") or ""
                task_labels = [label.get("name", "") for label in pr.get("labels", [])]

            if not task_id or not dev_name:
                continue

            # インタラクション記録
            action = payload.get("action", "")
            created_at = event.get("created_at", "")

            interactions.append(
                {
                    "dev_name": dev_name,
                    "task_id": task_id,
                    "event_type": event_type,
                    "action": action,
                    "created_at": created_at,
                }
            )

            # タスク情報を蓄積
            if task_id not in task_info:
                task_info[task_id] = {
                    "title": task_title,
                    "body": task_body,
                    "labels": task_labels,
                    "first_seen": created_at,
                }

            # 開発者情報を蓄積
            if dev_name not in dev_info:
                dev_info[dev_name] = {
                    "interactions": 0,
                    "tasks": set(),
                    "labels": set(),
                }

            dev_info[dev_name]["interactions"] += 1
            dev_info[dev_name]["tasks"].add(task_id)
            dev_info[dev_name]["labels"].update(task_labels)

    print(f"📊 抽出結果:")
    print(f"  インタラクション数: {len(interactions)}")
    print(f"  ユニークタスク数: {len(task_info)}")
    print(f"  ユニーク開発者数: {len(dev_info)}")

    return interactions, task_info, dev_info


def create_modern_graph_data(interactions, task_info, dev_info):
    """最新データに基づいてグラフデータを作成"""
    print("🕸️  最新グラフデータを作成中...")

    # 現在の開発者プロファイルを読み込み
    with open("configs/dev_profiles.yaml", "r") as f:
        existing_profiles = yaml.safe_load(f)

    # フィルタリング: 十分なインタラクションがある開発者・タスクのみ
    min_interactions = 2
    active_devs = {
        name: info
        for name, info in dev_info.items()
        if info["interactions"] >= min_interactions
    }

    min_dev_interactions = 1
    active_tasks = {
        tid: info
        for tid, info in task_info.items()
        if sum(1 for i in interactions if i["task_id"] == tid) >= min_dev_interactions
    }

    print(f"  アクティブ開発者: {len(active_devs)}")
    print(f"  アクティブタスク: {len(active_tasks)}")

    # グラフノードとエッジを作成
    import torch
    from torch_geometric.data import HeteroData

    # 開発者ノード特徴量を作成
    dev_features = []
    dev_node_ids = []

    for dev_name in active_devs.keys():
        dev_info_item = active_devs[dev_name]

        # 既存プロファイルがあれば使用、なければデフォルト
        profile = existing_profiles.get(
            dev_name, {"skills": ["general"], "touched_files": [], "label_affinity": {}}
        )

        # 特徴量計算
        feature_vector = [
            len(profile.get("skills", [])),  # スキル数
            len(profile.get("touched_files", [])),  # 経験ファイル数
            dev_info_item["interactions"],  # インタラクション数
            len(dev_info_item["tasks"]),  # 参加タスク数
            len(dev_info_item["labels"]),  # 扱ったラベル数
            profile.get("label_affinity", {}).get("bug", 0.0),  # バグ親和性
            profile.get("label_affinity", {}).get("enhancement", 0.0),  # 機能強化親和性
            profile.get("label_affinity", {}).get(
                "documentation", 0.0
            ),  # ドキュメント親和性
        ]

        dev_features.append(feature_vector)
        dev_node_ids.append(dev_name)

    # タスクノード特徴量を作成
    task_features = []
    task_node_ids = []

    for task_id in active_tasks.keys():
        task_info_item = active_tasks[task_id]

        # 特徴量計算
        labels = task_info_item["labels"]
        feature_vector = [
            len(task_info_item["title"]),  # タイトル長
            len(task_info_item["body"]),  # 本文長
            len(labels),  # ラベル数
            1 if "bug" in labels else 0,  # バグフラグ
            1 if "enhancement" in labels else 0,  # 機能強化フラグ
            1 if "documentation" in labels else 0,  # ドキュメントフラグ
            sum(
                1 for i in interactions if i["task_id"] == task_id
            ),  # インタラクション数
            len(
                set(i["dev_name"] for i in interactions if i["task_id"] == task_id)
            ),  # 関与開発者数
            task_info_item["body"].count("```") // 2,  # コードブロック数
        ]

        task_features.append(feature_vector)
        task_node_ids.append(task_id)

    # エッジを作成 (開発者 → タスクのインタラクション)
    edge_indices = []

    for interaction in interactions:
        dev_name = interaction["dev_name"]
        task_id = interaction["task_id"]

        if dev_name in active_devs and task_id in active_tasks:
            dev_idx = dev_node_ids.index(dev_name)
            task_idx = task_node_ids.index(task_id)
            edge_indices.append([dev_idx, task_idx])

    # HeteroDataオブジェクトを作成
    data = HeteroData()

    # ノード特徴量とIDを設定
    data["dev"].x = torch.tensor(dev_features, dtype=torch.float)
    data["dev"].node_id = dev_node_ids

    data["task"].x = torch.tensor(task_features, dtype=torch.float)
    data["task"].node_id = task_node_ids

    # エッジを設定
    if edge_indices:
        edge_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        data["dev", "writes", "task"].edge_index = edge_tensor
        # 逆方向エッジも追加
        data["task", "written_by", "dev"].edge_index = edge_tensor.flip(0)
    else:
        # エッジがない場合は空のテンソル
        data["dev", "writes", "task"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["task", "written_by", "dev"].edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )

    print(f"✅ グラフデータ作成完了:")
    print(f"  開発者ノード: {data['dev'].x.shape}")
    print(f"  タスクノード: {data['task'].x.shape}")
    print(f"  エッジ数: {data['dev', 'writes', 'task'].edge_index.shape[1]}")

    return data


def retrain_gnn_model(graph_data):
    """新しいグラフデータでGNNモデルを再訓練"""
    print("🤖 GNNモデルを再訓練中...")

    # GNNモデルをインポート
    try:
        from kazoo.gnn.gnn_model import GNNModel
    except ImportError:
        print("❌ GNNモデルをインポートできません")
        return None

    import torch
    import torch.nn.functional as F
    from torch.optim import Adam

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデル初期化
    model = GNNModel(
        in_channels_dict={
            "dev": graph_data["dev"].x.shape[1],  # 実際の開発者特徴量次元
            "task": graph_data["task"].x.shape[1],  # 実際のタスク特徴量次元
        },
        out_channels=32,
    )
    model = model.to(device)
    graph_data = graph_data.to(device)

    # オプティマイザー設定
    optimizer = Adam(model.parameters(), lr=0.01)

    # 自己教師あり学習でグラフ再構築を学習
    print("学習開始...")

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()

        # フォワードパス
        embeddings = model(graph_data.x_dict, graph_data.edge_index_dict)

        # 損失計算（コントラスト学習風）
        dev_emb = embeddings["dev"]
        task_emb = embeddings["task"]

        # 正のペア（実際のエッジ）
        edge_index = graph_data["dev", "writes", "task"].edge_index
        if edge_index.shape[1] > 0:
            pos_dev_emb = dev_emb[edge_index[0]]
            pos_task_emb = task_emb[edge_index[1]]
            pos_score = F.cosine_similarity(pos_dev_emb, pos_task_emb, dim=1)

            # 負のペア（ランダムサンプリング）
            num_neg = min(edge_index.shape[1], 100)
            neg_dev_idx = torch.randint(0, dev_emb.shape[0], (num_neg,)).to(device)
            neg_task_idx = torch.randint(0, task_emb.shape[0], (num_neg,)).to(device)
            neg_dev_emb = dev_emb[neg_dev_idx]
            neg_task_emb = task_emb[neg_task_idx]
            neg_score = F.cosine_similarity(neg_dev_emb, neg_task_emb, dim=1)

            # コントラスト損失
            pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()
            neg_loss = -torch.log(torch.sigmoid(-neg_score)).mean()
            loss = pos_loss + neg_loss
        else:
            # エッジがない場合は埋め込みの正則化のみ
            loss = torch.norm(dev_emb) + torch.norm(task_emb)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  エポック {epoch}: 損失 = {loss.item():.4f}")

    print("✅ 学習完了")

    # モデルとグラフデータを保存
    model.eval()
    model_save_path = "data/gnn_model_retrained.pt"
    graph_save_path = "data/graph_retrained.pt"

    torch.save(model.state_dict(), model_save_path)
    torch.save(graph_data.cpu(), graph_save_path)

    print(f"💾 再訓練済みモデル保存: {model_save_path}")
    print(f"💾 新しいグラフデータ保存: {graph_save_path}")

    return model


def main():
    """メイン実行関数"""
    print("🚀 GNN再訓練プロセス開始")
    print("=" * 50)

    # 1. 最近のデータを読み込み
    events = load_recent_github_data()

    if not events:
        print("❌ データが見つかりません")
        return

    # 2. インタラクションを抽出
    interactions, task_info, dev_info = extract_developer_task_interactions(events)

    # 3. グラフデータを作成
    graph_data = create_modern_graph_data(interactions, task_info, dev_info)

    # 4. GNNを再訓練
    model = retrain_gnn_model(graph_data)

    if model:
        print("\n🎉 GNN再訓練が正常に完了しました！")
        print("新しいモデルファイル:")
        print("  - data/gnn_model_retrained.pt")
        print("  - data/graph_retrained.pt")
        print("\n設定ファイルを更新して新しいモデルを使用してください。")
    else:
        print("❌ GNN再訓練に失敗しました")


if __name__ == "__main__":
    main()
