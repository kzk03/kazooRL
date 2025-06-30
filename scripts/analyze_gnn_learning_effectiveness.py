#!/usr/bin/env python3
"""
GNN学習効果の詳細分析
なぜ類似度が変化しないのかを調査
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv


def analyze_gnn_learning_effectiveness():
    """GNN学習効果の詳細分析"""
    print("🔬 GNN学習効果の詳細分析")
    print("=" * 50)

    # 環境初期化
    cfg = OmegaConf.load(project_root / "configs" / "base.yaml")

    with open(project_root / cfg.env.backlog_path, "r") as f:
        backlog = json.load(f)

    with open(project_root / cfg.env.dev_profiles_path, "r") as f:
        dev_profiles = yaml.safe_load(f)

    env = OSSSimpleEnv(cfg, backlog, dev_profiles)
    gnn_extractor = env.feature_extractor.gnn_extractor

    print("✅ 分析準備完了")

    # モデルパラメータの初期状態を保存
    initial_params = {}
    for name, param in gnn_extractor.model.named_parameters():
        initial_params[name] = param.clone().detach()

    print(
        f"📊 モデルパラメータ数: {sum(p.numel() for p in gnn_extractor.model.parameters())}"
    )

    # 学習率を確認
    print(f"📊 学習率: {gnn_extractor.learning_rate}")
    print(f"📊 更新頻度: {gnn_extractor.update_frequency}")

    # テスト用のインタラクション
    class TestTask:
        def __init__(self, task_id):
            self.id = task_id

    class TestDev:
        def __init__(self, dev_name):
            self.name = dev_name

        def get(self, key, default=None):
            return self.name if key == "name" else default

    test_devs = list(gnn_extractor.dev_id_to_idx.keys())[:2]
    test_tasks = list(gnn_extractor.task_id_to_idx.keys())[:2]

    print(f"🎯 テスト対象: {test_devs[0]} & {test_tasks[0]}")

    # 初期埋め込みを記録
    def get_specific_embeddings():
        with torch.no_grad():
            embeddings = gnn_extractor.model(
                gnn_extractor.graph_data.x_dict,
                gnn_extractor.graph_data.edge_index_dict,
            )

        dev_idx = gnn_extractor.dev_id_to_idx[test_devs[0]]
        task_idx = gnn_extractor.task_id_to_idx[test_tasks[0]]

        return {
            "dev_emb": embeddings["dev"][dev_idx].clone(),
            "task_emb": embeddings["task"][task_idx].clone(),
            "similarity": F.cosine_similarity(
                embeddings["dev"][dev_idx].unsqueeze(0),
                embeddings["task"][task_idx].unsqueeze(0),
            ).item(),
        }

    initial_state = get_specific_embeddings()
    print(f"📊 初期類似度: {initial_state['similarity']:.6f}")
    print(
        f"📊 初期開発者埋め込みノルム: {torch.norm(initial_state['dev_emb']).item():.6f}"
    )
    print(
        f"📊 初期タスク埋め込みノルム: {torch.norm(initial_state['task_emb']).item():.6f}"
    )

    # 強いインタラクションを複数回実行
    base_time = env.current_time

    print(f"\n🔥 強いポジティブインタラクションを実行:")
    for i in range(20):  # 20回の強いインタラクション
        developer = TestDev(test_devs[0])
        task = TestTask(test_tasks[0])
        reward = 5.0  # 非常に強いポジティブ報酬
        sim_time = base_time + timedelta(hours=i)

        gnn_extractor.record_interaction(
            task, developer, reward, "strong_positive", simulation_time=sim_time
        )

        # 5回ごとに状態をチェック
        if (i + 1) % 5 == 0:
            current_state = get_specific_embeddings()
            similarity_change = (
                current_state["similarity"] - initial_state["similarity"]
            )
            print(
                f"  インタラクション {i+1}: 類似度 = {current_state['similarity']:.6f} (Δ{similarity_change:+.6f})"
            )

    # 最終状態
    final_state = get_specific_embeddings()
    final_similarity_change = final_state["similarity"] - initial_state["similarity"]

    print(f"\n📊 最終結果:")
    print(f"  初期類似度: {initial_state['similarity']:.6f}")
    print(f"  最終類似度: {final_state['similarity']:.6f}")
    print(f"  変化量: {final_similarity_change:+.6f}")

    # パラメータの変化を確認
    param_changes = []
    for name, initial_param in initial_params.items():
        current_param = dict(gnn_extractor.model.named_parameters())[name]
        change = torch.norm(current_param - initial_param).item()
        param_changes.append((name, change))

    # 最も変化したパラメータを表示
    param_changes.sort(key=lambda x: x[1], reverse=True)
    print(f"\n📊 パラメータ変化（上位5層）:")
    for name, change in param_changes[:5]:
        print(f"  {name}: {change:.8f}")

    # 損失値の詳細分析
    print(f"\n🔍 損失計算の詳細:")

    # 手動で損失を計算
    current_embeddings = gnn_extractor.model(
        gnn_extractor.graph_data.x_dict, gnn_extractor.graph_data.edge_index_dict
    )

    dev_idx = gnn_extractor.dev_id_to_idx[test_devs[0]]
    task_idx = gnn_extractor.task_id_to_idx[test_tasks[0]]

    dev_emb = current_embeddings["dev"][dev_idx]
    task_emb = current_embeddings["task"][task_idx]

    similarity = F.cosine_similarity(dev_emb.unsqueeze(0), task_emb.unsqueeze(0))
    sigmoid_sim = torch.sigmoid(similarity)

    print(f"  コサイン類似度: {similarity.item():.6f}")
    print(f"  シグモイド後: {sigmoid_sim.item():.6f}")
    print(f"  ログ値: {torch.log(sigmoid_sim + 1e-8).item():.6f}")

    # 学習設定の確認
    print(f"\n⚙️ 学習設定:")
    print(f"  更新回数: {gnn_extractor.stats.get('updates', 0)}")
    print(f"  学習率: {gnn_extractor.learning_rate}")
    print(f"  オプティマイザー: {type(gnn_extractor.optimizer).__name__}")

    # 埋め込み分布の分析
    dev_embeddings = current_embeddings["dev"]
    task_embeddings = current_embeddings["task"]

    print(f"\n📊 埋め込み統計:")
    print(
        f"  開発者埋め込み - 平均: {dev_embeddings.mean().item():.6f}, 標準偏差: {dev_embeddings.std().item():.6f}"
    )
    print(
        f"  タスク埋め込み - 平均: {task_embeddings.mean().item():.6f}, 標準偏差: {task_embeddings.std().item():.6f}"
    )

    # 類似度分布
    all_similarities = []
    for i in range(min(10, dev_embeddings.shape[0])):
        for j in range(min(10, task_embeddings.shape[0])):
            sim = F.cosine_similarity(
                dev_embeddings[i].unsqueeze(0), task_embeddings[j].unsqueeze(0)
            ).item()
            all_similarities.append(sim)

    import numpy as np

    all_similarities = np.array(all_similarities)
    print(
        f"  類似度分布 - 平均: {all_similarities.mean():.6f}, 標準偏差: {all_similarities.std():.6f}"
    )
    print(f"  類似度範囲: [{all_similarities.min():.6f}, {all_similarities.max():.6f}]")

    # 勾配の確認
    print(f"\n🔍 勾配確認:")
    gnn_extractor.model.train()
    gnn_extractor.optimizer.zero_grad()

    # 損失を手動計算
    weight = 5.0
    pos_loss = -weight * torch.log(torch.sigmoid(similarity) + 1e-8)
    print(f"  計算された損失: {pos_loss.item():.6f}")

    pos_loss.backward()

    # 勾配のノルムを確認
    total_grad_norm = 0
    for name, param in gnn_extractor.model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm += grad_norm**2
            if grad_norm > 1e-6:
                print(f"  {name}: 勾配ノルム = {grad_norm:.8f}")

    total_grad_norm = total_grad_norm**0.5
    print(f"  総勾配ノルム: {total_grad_norm:.8f}")


if __name__ == "__main__":
    analyze_gnn_learning_effectiveness()
