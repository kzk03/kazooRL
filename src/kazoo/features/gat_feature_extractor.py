#!/usr/bin/env python3
"""
GAT特徴量抽出器
Graph Attention Networkを使用した開発者・タスク特徴量抽出
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# GATモデルをインポート
sys.path.append(str(Path(__file__).resolve().parents[2]))


class GATFeatureExtractor:
    """GAT特徴量抽出器"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 統計情報
        self.stats = {
            "total_calls": 0,
            "successful_extractions": 0,
            "fallback_used": 0,
            "missing_developers": 0,
            "missing_tasks": 0,
        }

        # GAT設定の確認
        if getattr(cfg.irl, "use_gat", False):
            self._load_collaboration_network()
        else:
            self.model = None
            self.dev_network = None

    def _load_collaboration_network(self):
        """開発者協力ネットワークを読み込み"""
        try:
            # 開発者協力ネットワークを読み込み
            network_path = Path("data/developer_collaboration_network.pt")
            if network_path.exists():
                print("Loading developer collaboration network...")
                self.dev_network = torch.load(network_path, weights_only=False)
                print(
                    f"✅ Developer network loaded: {self.dev_network['num_developers']} devs, {self.dev_network['dev_collaboration_edge_index'].shape[1]} edges"
                )

                # ID mappings
                self.dev_to_id = self.dev_network["dev_to_id"]
                self.id_to_dev = self.dev_network["id_to_dev"]

                # 簡単なGAT特徴量を事前計算（協力ネットワークベース）
                self._precompute_embeddings()

                print("✅ GAT feature extractor initialized")
            else:
                print(f"Warning: Developer network file not found: {network_path}")
                self.model = None
                self.dev_network = None

        except Exception as e:
            print(f"Error loading GAT model: {e}")
            self.model = None
            self.dev_network = None

    def _precompute_embeddings(self):
        """協力ネットワークから簡単な埋め込みを事前計算"""
        try:
            # 開発者特徴量を取得
            dev_features = self.dev_network[
                "dev_features_enhanced"
            ]  # [num_devs, feature_dim]
            edge_index = self.dev_network[
                "dev_collaboration_edge_index"
            ]  # [2, num_edges]

            # 簡単なGraph Convolutionで埋め込み計算
            num_devs = dev_features.shape[0]

            # 隣接行列を作成
            adj_matrix = torch.zeros(num_devs, num_devs)
            adj_matrix[edge_index[0], edge_index[1]] = 1.0

            # 自己ループを追加
            adj_matrix += torch.eye(num_devs)

            # 正規化
            degree = adj_matrix.sum(dim=1, keepdim=True)
            degree[degree == 0] = 1  # ゼロ除算を防ぐ
            adj_matrix = adj_matrix / degree

            # 簡単な線形変換で32次元埋め込みを生成
            embedding_dim = 32
            linear_transform = torch.randn(dev_features.shape[1], embedding_dim) * 0.1

            # 埋め込み計算: (隣接行列 × 特徴量) × 線形変換
            self.dev_embeddings = torch.matmul(
                torch.matmul(adj_matrix, dev_features), linear_transform
            )

            print(f"✅ Precomputed embeddings: {self.dev_embeddings.shape}")

        except Exception as e:
            print(f"Warning: Failed to precompute embeddings: {e}")
            # フォールバック: ランダム埋め込み
            num_devs = len(self.dev_to_id)
            self.dev_embeddings = torch.randn(num_devs, 32) * 0.1

    def get_gat_features(self, task, developer, env):
        """GAT特徴量を取得"""
        self.stats["total_calls"] += 1

        if self.dev_network is None:
            # GAT無効時はゼロ埋め込みを返す
            return [0.0] * 32

        try:
            developer_name = developer.get("name", "")
            if not developer_name:
                self.stats["missing_developers"] += 1
                return self._get_fallback_features()

            # 開発者IDを取得
            if developer_name not in self.dev_to_id:
                self.stats["missing_developers"] += 1
                return self._get_fallback_features()

            dev_id = self.dev_to_id[developer_name]

            # 事前計算された埋め込みを取得
            dev_embedding = self.dev_embeddings[dev_id].numpy()

            # GAT特徴量を構築
            gat_features = []

            # 1. 開発者埋め込み (32次元)
            gat_features.extend(dev_embedding.tolist())

            self.stats["successful_extractions"] += 1
            return gat_features

        except Exception as e:
            print(f"Warning: Error in GAT feature extraction: {e}")
            self.stats["fallback_used"] += 1
            return self._get_fallback_features()

    def _get_fallback_features(self):
        """フォールバック特徴量（32次元のゼロベクトル）"""
        return [0.0] * 32

    def get_feature_names(self):
        """GAT特徴量の名前を返す"""
        names = []

        # 開発者埋め込み特徴量名
        for i in range(32):
            names.append(f"gat_dev_emb_{i}")

        return names

    def print_statistics(self):
        """統計情報を表示"""
        if self.stats["total_calls"] > 0:
            success_rate = (
                self.stats["successful_extractions"] / self.stats["total_calls"]
            )
            print(f"GAT Feature Extraction Statistics:")
            print(f"  Total calls: {self.stats['total_calls']}")
            print(
                f"  Successful extractions: {self.stats['successful_extractions']} ({success_rate:.1%})"
            )
            print(f"  Fallback used: {self.stats['fallback_used']}")
            print(f"  Missing developers: {self.stats['missing_developers']}")
            print(f"  Missing tasks: {self.stats['missing_tasks']}")
        else:
            print("No GAT feature extraction calls made yet.")
