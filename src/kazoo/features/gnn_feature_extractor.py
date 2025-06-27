import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# GNNモデルをインポート
sys.path.append(str(Path(__file__).resolve().parents[2]))
from kazoo.gnn.gnn_model import GNNModel


class GNNFeatureExtractor:
    """IRLのためのGNNベースの特徴量抽出器"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if getattr(cfg.irl, "use_gnn", False):
            self._load_gnn_model()
        else:
            self.model = None
            self.embeddings = None

    def _load_gnn_model(self):
        """GNNモデルとグラフデータを読み込み"""
        try:
            # グラフデータ読み込み
            graph_path = Path(getattr(self.cfg.irl, "gnn_graph_path", "data/graph.pt"))
            if not graph_path.exists():
                print(f"Warning: GNN graph file not found: {graph_path}")
                self.model = None
                return

            self.graph_data = torch.load(graph_path, weights_only=False)

            # モデル読み込み
            model_path = Path(
                getattr(self.cfg.irl, "gnn_model_path", "data/gnn_model.pt")
            )
            if not model_path.exists():
                print(f"Warning: GNN model file not found: {model_path}")
                self.model = None
                return

            self.model = GNNModel(
                in_channels_dict={"dev": 8, "task": 9}, out_channels=32
            )
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()
            self.model.to(self.device)

            # 埋め込みを事前計算
            with torch.no_grad():
                self.embeddings = self.model(
                    self.graph_data.x_dict, self.graph_data.edge_index_dict
                )

            # ノードIDマッピングを作成
            self._create_id_mappings()

            print("✅ GNN model loaded successfully")

        except Exception as e:
            print(f"Error loading GNN model: {e}")
            self.model = None
            self.embeddings = None

    def _create_id_mappings(self):
        """ノードIDと埋め込みインデックスのマッピングを作成"""
        if hasattr(self.graph_data["dev"], "node_id"):
            if isinstance(self.graph_data["dev"].node_id, list):
                dev_ids = self.graph_data["dev"].node_id
            else:
                dev_ids = self.graph_data["dev"].node_id.tolist()
        else:
            dev_ids = list(range(self.embeddings["dev"].size(0)))

        if hasattr(self.graph_data["task"], "node_id"):
            if isinstance(self.graph_data["task"].node_id, list):
                task_ids = self.graph_data["task"].node_id
            else:
                task_ids = self.graph_data["task"].node_id.tolist()
        else:
            task_ids = list(range(self.embeddings["task"].size(0)))

        # ID → インデックスのマッピング
        self.dev_id_to_idx = {str(dev_id): idx for idx, dev_id in enumerate(dev_ids)}
        self.task_id_to_idx = {
            str(task_id): idx for idx, task_id in enumerate(task_ids)
        }

        print(
            f"ID mappings created: {len(self.dev_id_to_idx)} devs, {len(self.task_id_to_idx)} tasks"
        )

    def get_gnn_features(self, task, developer, env):
        """GNNベースの特徴量を取得"""
        if not self.model or not self.embeddings:
            return [0.0] * 35  # 32 + 3 = 35次元のゼロベクトル

        try:
            # 開発者IDとタスクIDを取得
            dev_id = str(developer.get("name", ""))
            task_id = str(task.id)

            # インデックスを取得
            dev_idx = self.dev_id_to_idx.get(dev_id)
            task_idx = self.task_id_to_idx.get(task_id)

            if dev_idx is None or task_idx is None:
                # GNNに存在しない場合は零ベクトル
                return [0.0] * 35  # 32 + 3 = 35次元

            # 埋め込みを取得
            dev_emb = self.embeddings["dev"][dev_idx]
            task_emb = self.embeddings["task"][task_idx]

            # 特徴量を計算
            features = []

            # 1. 類似度スコア
            similarity = F.cosine_similarity(
                dev_emb.unsqueeze(0), task_emb.unsqueeze(0)
            ).item()
            features.append(similarity)

            # 2. 開発者の専門性スコア
            all_task_sims = F.cosine_similarity(
                dev_emb.unsqueeze(0), self.embeddings["task"]
            )
            dev_expertise = torch.mean(
                torch.topk(all_task_sims, k=min(10, all_task_sims.size(0))).values
            ).item()
            features.append(dev_expertise)

            # 3. タスクの人気度スコア
            all_dev_sims = F.cosine_similarity(
                task_emb.unsqueeze(0), self.embeddings["dev"]
            )
            task_popularity = torch.mean(
                torch.topk(all_dev_sims, k=min(10, all_dev_sims.size(0))).values
            ).item()
            features.append(task_popularity)

            # 4. 開発者埋め込み（32次元）
            features.extend(dev_emb.tolist())

            return features

        except Exception as e:
            print(f"Error extracting GNN features: {e}")
            return [0.0] * 35

    def get_feature_names(self):
        """GNN特徴量の名前リストを返す"""
        if not self.model:
            return []

        names = ["gnn_similarity", "gnn_dev_expertise", "gnn_task_popularity"]

        # 開発者埋め込みの各次元
        for i in range(32):
            names.append(f"gnn_dev_emb_{i}")

        return names
