import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# GNNモデルをインポート
sys.path.append(str(Path(__file__).resolve().parents[2]))
from kazoo.GAT.GAT_model import GNNModel


class GNNFeatureExtractor:
    """IRLのためのGNNベースの特徴量抽出器"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 統計カウンタ
        self.stats = {
            "total_requests": 0,
            "full_features": 0,
            "missing_dev": 0,
            "missing_task": 0,
            "missing_both": 0,
            "errors": 0,
            "updates": 0,  # GNN更新回数を追加
        }

        # オンライン学習用の設定
        self.online_learning = cfg.irl.get("online_gnn_learning", False)
        self.update_frequency = cfg.irl.get("gnn_update_frequency", 100)  # N回に1回更新
        self.learning_rate = cfg.irl.get("gnn_learning_rate", 0.001)
        self.time_window_hours = cfg.irl.get(
            "gnn_time_window_hours", 24
        )  # 学習に使用する時間窓（時間）
        self.optimizer = None

        # 新しいインタラクションデータを蓄積
        self.interaction_buffer = []
        self.max_buffer_size = cfg.irl.get("gnn_buffer_size", 1000)

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

            # 🆕 開発者協力ネットワークも読み込み
            network_path = Path("data/developer_collaboration_network.pt")
            if network_path.exists():
                print("Loading developer collaboration network...")
                self.dev_network = torch.load(network_path, weights_only=False)
                print(
                    f"✅ Developer network loaded: {self.dev_network['num_developers']} devs, {self.dev_network['network_stats']['num_edges']} edges"
                )
            else:
                print(
                    "Warning: Developer collaboration network not found, using basic features only"
                )
                self.dev_network = None

            # モデル読み込み
            model_path = Path(
                getattr(self.cfg.irl, "gnn_model_path", "data/gnn_model.pt")
            )
            if not model_path.exists():
                print(f"Warning: GNN model file not found: {model_path}")
                self.model = None
                return

            # Import GNN model here to avoid circular imports
            from kazoo.GAT.GAT_model import GNNModel

            self.model = GNNModel(
                in_channels_dict={"dev": 8, "task": 9}, out_channels=32
            )
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()
            self.model.to(self.device)

            # 埋め込みを事前計算（協力ネットワークを含む）
            edge_index_dict = {
                ("dev", "writes", "task"): self.graph_data[
                    ("dev", "writes", "task")
                ].edge_index,
                ("task", "written_by", "dev"): self.graph_data[
                    ("dev", "writes", "task")
                ].edge_index.flip([0]),
            }

            # 🆕 開発者協力関係エッジを追加
            if self.dev_network is not None:
                edge_index_dict[("dev", "collaborates", "dev")] = self.dev_network[
                    "dev_collaboration_edge_index"
                ]

            with torch.no_grad():
                self.embeddings = self.model(self.graph_data.x_dict, edge_index_dict)

            # ノードIDマッピングを作成
            self._create_id_mappings()

            # オンライン学習用のオプティマイザーを初期化
            if self.online_learning:
                from torch.optim import Adam

                self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
                print(
                    f"✅ Online GNN learning enabled (update every {self.update_frequency} steps)"
                )

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
            dev_ids = list(range(self.embeddings["dev"].shape[0]))

        if hasattr(self.graph_data["task"], "node_id"):
            if isinstance(self.graph_data["task"].node_id, list):
                task_ids = self.graph_data["task"].node_id
            else:
                task_ids = self.graph_data["task"].node_id.tolist()
        else:
            task_ids = list(range(self.embeddings["task"].shape[0]))

        # ID → インデックスのマッピング
        self.dev_id_to_idx = {str(dev_id): idx for idx, dev_id in enumerate(dev_ids)}
        self.task_id_to_idx = {
            str(task_id): idx for idx, task_id in enumerate(task_ids)
        }

        print(
            f"ID mappings created: {len(self.dev_id_to_idx)} devs, {len(self.task_id_to_idx)} tasks"
        )

    def get_gnn_features(self, task, developer, env):
        """GNNベースの特徴量を取得（3次元のスコアのみ）"""
        self.stats["total_requests"] += 1

        if not self.model or not self.embeddings:
            # 協力ネットワークが利用可能かどうかで特徴量数を決定
            num_features = 5 if self.dev_network is not None else 3
            return [0.0] * num_features

        try:
            # 開発者IDとタスクIDを取得
            dev_id = str(developer.get("name", ""))
            task_id = str(task.id)

            # インデックスを取得
            dev_idx = self.dev_id_to_idx.get(dev_id)
            task_idx = self.task_id_to_idx.get(task_id)

            # Missing node handling with more informative approach
            missing_dev = dev_idx is None
            missing_task = task_idx is None

            # 協力ネットワークが利用可能かどうかで特徴量数を決定
            num_features = 5 if self.dev_network is not None else 3

            if missing_dev and missing_task:
                # Both missing - return zero features
                self.stats["missing_both"] += 1
                return [0.0] * num_features
            elif missing_dev:
                # Developer missing - use average developer embedding as fallback
                self.stats["missing_dev"] += 1
                return self._get_fallback_features_missing_dev(task_idx, dev_id)
            elif missing_task:
                # Task missing - use average task embedding as fallback
                self.stats["missing_task"] += 1
                return self._get_fallback_features_missing_task(dev_idx, task_id)
            else:
                # Both nodes exist - compute full features
                self.stats["full_features"] += 1
                return self._get_simplified_gnn_features(dev_idx, task_idx)

        except Exception as e:
            self.stats["errors"] += 1
            print(
                f"Error extracting GNN features for dev={dev_id}, task={task_id}: {e}"
            )
            num_features = 5 if self.dev_network is not None else 3
            return [0.0] * num_features

    def record_interaction(
        self, task, developer, reward, action_taken=None, simulation_time=None
    ):
        """新しいインタラクションを記録（強化学習のステップごとに呼び出される）"""
        if not self.online_learning:
            return

        # 実際のシミュレーション時間を記録
        from datetime import datetime

        if simulation_time is None:
            timestamp = datetime.now()
        else:
            timestamp = simulation_time

        interaction = {
            "dev_id": str(developer.get("name", "")),
            "task_id": str(task.id),
            "reward": float(reward),
            "timestamp": timestamp,
            "simulation_time": timestamp,
            "step_number": len(self.interaction_buffer),
            "action_taken": action_taken,  # アサインされたかどうか等
        }

        self.interaction_buffer.append(interaction)

        # バッファサイズを制限
        if len(self.interaction_buffer) > self.max_buffer_size:
            self.interaction_buffer.pop(0)  # 古いものから削除

        # 定期的にGNNを更新
        if len(self.interaction_buffer) % self.update_frequency == 0:
            self._update_gnn_online()

    def _update_gnn_online(self):
        """蓄積されたインタラクションデータでGNNをオンライン更新（時系列考慮）"""
        if (
            not self.online_learning
            or not self.optimizer
            or len(self.interaction_buffer) < 5
        ):
            return

        # 最新の時間を取得
        if not self.interaction_buffer:
            return

        latest_time = max(
            interaction["simulation_time"] for interaction in self.interaction_buffer
        )
        print(f"🔄 GNNをオンライン更新中... (最新時刻: {latest_time})")

        # 時間窓内のインタラクションのみを使用
        from datetime import timedelta

        time_window = timedelta(hours=self.time_window_hours)
        cutoff_time = latest_time - time_window

        recent_interactions = [
            interaction
            for interaction in self.interaction_buffer
            if interaction["simulation_time"] >= cutoff_time
        ]

        print(
            f"  時間窓内のインタラクション: {len(recent_interactions)}/{len(self.interaction_buffer)}"
        )
        print(f"  時間範囲: {cutoff_time} ～ {latest_time}")

        if len(recent_interactions) < 3:
            print("  時間窓内のインタラクション数が不足、スキップ")
            return

        try:
            self.model.train()
            self.optimizer.zero_grad()

            # 時間窓内のインタラクションから正負のペアを生成
            positive_pairs = []
            negative_pairs = []

            # 報酬に基づいてポジティブ/ネガティブペアを作成（時間窓内のみ）
            for interaction in recent_interactions:
                dev_id = interaction["dev_id"]
                task_id = interaction["task_id"]
                reward = interaction["reward"]

                dev_idx = self.dev_id_to_idx.get(dev_id)
                task_idx = self.task_id_to_idx.get(task_id)

                # デバッグ情報
                print(
                    f"  チェック: {dev_id} (idx={dev_idx}) + {task_id} (idx={task_idx}) = {reward}"
                )

                if dev_idx is not None and task_idx is not None:
                    if reward > 0:  # 良いインタラクション
                        positive_pairs.append((dev_idx, task_idx, reward))
                        print(f"    → ポジティブペア追加")
                    elif reward < 0:  # 悪いインタラクション
                        negative_pairs.append((dev_idx, task_idx, abs(reward)))
                        print(f"    → ネガティブペア追加")
                else:
                    print(f"    → スキップ（ノードなし）")

            print(
                f"  有効なペア: ポジティブ={len(positive_pairs)}, ネガティブ={len(negative_pairs)}"
            )

            if len(positive_pairs) == 0 and len(negative_pairs) == 0:
                print("  有効なペアが見つかりませんでした")
                return

            # 現在の埋め込みを取得
            current_embeddings = self.model(
                self.graph_data.x_dict, self.graph_data.edge_index_dict
            )

            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_terms = []

            # ポジティブペアの損失（類似度を高める）
            if positive_pairs:
                for dev_idx, task_idx, weight in positive_pairs:
                    dev_emb = current_embeddings["dev"][dev_idx]
                    task_emb = current_embeddings["task"][task_idx]

                    similarity = F.cosine_similarity(
                        dev_emb.unsqueeze(0), task_emb.unsqueeze(0)
                    )
                    # 類似度を高めたいので、負の損失（最大化）
                    pos_loss = -weight * torch.log(torch.sigmoid(similarity) + 1e-8)
                    loss_terms.append(pos_loss)
                    print(f"    ポジティブ損失: {pos_loss.item():.4f}")

            # ネガティブペアの損失（類似度を下げる）
            if negative_pairs:
                for dev_idx, task_idx, weight in negative_pairs:
                    dev_emb = current_embeddings["dev"][dev_idx]
                    task_emb = current_embeddings["task"][task_idx]

                    similarity = F.cosine_similarity(
                        dev_emb.unsqueeze(0), task_emb.unsqueeze(0)
                    )
                    # 類似度を下げたいので、正の損失（最小化）
                    neg_loss = weight * torch.log(torch.sigmoid(similarity) + 1e-8)
                    loss_terms.append(neg_loss)
                    print(f"    ネガティブ損失: {neg_loss.item():.4f}")

            # 全ての損失を合計
            if loss_terms:
                total_loss = torch.stack(loss_terms).sum()

            # 正則化項（埋め込みが大きくなりすぎないように）
            reg_loss = 0.001 * (
                torch.norm(current_embeddings["dev"])
                + torch.norm(current_embeddings["task"])
            )
            total_loss = total_loss + reg_loss

            print(f"  総損失: {total_loss.item():.4f}")

            # バックプロパゲーション
            if total_loss.requires_grad:
                total_loss.backward()
                self.optimizer.step()

                # 埋め込みを再計算
                self.model.eval()
                with torch.no_grad():
                    self.embeddings = self.model(
                        self.graph_data.x_dict, self.graph_data.edge_index_dict
                    )

                self.stats["updates"] += 1
                print(
                    f"  ✅ GNN更新完了 (損失: {total_loss.item():.4f}, 更新回数: {self.stats['updates']})"
                )
            else:
                print("  損失に勾配がありません")

        except Exception as e:
            print(f"  ❌ GNN更新中にエラー: {e}")
            import traceback

            traceback.print_exc()

        finally:
            self.model.eval()

    def add_new_nodes(self, new_developers=None, new_tasks=None):
        """新しい開発者やタスクノードをGNNに追加"""
        if not self.online_learning:
            return

        print("🆕 新しいノードをGNNに追加中...")

        try:
            # 新しい開発者ノードを追加
            if new_developers:
                for dev_name, dev_profile in new_developers.items():
                    if dev_name not in self.dev_id_to_idx:
                        # 新しい開発者の特徴量を計算
                        dev_features = self._compute_dev_features(dev_name, dev_profile)

                        # グラフデータに追加
                        new_dev_tensor = torch.tensor(
                            [dev_features], dtype=torch.float, device=self.device
                        )
                        self.graph_data["dev"].x = torch.cat(
                            [self.graph_data["dev"].x, new_dev_tensor], dim=0
                        )

                        # ID マッピングを更新
                        new_idx = len(self.dev_id_to_idx)
                        self.dev_id_to_idx[dev_name] = new_idx
                        self.graph_data["dev"].node_id.append(dev_name)

                        print(f"  ✅ 新しい開発者追加: {dev_name} (index: {new_idx})")

            # 新しいタスクノードを追加
            if new_tasks:
                for task_id, task_info in new_tasks.items():
                    if task_id not in self.task_id_to_idx:
                        # 新しいタスクの特徴量を計算
                        task_features = self._compute_task_features(task_id, task_info)

                        # グラフデータに追加
                        new_task_tensor = torch.tensor(
                            [task_features], dtype=torch.float, device=self.device
                        )
                        self.graph_data["task"].x = torch.cat(
                            [self.graph_data["task"].x, new_task_tensor], dim=0
                        )

                        # ID マッピングを更新
                        new_idx = len(self.task_id_to_idx)
                        self.task_id_to_idx[task_id] = new_idx
                        self.graph_data["task"].node_id.append(task_id)

                        print(f"  ✅ 新しいタスク追加: {task_id} (index: {new_idx})")

            # 埋め込みを再計算
            if new_developers or new_tasks:
                with torch.no_grad():
                    self.embeddings = self.model(
                        self.graph_data.x_dict, self.graph_data.edge_index_dict
                    )
                print(f"  ✅ 埋め込み再計算完了")

        except Exception as e:
            print(f"  ❌ ノード追加中にエラー: {e}")

    def _compute_dev_features(self, dev_name, dev_profile):
        """新しい開発者の特徴量を計算"""
        return [
            len(dev_profile.get("skills", [])),
            len(dev_profile.get("touched_files", [])),
            0,  # 初期インタラクション数
            0,  # 初期参加タスク数
            0,  # 初期扱ったラベル数
            dev_profile.get("label_affinity", {}).get("bug", 0.0),
            dev_profile.get("label_affinity", {}).get("enhancement", 0.0),
            dev_profile.get("label_affinity", {}).get("documentation", 0.0),
        ]

    def _compute_task_features(self, task_id, task_info):
        """新しいタスクの特徴量を計算"""
        labels = task_info.get("labels", [])
        return [
            len(task_info.get("title", "")),
            len(task_info.get("body", "")),
            len(labels),
            1 if "bug" in labels else 0,
            1 if "enhancement" in labels else 0,
            1 if "documentation" in labels else 0,
            0,  # 初期インタラクション数
            0,  # 初期関与開発者数
            task_info.get("body", "").count("```") // 2,
        ]

    def _get_simplified_gnn_features(self, dev_idx, task_idx):
        """簡略化されたGNN特徴量（基本3次元 + 協力ネットワーク特徴量）"""
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

        # 🆕 4. 協力ネットワーク特徴量（利用可能な場合）
        if self.dev_network is not None:
            # 協力強度スコア（開発者の協力エッジの重み合計）
            collab_strength = self._calculate_collaboration_strength(dev_idx)
            features.append(collab_strength)

            # ネットワーク中心性スコア（開発者のネットワーク内での重要度）
            centrality = self._calculate_network_centrality(dev_idx)
            features.append(centrality)

        return features

    def _calculate_collaboration_strength(self, dev_idx):
        """開発者の協力ネットワーク内での強度を計算"""
        try:
            if self.dev_network is None:
                return 0.0

            edge_index = self.dev_network["dev_collaboration_edge_index"]
            edge_weights = self.dev_network["dev_collaboration_edge_weights"]

            # この開発者が関与するエッジを見つける
            dev_edges = (edge_index[0] == dev_idx) | (edge_index[1] == dev_idx)
            if not dev_edges.any():
                return 0.0

            # 関連するエッジの重みの合計を計算
            strength = edge_weights[dev_edges].sum().item()
            # 正規化（最大強度で割る）
            max_strength = (
                edge_weights.max().item() if edge_weights.numel() > 0 else 1.0
            )
            return strength / max_strength if max_strength > 0 else 0.0

        except Exception as e:
            print(f"Error calculating collaboration strength: {e}")
            return 0.0

    def _calculate_network_centrality(self, dev_idx):
        """開発者のネットワーク中心性を計算（簡易版）"""
        try:
            if self.dev_network is None:
                return 0.0

            edge_index = self.dev_network["dev_collaboration_edge_index"]

            # この開発者の次数（接続数）を計算
            degree = (
                ((edge_index[0] == dev_idx) | (edge_index[1] == dev_idx)).sum().item()
            )

            # 最大次数で正規化
            total_devs = self.dev_network["num_developers"]
            max_possible_degree = total_devs - 1
            return degree / max_possible_degree if max_possible_degree > 0 else 0.0

        except Exception as e:
            print(f"Error calculating network centrality: {e}")
            return 0.0

    def _get_full_gnn_features(self, dev_idx, task_idx):
        """両方のノードが存在する場合の完全なGNN特徴量を計算"""
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

    def _get_fallback_features_missing_dev(self, task_idx, dev_id):
        """開発者ノードが存在しない場合のフォールバック特徴量"""
        # タスクは存在するので、平均的な開発者との比較を使用
        task_emb = self.embeddings["task"][task_idx]
        avg_dev_emb = torch.mean(self.embeddings["dev"], dim=0)

        features = []

        # 1. 平均開発者との類似度（低めの値になると予想）
        similarity = (
            F.cosine_similarity(avg_dev_emb.unsqueeze(0), task_emb.unsqueeze(0)).item()
            * 0.5
        )  # ペナルティとして半分にする
        features.append(similarity)

        # 2. 平均的な専門性スコア（中程度の値）
        features.append(0.3)

        # 3. タスクの人気度は計算可能
        all_dev_sims = F.cosine_similarity(
            task_emb.unsqueeze(0), self.embeddings["dev"]
        )
        task_popularity = torch.mean(
            torch.topk(all_dev_sims, k=min(10, all_dev_sims.size(0))).values
        ).item()
        features.append(task_popularity)

        # 🆕 4-5. 協力ネットワーク特徴量（利用可能な場合）
        if self.dev_network is not None:
            # 開発者が存在しないので、協力特徴量は0
            features.append(0.0)  # collaboration_strength
            features.append(0.0)  # network_centrality

        return features

    def _get_fallback_features_missing_task(self, dev_idx, task_id):
        """タスクノードが存在しない場合のフォールバック特徴量"""
        # 開発者は存在するので、平均的なタスクとの比較を使用
        dev_emb = self.embeddings["dev"][dev_idx]
        avg_task_emb = torch.mean(self.embeddings["task"], dim=0)

        features = []

        # 1. 平均タスクとの類似度（低めの値になると予想）
        similarity = (
            F.cosine_similarity(dev_emb.unsqueeze(0), avg_task_emb.unsqueeze(0)).item()
            * 0.5
        )  # ペナルティとして半分にする
        features.append(similarity)

        # 2. 開発者の専門性は計算可能
        all_task_sims = F.cosine_similarity(
            dev_emb.unsqueeze(0), self.embeddings["task"]
        )
        dev_expertise = torch.mean(
            torch.topk(all_task_sims, k=min(10, all_task_sims.size(0))).values
        ).item()
        features.append(dev_expertise)

        # 3. 平均的な人気度スコア（中程度の値）
        features.append(0.3)

        # 🆕 4-5. 協力ネットワーク特徴量（利用可能な場合）
        if self.dev_network is not None:
            # 開発者が存在するので協力特徴量を計算
            collab_strength = self._calculate_collaboration_strength(dev_idx)
            features.append(collab_strength)

            centrality = self._calculate_network_centrality(dev_idx)
            features.append(centrality)

        return features

    def get_feature_names(self):
        """GNN特徴量の名前リストを返す"""
        if not self.model:
            return []

        base_features = ["gnn_similarity", "gnn_dev_expertise", "gnn_task_popularity"]

        # 🆕 協力ネットワークが利用可能な場合、追加の特徴量を含める
        if self.dev_network is not None:
            base_features.extend(
                [
                    "gnn_collaboration_strength",  # 開発者の協力ネットワーク内での重要度
                    "gnn_network_centrality",  # ネットワーク内での中心性
                ]
            )

        return base_features

    def print_statistics(self):
        """GNN特徴量抽出の統計を表示"""
        total = self.stats["total_requests"]
        if total == 0:
            print("No GNN feature requests yet.")
            return

        print(f"\n=== GNN Feature Extraction Statistics ===")
        print(f"Total requests: {total}")
        print(
            f"Full features (both nodes found): {self.stats['full_features']} ({self.stats['full_features']/total*100:.1f}%)"
        )
        print(
            f"Missing developer: {self.stats['missing_dev']} ({self.stats['missing_dev']/total*100:.1f}%)"
        )
        print(
            f"Missing task: {self.stats['missing_task']} ({self.stats['missing_task']/total*100:.1f}%)"
        )
        print(
            f"Missing both: {self.stats['missing_both']} ({self.stats['missing_both']/total*100:.1f}%)"
        )
        print(f"Errors: {self.stats['errors']} ({self.stats['errors']/total*100:.1f}%)")

        if self.online_learning:
            print(f"Online learning enabled: True")
            print(f"GNN updates performed: {self.stats['updates']}")
            print(f"Interaction buffer size: {len(self.interaction_buffer)}")
            print(f"Update frequency: every {self.update_frequency} requests")
        else:
            print(f"Online learning enabled: False")

        print("=" * 45)

    def save_updated_model(self, save_path=None):
        """更新されたGNNモデルを保存"""
        if not self.model:
            print("保存するモデルがありません")
            return

        if save_path is None:
            save_path = "data/gnn_model_online_updated.pt"

        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"💾 更新されたGNNモデルを保存: {save_path}")

            # グラフデータも保存
            graph_save_path = save_path.replace("model", "graph")
            torch.save(self.graph_data.cpu(), graph_save_path)
            print(f"💾 更新されたグラフデータを保存: {graph_save_path}")

        except Exception as e:
            print(f"❌ モデル保存中にエラー: {e}")

    def reset_interaction_buffer(self):
        """インタラクションバッファをリセット"""
        self.interaction_buffer.clear()
        print("🔄 インタラクションバッファをリセットしました")
