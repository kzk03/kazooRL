from datetime import datetime

import numpy as np

try:
    from kazoo.features.gnn_feature_extractor import \
        GNNFeatureExtractor as IRLGNNFeatureExtractor
except ImportError:
    print("Warning: GNN feature extractor not available")
    IRLGNNFeatureExtractor = None


class FeatureExtractor:
    def __init__(self, cfg):
        self.all_labels = cfg.features.get(
            "all_labels",
            ["bug", "enhancement", "documentation", "question", "help wanted"],
        )
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

        self.LABEL_TO_SKILLS = cfg.features.get(
            "label_to_skills",
            {
                "bug": {"debugging", "analysis"},
                "enhancement": {"python", "design"},
                "documentation": {"writing"},
                "question": {"communication"},
                "help wanted": {"collaboration"},
            },
        )

        # GNNFeatureExtractorを初期化
        self.gnn_extractor = None
        if (
            IRLGNNFeatureExtractor
            and hasattr(cfg, "irl")
            and cfg.irl.get("use_gnn", False)
        ):
            try:
                # IRLに必要な設定が存在する場合のみ初期化
                if hasattr(cfg.irl, "gnn_graph_path") and hasattr(
                    cfg.irl, "gnn_model_path"
                ):
                    self.gnn_extractor = IRLGNNFeatureExtractor(cfg)
                    print("✅ GNN feature extractor initialized")
                else:
                    print("Warning: GNN paths not configured in IRL section")
            except Exception as e:
                print(f"Warning: Failed to initialize GNN feature extractor: {e}")
                self.gnn_extractor = None

        # データ内での最新日時を基準にするため、初期化時にNoneに設定
        # 実際の値は初回のget_features呼び出し時に計算される
        self.data_max_date = None

        self.feature_names = self._define_feature_names()
        print(f"FeatureExtractor initialized with {len(self.feature_names)} features.")
        print(f"Feature names: {self.feature_names}")

    def _define_feature_names(self) -> list[str]:
        """
        生成する特徴量の名前を順番通りに定義し、リストとして返す。
        """
        names = []
        names.extend(
            [
                "task_days_since_last_activity",  # 最後の活動（コメント等）からの日数
                "task_discussion_activity",
                "task_text_length",
                "task_code_block_count",
            ]
        )
        names.extend([f"task_label_{label}" for label in self.all_labels])
        names.extend(
            [
                "dev_recent_activity_count",
                "dev_current_workload",
                "dev_total_lines_changed",
            ]
        )
        names.extend(["match_skill_intersection_count", "match_file_experience_count"])
        names.extend([f"match_affinity_for_{label}" for label in self.all_labels])
        # ▼▼▼【追加】GNN特徴量名を追加▼▼▼
        if self.gnn_extractor:
            gnn_names = self.gnn_extractor.get_feature_names()
            names.extend(gnn_names)

        return names

    def _get_data_max_date(self, env):
        """
        環境内の全タスクから最新の更新日時を取得する。
        初回呼び出し時にキャッシュして、以降は再利用する。
        """
        if self.data_max_date is None:
            max_date = None
            # バックログ内の全タスクをチェック
            for task in env.backlog:
                if max_date is None or task.updated_at > max_date:
                    max_date = task.updated_at
            # アサインされているタスクもチェック
            for assigned_tasks in env.assignments.values():
                for task in assigned_tasks:
                    if task.updated_at > max_date:
                        max_date = task.updated_at

            self.data_max_date = max_date
            print(f"[FeatureExtractor] Data max date set to: {self.data_max_date}")

        return self.data_max_date

    def get_features(self, task, developer, env) -> np.ndarray:
        """
        指定されたタスクと開発者のペアに関する特徴量ベクトルを生成する。
        """
        feature_values = []

        # ▼▼▼【ここからが修正箇所】▼▼▼
        # developer オブジェクトへのアクセスを .name から ['name'] の形式に変更
        developer_name = developer["name"]
        developer_profile = developer["profile"]
        # ▲▲▲【ここまでが修正箇所】▲▲▲

        # === カテゴリ1: タスク自体の特徴 ===
        # データ内での最新日時を基準とした相対的な放置時間を計算
        data_max_date = self._get_data_max_date(env)
        neglect_time_days = (data_max_date - task.updated_at).total_seconds() / (
            3600.0 * 24.0
        )  # 秒 → 日数に変換
        feature_values.append(neglect_time_days)
        feature_values.append(float(task.comments))
        feature_values.append(float(len(task.body)))
        feature_values.append(float(task.body.count("```") // 2))

        label_vec = [0.0] * len(self.all_labels)
        for label_name in task.labels:
            if label_name in self.label_to_idx:
                label_vec[self.label_to_idx[label_name]] = 1.0
        feature_values.extend(label_vec)

        # === カテゴリ2: 開発者の特徴 ===
        recent_activity_count = float(
            len(env.dev_action_history.get(developer_name, []))
        )
        feature_values.append(recent_activity_count)

        workload = float(len(env.assignments.get(developer_name, set())))
        feature_values.append(workload)

        # 総変更行数を追加
        total_lines_changed = float(developer_profile.get("total_lines_changed", 0))
        feature_values.append(total_lines_changed)

        # === カテゴリ3: 相互作用（マッチング）の特徴 ===
        required_skills = set().union(
            *(self.LABEL_TO_SKILLS.get(label_name, set()) for label_name in task.labels)
        )
        developer_skills = set(developer_profile.get("skills", []))
        skill_intersection_count = float(
            len(required_skills.intersection(developer_skills))
        )
        feature_values.append(skill_intersection_count)

        pr_changed_files = set(getattr(task, "changed_files", []))
        dev_touched_files = set(developer_profile.get("touched_files", []))
        file_exp_count = float(len(pr_changed_files.intersection(dev_touched_files)))
        feature_values.append(file_exp_count)

        affinity_vec = [0.0] * len(self.all_labels)
        dev_affinity_profile = developer_profile.get("label_affinity", {})
        for i, label_name in enumerate(self.all_labels):
            if label_vec[i] == 1.0:
                affinity_vec[i] = dev_affinity_profile.get(label_name, 0.0)
        feature_values.extend(affinity_vec)

        if self.gnn_extractor:
            gnn_features = self.gnn_extractor.get_gnn_features(task, developer, env)
            feature_values.extend(gnn_features)

        if len(feature_values) != len(self.feature_names):
            raise ValueError("Feature dimension mismatch.")

        return np.array(feature_values, dtype=np.float32)

    def print_gnn_statistics(self):
        """GNN特徴量抽出の統計を表示"""
        if self.gnn_extractor:
            self.gnn_extractor.print_statistics()
        else:
            print("GNN feature extractor not available.")


import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    強化学習エージェントのポリシーネットワーク用。
    GNNの特徴量を含む辞書型の観測空間から、単一の特徴量ベクトルを生成する。
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        """
        コンストラクタ。最終的に生成する特徴量ベクトルの次元数を計算して親クラスに渡す。
        """
        # 観測空間の各要素の次元数を取得
        simple_obs_dim = observation_space["simple_obs"].shape[0]
        # gnn_embeddings は (ノード数, 特徴量次元数) なので、特徴量次元数を取得
        gnn_embedding_dim = observation_space["gnn_embeddings"].shape[1]

        # 最終的な特徴量次元数 = simple_obsの次元 + gnn_embeddingsの次元（プーリング後）
        features_dim = simple_obs_dim + gnn_embedding_dim

        # 親クラスのコンストラクタを呼び出す
        super().__init__(observation_space, features_dim=features_dim)

        # 必要であれば、ここで線形層などを定義することも可能
        # self.linear = nn.Sequential(nn.Linear(features_dim, 256), nn.ReLU())
        # super().__init__(observation_space, features_dim=256)

    def forward(self, observations: dict) -> torch.Tensor:
        """
        観測辞書を受け取り、単一のテンソルに変換して返す。
        このメソッドは、エージェントが行動を決定するたびに内部で呼び出される。
        """
        # 観測辞書から各データを取り出す
        simple_obs = observations["simple_obs"]
        gnn_embeddings = observations["gnn_embeddings"]

        # gnn_embeddings は (バッチサイズ, ノード数, 特徴量次元数) の形をしている
        # これを固定長のベクトルにするために、ノードの軸でプーリング処理を行う
        # (例: Global Average Pooling)
        # dim=1 はノードの次元
        pooled_gnn_features = torch.mean(gnn_embeddings, dim=1)

        # 2つの特徴量テンソルを結合する
        # これにより、エージェントは両方の情報を同時に見ることができる
        combined_features = torch.cat([simple_obs, pooled_gnn_features], dim=1)

        # もし線形層を定義した場合は、ここで通す
        # return self.linear(combined_features)

        return combined_features
