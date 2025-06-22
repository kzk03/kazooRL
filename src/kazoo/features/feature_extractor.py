from datetime import datetime

import numpy as np


class FeatureExtractor:
    """
    シミュレーション環境の状態（タスク、開発者）から、
    機械学習モデル用の特徴量ベクトルを生成するクラス。
    """

    def __init__(self, cfg):
        """
        設定ファイルから必要な情報を読み込み、初期化する。

        Args:
            cfg: プロジェクトの設定オブジェクト (例: OmegaConf)
        """
        # 特徴量エンジニアリングで使用する固定的な設定
        # cfgから読み込むか、ここに直接定義する
        # 例: self.all_labels = cfg.labels.all
        self.all_labels = [
            "bug",
            "enhancement",
            "documentation",
            "question",
            "help wanted",
        ]
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

        # ラベルと要求スキルのマッピング（手動で定義）
        self.LABEL_TO_SKILLS = {
            "bug": {"debugging", "analysis"},
            "enhancement": {"python", "design"},
            "documentation": {"writing"},
            "question": {"communication"},
            "help wanted": {"collaboration"},
        }

        # このクラスが生成する特徴量の名前を順番通りに保持するリスト
        # 分析時に重みと対応させるために非常に重要
        self.feature_names = self._define_feature_names()
        print(f"FeatureExtractor initialized with {len(self.feature_names)} features.")
        print(f"Feature names: {self.feature_names}")

    def _define_feature_names(self) -> list[str]:
        """
        生成する特徴量の名前を順番通りに定義し、リストとして返す。
        分析の際に、学習した重みと特徴量を対応させるために使用する。
        """
        names = []

        # カテゴリ1: タスク自体の特徴
        names.extend(
            [
                "task_neglect_time_hours",
                "task_discussion_activity",
                "task_text_length",
                "task_code_block_count",
            ]
        )
        names.extend([f"task_label_{label}" for label in self.all_labels])

        # カテゴリ2: 開発者の特徴
        names.extend(
            [
                "dev_recent_activity_count",
                "dev_current_workload",
            ]
        )

        # カテゴリ3: 相互作用（マッチング）の特徴
        names.extend(
            [
                "match_skill_intersection_count",
                "match_file_experience_count",
            ]
        )
        names.extend([f"match_affinity_for_{label}" for label in self.all_labels])

        return names

    def get_features(self, task, developer, env) -> np.ndarray:
        """
        指定されたタスクと開発者のペアに関する特徴量ベクトルを生成する。
        _define_feature_namesで定義された順番通りに値を計算し、リストに追加していく。

        Args:
            task: タスクオブジェクト
            developer: 開発者オブジェクト
            env: 環境オブジェクト

        Returns:
            np.ndarray: 計算された特徴量を格納したNumpy配列
        """

        # このリストに計算した特徴量を追加していく
        feature_values = []

        # === カテゴリ1: タスク自体の特徴 ===
        # 1.1 放置時間 (時間単位)
        neglect_time_hours = (
            env.current_time - task.updated_at
        ).total_seconds() / 3600.0
        feature_values.append(neglect_time_hours)

        # 1.2 議論の活発度 (コメント数)
        discussion_activity = float(task.comments)
        feature_values.append(discussion_activity)

        # 1.3 テキストの複雑度
        feature_values.append(float(len(task.body)))
        feature_values.append(float(task.body.count("```") // 2))

        # 1.4 ラベル情報 (One-Hot)
        label_vec = [0.0] * len(self.all_labels)
        for label_name in task.labels:
            if label_name in self.label_to_idx:
                label_vec[self.label_to_idx[label_name]] = 1.0
        feature_values.extend(label_vec)

        # === カテゴリ2: 開発者の特徴 ===
        # 2.1 最近の活動量
        recent_activity_count = float(
            len(env.dev_action_history.get(developer.name, []))
        )
        feature_values.append(recent_activity_count)

        # 2.2 現在の作業負荷
        workload = float(len(env.assignments.get(developer.name, set())))
        feature_values.append(workload)

        # === カテゴリ3: 相互作用（マッチング）の特徴 ===
        # 3.1 スキルの一致度
        required_skills = set().union(
            *(self.LABEL_TO_SKILLS.get(label_name, set()) for label_name in task.labels)
        )
        developer_skills = set(developer.profile.get("skills", []))
        skill_intersection_count = float(
            len(required_skills.intersection(developer_skills))
        )
        feature_values.append(skill_intersection_count)

        # 3.2 ファイルの編集経験
        pr_changed_files = set(task.changed_files)
        dev_touched_files = set(developer.profile.get("touched_files", []))
        file_exp_count = float(len(pr_changed_files.intersection(dev_touched_files)))
        feature_values.append(file_exp_count)

        # 3.3 過去の作業親和性
        affinity_vec = [0.0] * len(self.all_labels)
        dev_affinity_profile = developer.profile.get("label_affinity", {})
        for i, label_name in enumerate(self.all_labels):
            if (
                label_vec[i] == 1.0
            ):  # タスクがこのラベルを持っている場合のみ親和性を特徴量とする
                affinity_vec[i] = dev_affinity_profile.get(label_name, 0.0)
        feature_values.extend(affinity_vec)

        # --- 最終チェックと返却 ---
        # 定義した特徴量の数と、実際に計算した特徴量の数が一致しているか確認
        if len(feature_values) != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch. Expected {len(self.feature_names)} features, "
                f"but got {len(feature_values)}."
            )

        return np.array(feature_values, dtype=np.float32)
