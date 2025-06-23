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
        """
        self.all_labels = cfg.features.get('all_labels', ['bug', 'enhancement', 'documentation', 'question', 'help wanted'])
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}
        
        self.LABEL_TO_SKILLS = cfg.features.get('label_to_skills', {
            'bug': {'debugging', 'analysis'},
            'enhancement': {'python', 'design'},
            'documentation': {'writing'},
            'question': {'communication'},
            'help wanted': {'collaboration'}
        })
        
        self.feature_names = self._define_feature_names()
        print(f"FeatureExtractor initialized with {len(self.feature_names)} features.")
        print(f"Feature names: {self.feature_names}")


    def _define_feature_names(self) -> list[str]:
        """
        生成する特徴量の名前を順番通りに定義し、リストとして返す。
        """
        names = []
        names.extend([
            'task_neglect_time_hours', 'task_discussion_activity',
            'task_text_length', 'task_code_block_count'
        ])
        names.extend([f'task_label_{label}' for label in self.all_labels])
        names.extend(['dev_recent_activity_count', 'dev_current_workload'])
        names.extend(['match_skill_intersection_count', 'match_file_experience_count'])
        names.extend([f'match_affinity_for_{label}' for label in self.all_labels])
        return names

    def get_features(self, task, developer, env) -> np.ndarray:
        """
        指定されたタスクと開発者のペアに関する特徴量ベクトルを生成する。
        """
        feature_values = []
        
        # ▼▼▼【ここからが修正箇所】▼▼▼
        # developer オブジェクトへのアクセスを .name から ['name'] の形式に変更
        developer_name = developer['name']
        developer_profile = developer['profile']
        # ▲▲▲【ここまでが修正箇所】▲▲▲

        # === カテゴリ1: タスク自体の特徴 ===
        neglect_time_hours = (env.current_time - task.updated_at).total_seconds() / 3600.0
        feature_values.append(neglect_time_hours)
        feature_values.append(float(task.comments))
        feature_values.append(float(len(task.body)))
        feature_values.append(float(task.body.count('```') // 2))

        label_vec = [0.0] * len(self.all_labels)
        for label_name in task.labels:
            if label_name in self.label_to_idx:
                label_vec[self.label_to_idx[label_name]] = 1.0
        feature_values.extend(label_vec)

        # === カテゴリ2: 開発者の特徴 ===
        recent_activity_count = float(len(env.dev_action_history.get(developer_name, [])))
        feature_values.append(recent_activity_count)
        
        workload = float(len(env.assignments.get(developer_name, set())))
        feature_values.append(workload)
        
        # === カテゴリ3: 相互作用（マッチング）の特徴 ===
        required_skills = set().union(*(self.LABEL_TO_SKILLS.get(label_name, set()) for label_name in task.labels))
        developer_skills = set(developer_profile.get('skills', []))
        skill_intersection_count = float(len(required_skills.intersection(developer_skills)))
        feature_values.append(skill_intersection_count)

        pr_changed_files = set(getattr(task, 'changed_files', []))
        dev_touched_files = set(developer_profile.get('touched_files', []))
        file_exp_count = float(len(pr_changed_files.intersection(dev_touched_files)))
        feature_values.append(file_exp_count)

        affinity_vec = [0.0] * len(self.all_labels)
        dev_affinity_profile = developer_profile.get('label_affinity', {})
        for i, label_name in enumerate(self.all_labels):
            if label_vec[i] == 1.0:
                affinity_vec[i] = dev_affinity_profile.get(label_name, 0.0)
        feature_values.extend(affinity_vec)

        if len(feature_values) != len(self.feature_names):
            raise ValueError("Feature dimension mismatch.")
            
        return np.array(feature_values, dtype=np.float32)

