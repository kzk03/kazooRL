#!/usr/bin/env python3
"""
観測データ処理システム
高次元・複合観測を効率的に処理
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ObservationProcessor:
    """観測データの前処理・正規化システム"""
    
    def __init__(self, config):
        self.config = config
        self.feature_dim = config.get('processed_feature_dim', 64)
        self.task_feature_dim = config.get('task_feature_dim', 16)
        self.dev_feature_dim = config.get('dev_feature_dim', 16)
        self.context_feature_dim = config.get('context_feature_dim', 16)
        self.gnn_feature_dim = config.get('gnn_feature_dim', 16)
        
        # 正規化器
        self.task_scaler = StandardScaler()
        self.dev_scaler = StandardScaler()
        self.context_scaler = StandardScaler()
        self.gnn_scaler = StandardScaler()
        
        # 初期化フラグ
        self.scalers_fitted = False
    
    def process_observation(self, raw_obs: Dict, env_state: Dict, 
                          agent_id: str) -> np.ndarray:
        """
        生の観測データを処理済み特徴量に変換
        
        Args:
            raw_obs: 生の観測データ
            env_state: 環境状態
            agent_id: エージェントID
        
        Returns:
            処理済み特徴量ベクトル
        """
        # 1. タスク特徴量
        task_features = self._extract_task_features(env_state)
        
        # 2. 開発者特徴量
        dev_features = self._extract_developer_features(agent_id, env_state)
        
        # 3. コンテキスト特徴量
        context_features = self._extract_context_features(env_state)
        
        # 4. GNN特徴量（利用可能な場合）
        gnn_features = self._extract_gnn_features(raw_obs)
        
        # 5. 特徴量の正規化
        if self.scalers_fitted:
            task_features = self.task_scaler.transform([task_features])[0]
            dev_features = self.dev_scaler.transform([dev_features])[0]
            context_features = self.context_scaler.transform([context_features])[0]
            gnn_features = self.gnn_scaler.transform([gnn_features])[0]
        else:
            # 初回は正規化をスキップ（後でfit）
            pass
        
        # 6. 特徴量の結合
        combined_features = np.concatenate([
            task_features,
            dev_features,
            context_features,
            gnn_features
        ])
        
        # 7. 次元調整
        if len(combined_features) > self.feature_dim:
            # PCA的な次元削減（簡易版）
            combined_features = combined_features[:self.feature_dim]
        elif len(combined_features) < self.feature_dim:
            # パディング
            padding = np.zeros(self.feature_dim - len(combined_features))
            combined_features = np.concatenate([combined_features, padding])
        
        return combined_features.astype(np.float32)
    
    def _extract_task_features(self, env_state: Dict) -> np.ndarray:
        """タスク関連特徴量を抽出"""
        features = np.zeros(self.task_feature_dim)
        
        backlog = env_state.get('backlog', [])
        tasks_in_progress = env_state.get('tasks_in_progress', {})
        
        # 基本統計
        features[0] = len(backlog)  # 残りタスク数
        features[1] = len(tasks_in_progress)  # 進行中タスク数
        
        if backlog:
            # タスクの複雑さ分布
            complexities = []
            priorities = []
            
            for task in backlog:
                # 複雑さ推定
                complexity = self._estimate_task_complexity(task.__dict__)
                complexities.append(complexity)
                
                # 優先度推定
                priority = self._estimate_task_priority(task.__dict__)
                priorities.append(priority)
            
            features[2] = np.mean(complexities)  # 平均複雑さ
            features[3] = np.std(complexities)   # 複雑さの分散
            features[4] = np.mean(priorities)    # 平均優先度
            features[5] = np.std(priorities)     # 優先度の分散
            
            # ラベル分布
            label_counts = {'bug': 0, 'enhancement': 0, 'documentation': 0, 'other': 0}
            for task in backlog:
                labels = task.__dict__.get('labels', [])
                if labels:
                    for label in labels:
                        label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
                        if 'bug' in label_name:
                            label_counts['bug'] += 1
                        elif 'enhancement' in label_name or 'feature' in label_name:
                            label_counts['enhancement'] += 1
                        elif 'doc' in label_name:
                            label_counts['documentation'] += 1
                        else:
                            label_counts['other'] += 1
            
            total_tasks = len(backlog)
            features[6] = label_counts['bug'] / total_tasks
            features[7] = label_counts['enhancement'] / total_tasks
            features[8] = label_counts['documentation'] / total_tasks
            features[9] = label_counts['other'] / total_tasks
        
        return features
    
    def _extract_developer_features(self, agent_id: str, env_state: Dict) -> np.ndarray:
        """開発者関連特徴量を抽出"""
        features = np.zeros(self.dev_feature_dim)
        
        developers = env_state.get('developers', {})
        assignments = env_state.get('assignments', {})
        
        if agent_id in developers:
            dev_info = developers[agent_id]
            profile = dev_info['profile']
            
            # 基本プロファイル
            features[0] = min(profile.get('rank', 5000) / 5000.0, 1.0)
            features[1] = min(profile.get('total_commits', 0) / 1000.0, 1.0)
            features[2] = min(profile.get('python_commits', 0) / 100.0, 1.0)
            features[3] = min(profile.get('javascript_commits', 0) / 100.0, 1.0)
            features[4] = min(profile.get('bug_fixes', 0) / 50.0, 1.0)
            features[5] = min(profile.get('doc_commits', 0) / 20.0, 1.0)
            
            # 現在の作業負荷
            current_workload = len(assignments.get(agent_id, set()))
            features[6] = min(current_workload / 10.0, 1.0)
            
            # 相対的な能力（全開発者との比較）
            all_ranks = [dev['profile'].get('rank', 5000) for dev in developers.values()]
            all_commits = [dev['profile'].get('total_commits', 0) for dev in developers.values()]
            
            if all_ranks:
                rank_percentile = (sorted(all_ranks).index(profile.get('rank', 5000)) + 1) / len(all_ranks)
                features[7] = rank_percentile
            
            if all_commits:
                commit_percentile = (sorted(all_commits).index(profile.get('total_commits', 0)) + 1) / len(all_commits)
                features[8] = commit_percentile
        
        return features
    
    def _extract_context_features(self, env_state: Dict) -> np.ndarray:
        """コンテキスト特徴量を抽出"""
        features = np.zeros(self.context_feature_dim)
        
        current_time = env_state.get('current_time')
        assignments = env_state.get('assignments', {})
        developers = env_state.get('developers', {})
        
        # 時間的特徴
        if current_time:
            features[0] = current_time.hour / 24.0  # 時刻
            features[1] = current_time.weekday() / 7.0  # 曜日
            features[2] = current_time.month / 12.0  # 月
        
        # 全体的な作業負荷分布
        if assignments and developers:
            workloads = [len(tasks) for tasks in assignments.values()]
            if workloads:
                features[3] = np.mean(workloads) / 10.0  # 平均作業負荷
                features[4] = np.std(workloads) / 5.0    # 作業負荷の分散
                features[5] = max(workloads) / 20.0      # 最大作業負荷
                features[6] = min(workloads) / 10.0      # 最小作業負荷
        
        # アクティブ開発者の割合
        active_devs = sum(1 for tasks in assignments.values() if len(tasks) > 0)
        total_devs = len(developers)
        if total_devs > 0:
            features[7] = active_devs / total_devs
        
        return features
    
    def _extract_gnn_features(self, raw_obs: Dict) -> np.ndarray:
        """GNN特徴量を抽出・処理"""
        features = np.zeros(self.gnn_feature_dim)
        
        if 'gnn_embeddings' in raw_obs:
            gnn_emb = raw_obs['gnn_embeddings']
            if isinstance(gnn_emb, (list, np.ndarray)):
                gnn_emb = np.array(gnn_emb)
                
                # 次元調整
                if len(gnn_emb) >= self.gnn_feature_dim:
                    features = gnn_emb[:self.gnn_feature_dim]
                else:
                    features[:len(gnn_emb)] = gnn_emb
        
        return features
    
    def _estimate_task_complexity(self, task: Dict) -> float:
        """タスクの複雑さを推定"""
        complexity = 0.0
        
        # コメント数
        comments_count = task.get('comments_count', 0)
        complexity += min(comments_count / 10.0, 0.3)
        
        # テキスト長
        title = task.get('title', '') or ''  # Noneの場合は空文字列
        body = task.get('body', '') or ''    # Noneの場合は空文字列
        title_length = len(title)
        body_length = len(body)
        complexity += min((title_length + body_length) / 1000.0, 0.4)
        
        # ラベル数
        labels = task.get('labels', []) or []  # Noneの場合は空リスト
        labels_count = len(labels)
        complexity += min(labels_count / 5.0, 0.3)
        
        return min(complexity, 1.0)
    
    def _estimate_task_priority(self, task: Dict) -> float:
        """タスクの優先度を推定"""
        priority = 0.5  # デフォルト優先度
        
        title = (task.get('title', '') or '').lower()
        body = (task.get('body', '') or '').lower()
        text = f"{title} {body}"
        
        # 緊急度キーワード
        if any(word in text for word in ['urgent', 'critical', 'hotfix', 'emergency']):
            priority += 0.4
        elif any(word in text for word in ['bug', 'error', 'fix']):
            priority += 0.2
        elif any(word in text for word in ['enhancement', 'feature']):
            priority += 0.1
        
        return min(priority, 1.0)
    
    def fit_scalers(self, observation_samples: List[Dict], env_states: List[Dict], 
                   agent_ids: List[str]):
        """正規化器を学習データでフィット"""
        task_features_list = []
        dev_features_list = []
        context_features_list = []
        gnn_features_list = []
        
        for obs, env_state, agent_id in zip(observation_samples, env_states, agent_ids):
            task_features = self._extract_task_features(env_state)
            dev_features = self._extract_developer_features(agent_id, env_state)
            context_features = self._extract_context_features(env_state)
            gnn_features = self._extract_gnn_features(obs)
            
            task_features_list.append(task_features)
            dev_features_list.append(dev_features)
            context_features_list.append(context_features)
            gnn_features_list.append(gnn_features)
        
        # 正規化器をフィット
        if task_features_list:
            self.task_scaler.fit(task_features_list)
            self.dev_scaler.fit(dev_features_list)
            self.context_scaler.fit(context_features_list)
            self.gnn_scaler.fit(gnn_features_list)
            self.scalers_fitted = True