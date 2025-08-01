#!/usr/bin/env python3
"""
改良された観測空間システム
高次元・複合観測の問題を解決
"""

from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ImprovedObservationSpace:
    """改良された観測空間管理システム"""
    
    def __init__(self, config):
        self.config = config
        
        # 観測空間の構成
        self.obs_components = {
            'agent_features': 16,        # エージェント固有特徴
            'task_context': 32,          # タスクコンテキスト特徴
            'environment_state': 16,     # 環境状態特徴
            'temporal_features': 8,      # 時系列特徴
            'graph_embeddings': 32       # グラフ埋め込み（圧縮済み）
        }
        
        self.total_obs_dim = sum(self.obs_components.values())  # 104次元
        
        # 正規化器
        self.scalers = {
            component: StandardScaler() 
            for component in self.obs_components.keys()
        }
        
        # 特徴量履歴（時系列特徴用）
        self.feature_history = {}
        self.history_length = 5
        
        # 初期化フラグ
        self.initialized = False
    
    def get_observation(self, agent_id: str, env_state: Dict, 
                       available_tasks: List[Dict]) -> np.ndarray:
        """
        改良された観測ベクトルを生成
        
        Args:
            agent_id: エージェントID
            env_state: 環境状態
            available_tasks: 利用可能なタスクリスト
            
        Returns:
            observation: 正規化済み観測ベクトル
        """
        
        # 各コンポーネントの特徴量を計算
        agent_features = self._extract_agent_features(agent_id, env_state)
        task_context = self._extract_task_context_features(available_tasks, env_state)
        env_features = self._extract_environment_features(env_state)
        temporal_features = self._extract_temporal_features(agent_id, env_state)
        graph_embeddings = self._extract_compressed_graph_embeddings(agent_id, env_state)
        
        # 特徴量の結合
        raw_observation = np.concatenate([
            agent_features,
            task_context,
            env_features,
            temporal_features,
            graph_embeddings
        ])
        
        # 正規化
        normalized_obs = self._normalize_observation(raw_observation, agent_id)
        
        # 履歴更新
        self._update_feature_history(agent_id, normalized_obs)
        
        return normalized_obs.astype(np.float32)
    
    def _extract_agent_features(self, agent_id: str, env_state: Dict) -> np.ndarray:
        """エージェント固有特徴の抽出"""
        features = np.zeros(self.obs_components['agent_features'])
        
        try:
            developer = env_state['developers'][agent_id]
            profile = developer['profile']
            
            # 基本統計
            features[0] = min(profile.get('total_commits', 0) / 1000.0, 1.0)
            features[1] = min(profile.get('rank', 5000) / 5000.0, 1.0)
            features[2] = min(profile.get('followers', 0) / 100.0, 1.0)
            features[3] = min(profile.get('public_repos', 0) / 50.0, 1.0)
            
            # スキル特徴
            features[4] = min(profile.get('python_commits', 0) / 500.0, 1.0)
            features[5] = min(profile.get('javascript_commits', 0) / 300.0, 1.0)
            features[6] = min(profile.get('bug_fixes', 0) / 100.0, 1.0)
            features[7] = min(profile.get('doc_commits', 0) / 50.0, 1.0)
            features[8] = min(profile.get('test_commits', 0) / 80.0, 1.0)
            
            # 活動パターン
            recent_activity = len(env_state.get('dev_action_history', {}).get(agent_id, []))
            features[9] = min(recent_activity / 10.0, 1.0)
            
            # 現在の作業負荷
            current_workload = len(env_state.get('assignments', {}).get(agent_id, set()))
            features[10] = min(current_workload / 5.0, 1.0)
            
            # 協力度指標
            features[11] = min(profile.get('collaborations', 0) / 20.0, 1.0)
            
            # 専門性指標
            features[12] = profile.get('specialization_score', 0.5)
            
            # 学習能力指標
            features[13] = profile.get('learning_rate', 0.5)
            
            # 信頼性指標
            features[14] = profile.get('reliability_score', 0.5)
            
            # 可用性指標
            features[15] = profile.get('availability', 1.0)
            
        except Exception as e:
            # エラー時はデフォルト値
            features[1] = 0.5  # 平均的なランク
            features[15] = 1.0  # 利用可能
        
        return features
    
    def _extract_task_context_features(self, available_tasks: List[Dict], 
                                     env_state: Dict) -> np.ndarray:
        """タスクコンテキスト特徴の抽出"""
        features = np.zeros(self.obs_components['task_context'])
        
        if not available_tasks:
            return features
        
        try:
            # タスク統計
            features[0] = min(len(available_tasks) / 50.0, 1.0)  # 利用可能タスク数
            
            # 複雑度統計
            complexities = []
            priorities = []
            ages = []
            
            for task in available_tasks[:10]:  # 上位10タスクのみ分析
                # 複雑度（コメント数、ラベル数）
                complexity = (
                    len(task.get('labels', [])) * 0.3 +
                    min(task.get('comments_count', 0) / 10.0, 1.0) * 0.7
                )
                complexities.append(complexity)
                
                # 優先度（ラベルベース）
                priority = self._calculate_task_priority(task)
                priorities.append(priority)
                
                # 経過時間（簡略化）
                ages.append(0.5)  # 実装簡略化
            
            # 統計値
            if complexities:
                features[1] = np.mean(complexities)
                features[2] = np.std(complexities)
                features[3] = np.max(complexities)
                features[4] = np.min(complexities)
            
            if priorities:
                features[5] = np.mean(priorities)
                features[6] = np.std(priorities)
            
            # カテゴリ分布
            categories = {'bug': 0, 'feature': 0, 'doc': 0, 'other': 0}
            for task in available_tasks[:20]:
                category = self._categorize_task(task)
                categories[category] += 1
            
            total_tasks = sum(categories.values())
            if total_tasks > 0:
                features[7] = categories['bug'] / total_tasks
                features[8] = categories['feature'] / total_tasks
                features[9] = categories['doc'] / total_tasks
                features[10] = categories['other'] / total_tasks
            
            # 言語分布
            languages = {'python': 0, 'javascript': 0, 'other': 0}
            for task in available_tasks[:20]:
                lang = self._detect_task_language(task)
                languages[lang] += 1
            
            total_lang_tasks = sum(languages.values())
            if total_lang_tasks > 0:
                features[11] = languages['python'] / total_lang_tasks
                features[12] = languages['javascript'] / total_lang_tasks
                features[13] = languages['other'] / total_lang_tasks
            
            # 緊急度分布
            urgency_levels = [self._calculate_task_urgency(task) for task in available_tasks[:10]]
            if urgency_levels:
                features[14] = np.mean(urgency_levels)
                features[15] = np.max(urgency_levels)
            
            # 協力度要求
            collab_requirements = [self._calculate_collaboration_requirement(task) for task in available_tasks[:10]]
            if collab_requirements:
                features[16] = np.mean(collab_requirements)
            
            # 学習機会
            learning_opportunities = [self._calculate_learning_opportunity(task) for task in available_tasks[:10]]
            if learning_opportunities:
                features[17] = np.mean(learning_opportunities)
            
            # 残りの特徴量は将来の拡張用
            
        except Exception as e:
            pass  # デフォルト値（ゼロ）を使用
        
        return features
    
    def _extract_environment_features(self, env_state: Dict) -> np.ndarray:
        """環境状態特徴の抽出"""
        features = np.zeros(self.obs_components['environment_state'])
        
        try:
            # 全体的な作業負荷
            total_assignments = sum(len(tasks) for tasks in env_state.get('assignments', {}).values())
            total_developers = len(env_state.get('developers', {}))
            
            if total_developers > 0:
                features[0] = min(total_assignments / (total_developers * 3.0), 1.0)  # 平均作業負荷
            
            # 進行中タスク数
            features[1] = min(len(env_state.get('tasks_in_progress', {})) / 50.0, 1.0)
            
            # 完了タスク数
            features[2] = min(len(env_state.get('completed_tasks', [])) / 100.0, 1.0)
            
            # バックログサイズ
            features[3] = min(len(env_state.get('backlog', [])) / 200.0, 1.0)
            
            # アクティブ開発者数
            active_devs = len([dev for dev, actions in env_state.get('dev_action_history', {}).items() if actions])
            features[4] = min(active_devs / total_developers, 1.0) if total_developers > 0 else 0
            
            # 時間進行
            current_time = env_state.get('current_time')
            start_time = env_state.get('start_time')
            if current_time and start_time:
                time_progress = (current_time - start_time).total_seconds() / (365 * 24 * 3600)  # 年単位
                features[5] = min(time_progress, 1.0)
            
            # 作業負荷分散
            if env_state.get('assignments'):
                workloads = [len(tasks) for tasks in env_state['assignments'].values()]
                if workloads:
                    features[6] = np.std(workloads) / (np.mean(workloads) + 1e-6)  # 変動係数
            
            # システム効率性指標
            if len(env_state.get('completed_tasks', [])) > 0 and total_assignments > 0:
                features[7] = len(env_state['completed_tasks']) / (total_assignments + len(env_state['completed_tasks']))
            
            # 残りは将来の拡張用
            
        except Exception as e:
            pass
        
        return features
    
    def _extract_temporal_features(self, agent_id: str, env_state: Dict) -> np.ndarray:
        """時系列特徴の抽出"""
        features = np.zeros(self.obs_components['temporal_features'])
        
        try:
            # 履歴から時系列パターンを抽出
            if agent_id in self.feature_history:
                history = self.feature_history[agent_id]
                
                if len(history) >= 2:
                    # 最近の変化率
                    recent_change = np.mean(np.abs(history[-1] - history[-2]))
                    features[0] = min(recent_change, 1.0)
                
                if len(history) >= 3:
                    # トレンド
                    trend = np.mean(history[-1] - history[-3])
                    features[1] = np.clip(trend, -1.0, 1.0)
                
                # 安定性
                if len(history) >= self.history_length:
                    stability = 1.0 / (1.0 + np.std([np.mean(obs) for obs in history]))
                    features[2] = min(stability, 1.0)
            
            # 活動パターン
            recent_actions = env_state.get('dev_action_history', {}).get(agent_id, [])
            if recent_actions:
                # 最近の活動頻度
                features[3] = min(len(recent_actions) / 10.0, 1.0)
                
                # 活動の規則性（簡略化）
                features[4] = 0.5  # 実装簡略化
            
            # 時刻特徴
            current_time = env_state.get('current_time')
            if current_time:
                # 時間帯（0-1の周期）
                hour_of_day = current_time.hour / 24.0
                features[5] = hour_of_day
                
                # 曜日（0-1の周期）
                day_of_week = current_time.weekday() / 7.0
                features[6] = day_of_week
            
            # 季節性（簡略化）
            features[7] = 0.5
            
        except Exception as e:
            pass
        
        return features
    
    def _extract_compressed_graph_embeddings(self, agent_id: str, 
                                           env_state: Dict) -> np.ndarray:
        """圧縮されたグラフ埋め込みの抽出"""
        features = np.zeros(self.obs_components['graph_embeddings'])
        
        try:
            # GNN埋め込みが利用可能な場合
            if hasattr(env_state, 'gnn_embeddings') and env_state.gnn_embeddings is not None:
                # 開発者固有の埋め込みを取得
                if agent_id in env_state.gnn_embeddings:
                    raw_embedding = env_state.gnn_embeddings[agent_id]
                    
                    # 次元圧縮（PCAまたは単純な切り詰め）
                    if len(raw_embedding) >= self.obs_components['graph_embeddings']:
                        features = raw_embedding[:self.obs_components['graph_embeddings']]
                    else:
                        features[:len(raw_embedding)] = raw_embedding
            else:
                # フォールバック: 開発者プロファイルベースの疑似埋め込み
                features = self._generate_pseudo_graph_embedding(agent_id, env_state)
                
        except Exception as e:
            # エラー時は疑似埋め込みを生成
            features = self._generate_pseudo_graph_embedding(agent_id, env_state)
        
        return features
    
    def _generate_pseudo_graph_embedding(self, agent_id: str, 
                                       env_state: Dict) -> np.ndarray:
        """疑似グラフ埋め込みの生成"""
        features = np.zeros(self.obs_components['graph_embeddings'])
        
        try:
            developer = env_state['developers'][agent_id]
            profile = developer['profile']
            
            # プロファイル情報から疑似埋め込みを生成
            features[0] = profile.get('total_commits', 0) / 1000.0
            features[1] = profile.get('rank', 5000) / 5000.0
            features[2] = profile.get('followers', 0) / 100.0
            features[3] = profile.get('python_commits', 0) / 500.0
            features[4] = profile.get('javascript_commits', 0) / 300.0
            
            # 残りはランダムノイズ（一貫性のため、agent_idベースのシード使用）
            np.random.seed(hash(agent_id) % 2**32)
            features[5:] = np.random.normal(0, 0.1, len(features) - 5)
            
        except Exception as e:
            # 完全なフォールバック
            np.random.seed(hash(agent_id) % 2**32)
            features = np.random.normal(0, 0.1, self.obs_components['graph_embeddings'])
        
        return features
    
    def _normalize_observation(self, observation: np.ndarray, agent_id: str) -> np.ndarray:
        """観測の正規化"""
        try:
            # 初回は正規化せずに返す
            if not self.initialized:
                self.initialized = True
                return observation
            
            # 各コンポーネントごとに正規化
            normalized_obs = observation.copy()
            start_idx = 0
            
            for component, dim in self.obs_components.items():
                end_idx = start_idx + dim
                component_data = observation[start_idx:end_idx].reshape(1, -1)
                
                # 正規化器が学習済みの場合のみ適用
                try:
                    normalized_component = self.scalers[component].transform(component_data)
                    normalized_obs[start_idx:end_idx] = normalized_component.flatten()
                except:
                    # 学習データが不足している場合はそのまま使用
                    pass
                
                start_idx = end_idx
            
            # 全体のクリッピング
            normalized_obs = np.clip(normalized_obs, -3.0, 3.0)
            
            return normalized_obs
            
        except Exception as e:
            # エラー時は元の観測を返す
            return observation
    
    def _update_feature_history(self, agent_id: str, observation: np.ndarray):
        """特徴量履歴の更新"""
        if agent_id not in self.feature_history:
            self.feature_history[agent_id] = deque(maxlen=self.history_length)
        
        self.feature_history[agent_id].append(observation.copy())
    
    def update_normalizers(self, observations: Dict[str, np.ndarray]):
        """正規化器の更新（バッチ学習）"""
        try:
            # 各エージェントの観測を収集
            all_observations = np.array(list(observations.values()))
            
            if len(all_observations) > 0:
                start_idx = 0
                for component, dim in self.obs_components.items():
                    end_idx = start_idx + dim
                    component_data = all_observations[:, start_idx:end_idx]
                    
                    # 正規化器を部分的に更新
                    self.scalers[component].partial_fit(component_data)
                    
                    start_idx = end_idx
                    
        except Exception as e:
            pass  # エラーは無視
    
    # ヘルパーメソッド
    def _calculate_task_priority(self, task: Dict) -> float:
        """タスクの優先度計算"""
        priority = 0.5  # デフォルト
        
        labels = task.get('labels', [])
        for label in labels:
            label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
            if 'critical' in label_name or 'urgent' in label_name:
                priority = 1.0
            elif 'high' in label_name:
                priority = 0.8
            elif 'low' in label_name:
                priority = 0.2
        
        return priority
    
    def _categorize_task(self, task: Dict) -> str:
        """タスクのカテゴリ分類"""
        labels = task.get('labels', [])
        title = task.get('title', '').lower()
        
        for label in labels:
            label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
            if 'bug' in label_name:
                return 'bug'
            elif 'enhancement' in label_name or 'feature' in label_name:
                return 'feature'
            elif 'documentation' in label_name:
                return 'doc'
        
        if 'bug' in title or 'fix' in title:
            return 'bug'
        elif 'add' in title or 'implement' in title:
            return 'feature'
        elif 'doc' in title:
            return 'doc'
        else:
            return 'other'
    
    def _detect_task_language(self, task: Dict) -> str:
        """タスクの言語検出"""
        title = task.get('title', '').lower()
        body = task.get('body', '').lower()
        content = f"{title} {body}"
        
        if 'python' in content or '.py' in content:
            return 'python'
        elif 'javascript' in content or '.js' in content or 'react' in content:
            return 'javascript'
        else:
            return 'other'
    
    def _calculate_task_urgency(self, task: Dict) -> float:
        """タスクの緊急度計算"""
        urgency = 0.5
        
        # コメント数ベース
        comments = task.get('comments_count', 0)
        urgency += min(comments / 20.0, 0.3)
        
        # ラベルベース
        labels = task.get('labels', [])
        for label in labels:
            label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
            if 'urgent' in label_name or 'critical' in label_name:
                urgency += 0.4
        
        return min(urgency, 1.0)
    
    def _calculate_collaboration_requirement(self, task: Dict) -> float:
        """協力要求度計算"""
        # 既存の参加者数
        participants = len(task.get('assignees', []))
        comments = task.get('comments_count', 0)
        
        collab_score = min((participants + comments / 5.0) / 10.0, 1.0)
        return collab_score
    
    def _calculate_learning_opportunity(self, task: Dict) -> float:
        """学習機会度計算"""
        # 複雑度ベース
        complexity = len(task.get('labels', [])) * 0.2 + min(task.get('comments_count', 0) / 10.0, 0.8)
        return min(complexity, 1.0)