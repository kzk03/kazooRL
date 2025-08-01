#!/usr/bin/env python3
"""
改良された報酬システム
スパース報酬問題を解決するための中間報酬を導入
"""

from datetime import datetime
from typing import Any, Dict

import numpy as np


class ImprovedRewardSystem:
    """改良された報酬計算システム"""
    
    def __init__(self, config):
        self.config = config
        self.reward_weights = None
        
        # 報酬の重み配分
        self.reward_components = {
            'task_completion': 1.0,      # タスク完了報酬
            'skill_match': 0.3,          # スキル適合度報酬
            'workload_balance': 0.2,     # 作業負荷バランス報酬
            'collaboration': 0.15,       # コラボレーション報酬
            'learning': 0.1              # 学習機会報酬
        }
    
    def calculate_reward(self, agent_id: str, task: Dict, action_type: str, 
                        env_state: Dict) -> float:
        """
        多面的な報酬計算
        
        Args:
            agent_id: エージェント（開発者）ID
            task: タスク情報
            action_type: 行動タイプ
            env_state: 環境状態
        """
        total_reward = 0.0
        reward_breakdown = {}
        
        # 1. タスク完了報酬（従来の報酬）
        completion_reward = self._calculate_completion_reward(action_type)
        total_reward += self.reward_components['task_completion'] * completion_reward
        reward_breakdown['completion'] = completion_reward
        
        # 2. スキル適合度報酬（新規追加）
        skill_reward = self._calculate_skill_match_reward(agent_id, task, env_state)
        total_reward += self.reward_components['skill_match'] * skill_reward
        reward_breakdown['skill_match'] = skill_reward
        
        # 3. 作業負荷バランス報酬（新規追加）
        workload_reward = self._calculate_workload_reward(agent_id, env_state)
        total_reward += self.reward_components['workload_balance'] * workload_reward
        reward_breakdown['workload'] = workload_reward
        
        # 4. コラボレーション報酬（新規追加）
        collab_reward = self._calculate_collaboration_reward(agent_id, task, env_state)
        total_reward += self.reward_components['collaboration'] * collab_reward
        reward_breakdown['collaboration'] = collab_reward
        
        # 5. 学習機会報酬（新規追加）
        learning_reward = self._calculate_learning_reward(agent_id, task, env_state)
        total_reward += self.reward_components['learning'] * learning_reward
        reward_breakdown['learning'] = learning_reward
        
        # 報酬の正規化とクリッピング
        total_reward = np.clip(total_reward, -2.0, 2.0)
        
        return total_reward, reward_breakdown
    
    def _calculate_completion_reward(self, action_type: str) -> float:
        """タスク完了報酬"""
        if action_type == "COMPLETE":
            return 1.0
        elif action_type == "START":
            return 0.1  # タスク開始時にも小さな報酬
        else:
            return 0.0
    
    def _calculate_skill_match_reward(self, agent_id: str, task: Dict, 
                                    env_state: Dict) -> float:
        """スキル適合度報酬"""
        try:
            developer = env_state['developers'][agent_id]
            dev_profile = developer['profile']
            
            # タスクのラベルから必要スキルを推定
            required_skills = self._extract_required_skills(task)
            
            # 開発者のスキルレベル
            dev_skills = {
                'python': dev_profile.get('python_commits', 0) / 100.0,
                'javascript': dev_profile.get('javascript_commits', 0) / 100.0,
                'debugging': dev_profile.get('bug_fixes', 0) / 50.0,
                'documentation': dev_profile.get('doc_commits', 0) / 20.0,
                'testing': dev_profile.get('test_commits', 0) / 30.0
            }
            
            # スキル適合度を計算
            skill_match = 0.0
            for skill, importance in required_skills.items():
                if skill in dev_skills:
                    skill_match += importance * min(dev_skills[skill], 1.0)
            
            return skill_match
            
        except Exception as e:
            return 0.0
    
    def _calculate_workload_reward(self, agent_id: str, env_state: Dict) -> float:
        """作業負荷バランス報酬"""
        try:
            # 現在の作業負荷
            current_workload = len(env_state['assignments'].get(agent_id, set()))
            
            # 全開発者の平均作業負荷
            all_workloads = [len(tasks) for tasks in env_state['assignments'].values()]
            avg_workload = np.mean(all_workloads) if all_workloads else 0
            
            # 負荷が平均に近いほど高い報酬
            workload_diff = abs(current_workload - avg_workload)
            workload_reward = max(0, 1.0 - workload_diff / 5.0)  # 5タスク差で報酬0
            
            return workload_reward
            
        except Exception as e:
            return 0.0
    
    def _calculate_collaboration_reward(self, agent_id: str, task: Dict, 
                                      env_state: Dict) -> float:
        """コラボレーション報酬"""
        try:
            developer = env_state['developers'][agent_id]
            
            # 過去のコラボレーション履歴
            collab_score = developer['profile'].get('collaboration_score', 0) / 100.0
            
            # タスクの複雑さ（コラボレーションが必要そうか）
            task_complexity = self._estimate_task_complexity(task)
            
            # 複雑なタスクでコラボレーション能力が高い開発者に報酬
            collab_reward = collab_score * task_complexity
            
            return collab_reward
            
        except Exception as e:
            return 0.0
    
    def _calculate_learning_reward(self, agent_id: str, task: Dict, 
                                 env_state: Dict) -> float:
        """学習機会報酬"""
        try:
            developer = env_state['developers'][agent_id]
            dev_profile = developer['profile']
            
            # 開発者の経験レベル
            experience_level = min(dev_profile.get('total_commits', 0) / 1000.0, 1.0)
            
            # タスクの学習価値（新しい技術・領域か）
            learning_value = self._estimate_learning_value(task, dev_profile)
            
            # 経験の浅い開発者が学習価値の高いタスクを取ると報酬
            learning_reward = (1.0 - experience_level) * learning_value
            
            return learning_reward
            
        except Exception as e:
            return 0.0
    
    def _extract_required_skills(self, task: Dict) -> Dict[str, float]:
        """タスクから必要スキルを抽出"""
        required_skills = {}
        
        # ラベルベースのスキル推定
        labels = task.get('labels', [])
        if isinstance(labels, list):
            label_names = [label.get('name', '').lower() if isinstance(label, dict) 
                          else str(label).lower() for label in labels]
            
            for label in label_names:
                if 'bug' in label:
                    required_skills['debugging'] = 0.8
                elif 'enhancement' in label or 'feature' in label:
                    required_skills['python'] = 0.6
                    required_skills['javascript'] = 0.4
                elif 'documentation' in label or 'doc' in label:
                    required_skills['documentation'] = 0.9
                elif 'test' in label:
                    required_skills['testing'] = 0.7
        
        # タイトル・本文からのスキル推定
        title = task.get('title', '').lower()
        body = task.get('body', '').lower()
        text = f"{title} {body}"
        
        if 'python' in text:
            required_skills['python'] = required_skills.get('python', 0) + 0.3
        if 'javascript' in text or 'js' in text:
            required_skills['javascript'] = required_skills.get('javascript', 0) + 0.3
        if 'test' in text:
            required_skills['testing'] = required_skills.get('testing', 0) + 0.2
        
        # 正規化
        for skill in required_skills:
            required_skills[skill] = min(required_skills[skill], 1.0)
        
        return required_skills
    
    def _estimate_task_complexity(self, task: Dict) -> float:
        """タスクの複雑さを推定"""
        complexity = 0.0
        
        # コメント数による複雑さ推定
        comments_count = task.get('comments_count', 0)
        complexity += min(comments_count / 10.0, 0.5)
        
        # タイトル・本文の長さによる推定
        title_length = len(task.get('title', ''))
        body_length = len(task.get('body', ''))
        complexity += min((title_length + body_length) / 1000.0, 0.3)
        
        # ラベル数による推定
        labels_count = len(task.get('labels', []))
        complexity += min(labels_count / 5.0, 0.2)
        
        return min(complexity, 1.0)
    
    def _estimate_learning_value(self, task: Dict, dev_profile: Dict) -> float:
        """タスクの学習価値を推定"""
        learning_value = 0.0
        
        # 新しい技術スタックの学習機会
        title = task.get('title', '').lower()
        body = task.get('body', '').lower()
        text = f"{title} {body}"
        
        # 開発者が経験の少ない領域かチェック
        python_exp = dev_profile.get('python_commits', 0)
        js_exp = dev_profile.get('javascript_commits', 0)
        
        if 'python' in text and python_exp < 50:
            learning_value += 0.4
        if ('javascript' in text or 'js' in text) and js_exp < 50:
            learning_value += 0.4
        if 'docker' in text or 'kubernetes' in text:
            learning_value += 0.3
        if 'machine learning' in text or 'ml' in text:
            learning_value += 0.5
        
        return min(learning_value, 1.0)