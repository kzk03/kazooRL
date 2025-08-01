#!/usr/bin/env python3
"""
改良されたOSS環境
PPO 0%精度問題を解決するための統合改良版
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from kazoo.envs.action_space_reducer import HierarchicalActionSpace
from kazoo.envs.improved_reward_system import ImprovedRewardSystem
from kazoo.envs.observation_processor import ObservationProcessor
from kazoo.envs.task import Task


class ImprovedOSSEnv(gym.Env):
    """改良されたOSS開発環境"""
    
    def __init__(self, config, backlog, dev_profiles, reward_weights_path=None):
        super().__init__()
        
        self.config = config
        self.initial_backlog = backlog
        self.dev_profiles = dev_profiles
        
        # 開発者数の制限
        self.num_developers = min(
            self.config.get("num_developers", len(self.dev_profiles)),
            len(self.dev_profiles)
        )
        self.developers = self._create_developers()
        self.agent_ids = list(self.developers.keys())
        
        # タスクの初期化
        self.backlog = [Task.from_dict(t) for t in self.initial_backlog]
        
        # 時間設定
        valid_dates = [t.created_at for t in self.backlog if t.created_at is not None]
        self.start_time = min(valid_dates) if valid_dates else datetime.now()
        self.current_time = self.start_time
        self.time_step = timedelta(
            hours=self.config.env.simulation.get("time_step_hours", 8)
        )
        
        # 改良されたコンポーネントの初期化
        self.improved_reward_system = ImprovedRewardSystem(config)
        self.hierarchical_action = HierarchicalActionSpace(config)
        self.observation_processor = ObservationProcessor(config)
        
        # 行動空間の設定（動的）
        max_action_size = config.get('max_action_candidates', 15) + 1  # +1 for NO_OP
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(max_action_size)
            for agent_id in self.agent_ids
        })
        
        # 観測空間の設定（固定次元）
        obs_dim = config.get('processed_feature_dim', 64)
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            for agent_id in self.agent_ids
        })
        
        # 現在の行動マッピング
        self.current_action_mappings = {}
        
        # 統計情報
        self.episode_stats = {
            'total_rewards': defaultdict(list),
            'task_completions': 0,
            'action_distributions': defaultdict(list),
            'reward_components': defaultdict(list)
        }
        
        # 初期化
        self.reset()
    
    def _create_developers(self):
        """開発者の作成"""
        developers = {}
        for i, (dev_id, profile) in enumerate(self.dev_profiles.items()):
            if i >= self.num_developers:
                break
            developers[dev_id] = {"name": dev_id, "profile": profile}
        return developers
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """環境のステップ実行"""
        self.current_time += self.time_step
        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        reward_breakdowns = {agent_id: {} for agent_id in self.agent_ids}
        
        # 各エージェントの行動処理
        for agent_id, action_idx in actions.items():
            if agent_id not in self.current_action_mappings:
                continue
            
            action_space = self.current_action_mappings[agent_id]
            
            # 行動の実行
            if 0 <= action_idx < len(action_space):
                selected_developer = action_space[action_idx]
                
                if selected_developer != 'NO_OP' and self.backlog:
                    # タスクを選択（最初のタスクを簡単化のため選択）
                    selected_task = self.backlog.pop(0)
                    
                    # タスクの割り当て
                    self.assignments[selected_developer].add(selected_task.id)
                    selected_task.assigned_to = selected_developer
                    selected_task.status = "in_progress"
                    self.tasks_in_progress[selected_task.id] = selected_task
                    
                    # 報酬計算
                    env_state = self._get_env_state()
                    reward, reward_breakdown = self.improved_reward_system.calculate_reward(
                        selected_developer, selected_task.__dict__, "START", env_state
                    )
                    
                    rewards[agent_id] = reward
                    reward_breakdowns[agent_id] = reward_breakdown
                    
                    print(f"Time {self.current_time.date()}: {selected_developer} started {selected_task.title}")
        
        # タスク完了処理
        completed_tasks = self._process_task_completion()
        
        # 完了報酬の追加
        for task in completed_tasks:
            if hasattr(task, 'assigned_to') and task.assigned_to:
                env_state = self._get_env_state()
                completion_reward, completion_breakdown = self.improved_reward_system.calculate_reward(
                    task.assigned_to, task.__dict__, "COMPLETE", env_state
                )
                
                if task.assigned_to in rewards:
                    rewards[task.assigned_to] += completion_reward
                    # 報酬内訳の統合
                    for key, value in completion_breakdown.items():
                        if key in reward_breakdowns[task.assigned_to]:
                            reward_breakdowns[task.assigned_to][key] += value
                        else:
                            reward_breakdowns[task.assigned_to][key] = value
        
        # 観測の更新
        observations = self._get_observations()
        
        # 終了条件
        is_done = not self.backlog and not self.tasks_in_progress
        terminateds = {agent_id: is_done for agent_id in self.agent_ids}
        truncateds = {agent_id: False for agent_id in self.agent_ids}  # 簡単化
        
        # 情報の更新
        infos = self._get_infos(reward_breakdowns)
        
        # 統計情報の更新
        self._update_stats(rewards, reward_breakdowns, actions)
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _process_task_completion(self) -> List[Task]:
        """タスク完了処理（改良版）"""
        completed_tasks = []
        
        for task in list(self.tasks_in_progress.values()):
            # より現実的な完了確率計算
            completion_prob = self._calculate_completion_probability(task)
            
            if np.random.rand() < completion_prob:
                task.status = "done"
                completed_tasks.append(task)
                
                # 割り当てから削除
                if hasattr(task, 'assigned_to') and task.assigned_to in self.assignments:
                    if task.id in self.assignments[task.assigned_to]:
                        self.assignments[task.assigned_to].remove(task.id)
                
                print(f"Time {self.current_time.date()}: {task.title} completed!")
        
        # 完了タスクを進行中から削除
        for task in completed_tasks:
            if task.id in self.tasks_in_progress:
                del self.tasks_in_progress[task.id]
            self.completed_tasks.append(task)
        
        self.episode_stats['task_completions'] += len(completed_tasks)
        
        return completed_tasks
    
    def _calculate_completion_probability(self, task: Task) -> float:
        """タスク完了確率の計算"""
        base_prob = 0.1  # ベース確率
        
        if not hasattr(task, 'assigned_to') or not task.assigned_to:
            return base_prob
        
        # 開発者の能力を考慮
        developer = self.developers.get(task.assigned_to)
        if developer:
            profile = developer['profile']
            
            # ランクによる調整
            rank = profile.get('rank', 5000)
            rank_factor = max(0.5, 2.0 - rank / 2500.0)  # ランクが高いほど完了確率上昇
            
            # 経験による調整
            total_commits = profile.get('total_commits', 0)
            experience_factor = min(2.0, 1.0 + total_commits / 1000.0)
            
            # 現在の作業負荷による調整
            current_workload = len(self.assignments.get(task.assigned_to, set()))
            workload_factor = max(0.5, 1.5 - current_workload / 10.0)
            
            # 総合確率
            adjusted_prob = base_prob * rank_factor * experience_factor * workload_factor
            return min(0.5, adjusted_prob)  # 最大50%
        
        return base_prob
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """観測の取得"""
        observations = {}
        env_state = self._get_env_state()
        
        for agent_id in self.agent_ids:
            # 行動空間の更新
            if self.backlog and self.backlog[0] is not None:  # タスクがある場合のみ
                current_task = self.backlog[0]
                task_dict = current_task.__dict__ if hasattr(current_task, '__dict__') else current_task
                action_space, metadata = self.hierarchical_action.get_action_space_for_task(
                    task_dict, env_state
                )
                self.current_action_mappings[agent_id] = action_space
            else:
                self.current_action_mappings[agent_id] = ['NO_OP']
            
            # 生の観測データ
            raw_obs = {
                'simple_obs': self._get_simple_obs(),
                'gnn_embeddings': np.random.normal(0, 0.1, 16)  # 簡易GNN埋め込み
            }
            
            # 観測の処理
            processed_obs = self.observation_processor.process_observation(
                raw_obs, env_state, agent_id
            )
            
            observations[agent_id] = processed_obs
        
        return observations
    
    def _get_simple_obs(self) -> np.ndarray:
        """シンプルな観測データ"""
        obs = []
        
        # バックログ情報
        obs.append(len(self.backlog))
        obs.append(len(self.tasks_in_progress))
        obs.append(len(self.completed_tasks))
        
        # 時間情報
        obs.append(self.current_time.hour / 24.0)
        obs.append(self.current_time.weekday() / 7.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_env_state(self) -> Dict:
        """環境状態の取得"""
        return {
            'developers': self.developers,
            'assignments': self.assignments,
            'current_time': self.current_time,
            'backlog': self.backlog,
            'tasks_in_progress': self.tasks_in_progress,
            'completed_tasks': self.completed_tasks
        }
    
    def _get_infos(self, reward_breakdowns: Dict) -> Dict:
        """情報の取得"""
        base_info = {
            'current_time': self.current_time,
            'backlog_size': len(self.backlog),
            'tasks_in_progress': len(self.tasks_in_progress),
            'completed_tasks': len(self.completed_tasks),
            'episode_stats': self.episode_stats
        }
        
        return {
            agent_id: {
                **base_info,
                'reward_breakdown': reward_breakdowns.get(agent_id, {}),
                'action_space_size': len(self.current_action_mappings.get(agent_id, []))
            }
            for agent_id in self.agent_ids
        }
    
    def _update_stats(self, rewards: Dict, reward_breakdowns: Dict, actions: Dict):
        """統計情報の更新"""
        for agent_id, reward in rewards.items():
            self.episode_stats['total_rewards'][agent_id].append(reward)
            
            # 行動分布
            action = actions.get(agent_id, -1)
            self.episode_stats['action_distributions'][agent_id].append(action)
            
            # 報酬コンポーネント
            breakdown = reward_breakdowns.get(agent_id, {})
            for component, value in breakdown.items():
                self.episode_stats['reward_components'][f"{agent_id}_{component}"].append(value)
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """環境のリセット"""
        super().reset(seed=seed)
        
        # 状態の初期化
        self.current_time = self.start_time
        self.backlog = [Task.from_dict(t) for t in self.initial_backlog]
        self.tasks_in_progress = {}
        self.completed_tasks = []
        self.assignments = defaultdict(set)
        self.current_action_mappings = {}
        
        # 統計情報のリセット
        self.episode_stats = {
            'total_rewards': defaultdict(list),
            'task_completions': 0,
            'action_distributions': defaultdict(list),
            'reward_components': defaultdict(list)
        }
        
        # 初期観測
        observations = self._get_observations()
        infos = self._get_infos({})
        
        return observations, infos