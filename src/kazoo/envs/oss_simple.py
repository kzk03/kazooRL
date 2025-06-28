import os
from collections import defaultdict
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium import spaces

from kazoo.envs.task import Task


class OSSSimpleEnv(gym.Env):
    """
    OSS開発プロセスをシミュレートする、Gymnasium互換の環境。
    """

    def __init__(self, config, backlog, dev_profiles, reward_weights_path=None):
        super().__init__()

        self.config = config
        self.initial_backlog = backlog
        self.dev_profiles = dev_profiles

        self.num_developers = self.config.get("num_developers", len(self.dev_profiles))
        self.developers = self._create_developers()
        self.agent_ids = list(self.developers.keys())

        self.backlog = [Task.from_dict(t) for t in self.initial_backlog]

        valid_dates = [t.created_at for t in self.backlog if t.created_at is not None]
        self.start_time = min(valid_dates) if valid_dates else datetime.now()
        self.current_time = self.start_time

        ##GNNモデルとグラフデーのimport
        gnn_config = self.config.env.get("gnn", {})
        self.gnn_model = None
        if gnn_config.get("model_path") and os.path.exists(gnn_config["model_path"]):
            try:
                # GNNモデルの初期化と読み込み
                from kazoo.gnn.gnn_model import GNNModel
                self.gnn_model = GNNModel(
                    in_channels_dict={"dev": 8, "task": 9}, 
                    out_channels=32
                )
                self.gnn_model.load_state_dict(torch.load(gnn_config["model_path"], weights_only=True))
                self.gnn_model.eval()
                print(f"[OSSSimpleEnv] Successfully loaded GNN model from {gnn_config['model_path']}")
            except Exception as e:
                print(f"[OSSSimpleEnv] Failed to load GNN model: {e}")
                self.gnn_model = None

        # PyTorch Geometricのデータ形式を想定
        self.graph_data = None
        if gnn_config.get("graph_data_path") and os.path.exists(
            gnn_config.graph_data_path
        ):
            try:
                self.graph_data = torch.load(gnn_config.graph_data_path, weights_only=False)
                print(
                    f"[OSSSimpleEnv] Loaded graph data from {gnn_config.graph_data_path}"
                )
            except Exception as e:
                print(
                    f"Warning: Failed to load graph data from {gnn_config.graph_data_path}: {e}"
                )
                self.graph_data = None

        rl_config = self.config.get("rl", {})  # rlセクションがない場合も考慮
        self.time_step = timedelta(
            hours=self.config.env.simulation.get("time_step_hours", 8)
        )
        self.activity_window = timedelta(
            days=self.config.features.get("recent_activity_window_days", 30)
        )

        self.reset()

        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(len(self.initial_backlog) + 1)
                for agent_id in self.agent_ids
            }
        )
        # 観測空間を単純なBox空間に変更（PPOAgentとの互換性のため）
        simple_obs_shape = (len(self.initial_backlog) * 3,)
        gnn_pooled_dim = 64  # GNN埋め込みをプールして固定サイズに
        total_obs_dim = simple_obs_shape[0] + gnn_pooled_dim
        
        self.observation_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(total_obs_dim,),
                    dtype=np.float32,
                )
                for agent_id in self.agent_ids
            }
        )

        self.reward_weights = None
        if reward_weights_path and os.path.exists(reward_weights_path):
            self.reward_weights = np.load(reward_weights_path)
            print(
                f"[OSSSimpleEnv] Loaded learned reward weights from {reward_weights_path}"
            )
        else:
            print("[OSSSimpleEnv] Using default hard-coded reward.")

    def _create_developers(self):
        developers = {}
        for i, (dev_id, profile) in enumerate(self.dev_profiles.items()):
            if i >= self.num_developers:
                break
            developers[dev_id] = {"name": dev_id, "profile": profile}
        return developers

    def _calculate_reward(self, agent_id, task, action_type="COMPLETE"):
        # IRLの重みを使う場合は、ここで特徴量を計算して報酬を返す
        if self.reward_weights is not None:
            try:
                # FeatureExtractorを使って特徴量を計算
                from kazoo.features.feature_extractor import FeatureExtractor
                feature_extractor = FeatureExtractor(self.config)
                
                # 開発者情報を取得
                developer = self.developers[agent_id]
                
                # 特徴量を計算
                features = feature_extractor.get_features(task, developer, self)
                
                # IRLで学習した重みで報酬を計算
                reward = float(np.dot(self.reward_weights, features))
                
                print(f"[IRL Reward] {agent_id} -> {task.title}: {reward:.3f}")
                return reward
                
            except Exception as e:
                print(f"Warning: Failed to calculate IRL reward: {e}")
                # フォールバックとしてデフォルト報酬を使用
                pass

        # デフォルト報酬
        if action_type == "COMPLETE":
            return 1.0
        return 0.0

    def step(self, actions):
        self.current_time += self.time_step
        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        for agent_id, action_val in actions.items():
            NO_OP_ACTION = len(self.initial_backlog)
            if action_val == NO_OP_ACTION or action_val >= len(self.backlog):
                continue

            developer = self.developers[agent_id]
            selected_task = self.backlog.pop(action_val)

            self.assignments[agent_id].add(selected_task.id)
            selected_task.assigned_to = agent_id
            self.dev_action_history[agent_id].append(self.current_time)
            self.tasks_in_progress[selected_task.id] = selected_task
            selected_task.status = "in_progress"
            print(
                f"Time {self.current_time.date()}: {agent_id} started {selected_task.title}"
            )

        completed_this_step = []
        for task in list(self.tasks_in_progress.values()):
            if np.random.rand() < 0.1:
                task.status = "done"
                completed_this_step.append(task)

                completing_agent = getattr(task, "assigned_to", None)
                if completing_agent and completing_agent in self.assignments:
                    if task.id in self.assignments[completing_agent]:
                        self.assignments[completing_agent].remove(task.id)
                    rewards[completing_agent] += self._calculate_reward(
                        completing_agent, task, "COMPLETE"
                    )
                print(
                    f"Time {self.current_time.date()}: {task.title} completed by {completing_agent}!"
                )

        for task in completed_this_step:
            if task.id in self.tasks_in_progress:
                del self.tasks_in_progress[task.id]
            self.completed_tasks.append(task)

        for dev_name in self.dev_action_history:
            self.dev_action_history[dev_name] = [
                ts
                for ts in self.dev_action_history[dev_name]
                if self.current_time - ts < self.activity_window
            ]

        observations = self._get_observations()
        is_done = not self.backlog and not self.tasks_in_progress
        terminateds = {agent_id: is_done for agent_id in self.agent_ids}

        rl_config = self.config.get("rl", {})
        max_steps = rl_config.get("max_steps", 365 * 3)  # 仮
        is_truncated = (
            self.current_time - self.start_time
        ).total_seconds() / self.time_step.total_seconds() >= max_steps
        truncateds = {agent_id: is_truncated for agent_id in self.agent_ids}
        infos = self._get_infos()

        return observations, rewards, terminateds, truncateds, infos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_time = self.start_time
        self.dev_action_history = defaultdict(list)
        self.assignments = defaultdict(set)

        self.backlog = [Task.from_dict(t) for t in self.initial_backlog]
        self.tasks_in_progress = {}
        self.completed_tasks = []

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def _get_observations(self):
        task_states = []
        initial_task_ids_ordered = [t["id"] for t in self.initial_backlog]

        for task_id in initial_task_ids_ordered:
            status_val = 0
            if task_id in self.tasks_in_progress:
                status_val = 1
            elif any(ct.id == task_id for ct in self.completed_tasks):
                status_val = 2

            # complexity, deadline は固定値と仮定。実際のデータから取得するのが望ましい。
            task_states.extend([status_val, 0, 0])
        obs_vector = np.array(task_states, dtype=np.float32)

        # --- GNNによる特徴量を計算 ---
        # graph_dataがNoneの場合のエラーハンドリング
        if self.graph_data is None:
            print("Warning: graph_data is None. Using empty embeddings.")
            gnn_embeddings = np.zeros((0, 64))  # 空の埋め込み
        else:
            # デフォルト値から開始
            gnn_embeddings = np.zeros((self.graph_data.num_nodes, 64))
            
            if self.gnn_model and self.graph_data:
                try:
                    # 現在のタスクステータスなどをグラフデータに反映
                    updated_graph_data = self._update_graph_features(self.graph_data)
                    
                    with torch.no_grad():
                        # GNNのフォワードパスを実行して最新のノード特徴量を取得
                        embeddings_dict = self.gnn_model(
                            updated_graph_data.x_dict, 
                            updated_graph_data.edge_index_dict
                        )
                        
                        # 開発者とタスクの埋め込みを結合
                        dev_embeddings = embeddings_dict["dev"].cpu().numpy()
                        task_embeddings = embeddings_dict["task"].cpu().numpy()
                        
                        # 結合して64次元に調整
                        combined_embeddings = np.concatenate([dev_embeddings, task_embeddings], axis=0)
                        
                        # 必要に応じてサイズ調整
                        if combined_embeddings.shape[0] >= self.graph_data.num_nodes:
                            gnn_embeddings = combined_embeddings[:self.graph_data.num_nodes, :32]
                            # 32次元を64次元にパディング
                            gnn_embeddings = np.pad(gnn_embeddings, ((0, 0), (0, 32)), mode='constant')
                        
                        print(f"[OSSSimpleEnv] Successfully computed GNN embeddings: {gnn_embeddings.shape}")
                        
                except Exception as e:
                    print(f"Warning: Failed to compute GNN embeddings: {e}")
                    gnn_embeddings = np.zeros((self.graph_data.num_nodes, 64))

        # --- 辞書形式で観測をまとめる ---
        observations = {}

        # GNN埋め込みをプール（Global Average Pooling）
        if gnn_embeddings.shape[0] > 0:
            pooled_gnn = np.mean(gnn_embeddings, axis=0)  # (64,)
        else:
            pooled_gnn = np.zeros(64)  # 空の場合はゼロベクトル

        # Simple obsとプールされたGNN埋め込みを結合
        for agent_id in self.agent_ids:
            combined_obs = np.concatenate([obs_vector, pooled_gnn])
            observations[agent_id] = combined_obs.astype(np.float32)
        
        return observations

    def _get_infos(self):
        full_state_info = {
            "current_time": self.current_time,
            "assignments": self.assignments,
            "dev_action_history": self.dev_action_history,
            "backlog": self.backlog,
            "developers": self.developers,
            "tasks_in_progress": self.tasks_in_progress,
        }
        return {agent_id: full_state_info for agent_id in self.agent_ids}

    def _update_graph_features(self, graph_data):
        """
        現在の環境状態に基づいてグラフデータの特徴量を更新
        """
        try:
            # グラフデータをコピー
            updated_data = graph_data.clone()
            
            # タスクの状態を更新
            if hasattr(updated_data, 'x_dict') and 'task' in updated_data.x_dict:
                task_features = updated_data.x_dict['task'].clone()
                
                # 各タスクの状態を更新
                for i, task_dict in enumerate(self.initial_backlog):
                    task_id = task_dict['id']
                    
                    # タスクの現在の状態を取得
                    if task_id in self.tasks_in_progress:
                        status = 1.0  # 進行中
                    elif any(ct.id == task_id for ct in self.completed_tasks):
                        status = 2.0  # 完了
                    else:
                        status = 0.0  # 未着手
                    
                    # 特徴量の最初の次元にステータスを設定
                    if i < task_features.shape[0]:
                        task_features[i, 0] = status
                
                updated_data.x_dict['task'] = task_features
            
            # 開発者の状態を更新
            if hasattr(updated_data, 'x_dict') and 'dev' in updated_data.x_dict:
                dev_features = updated_data.x_dict['dev'].clone()
                
                # 開発者の活動状況を更新
                for i, (dev_id, dev_info) in enumerate(self.developers.items()):
                    if i < dev_features.shape[0]:
                        # 現在の作業負荷
                        workload = len(self.assignments.get(dev_id, set()))
                        dev_features[i, 0] = float(workload)
                        
                        # 最近の活動数
                        recent_activity = len(self.dev_action_history.get(dev_id, []))
                        dev_features[i, 1] = float(recent_activity)
                
                updated_data.x_dict['dev'] = dev_features
            
            return updated_data
            
        except Exception as e:
            print(f"Warning: Failed to update graph features: {e}")
            return graph_data
