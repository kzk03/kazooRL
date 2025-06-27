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
            # GNNモデルの初期化コードは後で実装
            # self.gnn_model = GNN(...) # GNNモデルの初期化コードを追加
            # self.gnn_model.load.state_dict(torch.load(gnn_config.model_path))
            # self.gnn_model.eval()
            print(f"[OSSSimpleEnv] GNN model path found but loading not implemented: {gnn_config['model_path']}")

        # PyTorch Geometricのデータ形式を想定
        self.graph_data = None 
        if gnn_config.get("graph_data_path") and os.path.exists(gnn_config.graph_data_path):
            try:
                self.graph_data = torch.load(gnn_config.graph_data_path)
                print(f"[OSSSimpleEnv] Loaded graph data from {gnn_config.graph_data_path}")
            except Exception as e:
                print(f"Warning: Failed to load graph data from {gnn_config.graph_data_path}: {e}")
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
        # 2. 観測空間(Observation Space)の変更
        # 元の観測空間の定義
        simple_obs_shape = (len(self.initial_backlog) * 3,)
        # GNNの出力特徴量の次元数 (仮に64とする)
        gnn_embedding_dim = 64 
        # graph_dataがNoneの場合は空のグラフを想定
        num_nodes = self.graph_data.num_nodes if self.graph_data is not None else 0
        
        self.observation_space = spaces.Dict({
            agent_id: spaces.Dict({
                "simple_obs": spaces.Box(low=-np.inf, high=np.inf, shape=simple_obs_shape, dtype=np.float32),
                "gnn_embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(num_nodes, gnn_embedding_dim), dtype=np.float32)
            })
            for agent_id in self.agent_ids
        })


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
            # feature_extractorをインスタンス化して使う必要がある
            # ここでは単純な報酬を返す
            pass

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
            gnn_embeddings = np.zeros((self.graph_data.num_nodes, 64))  # デフォルト値
            if self.gnn_model and self.graph_data:
                # 現在のタスクステータスなどをグラフデータに反映させる (ここは別途実装が必要)
                # updated_graph_data = self._update_graph_features(self.graph_data)
                
                with torch.no_grad():
                    # GNNのフォワードパスを実行して最新のノード特徴量を取得
                    # gnn_embeddings = self.gnn_model(updated_graph_data).cpu().numpy()
                    pass  # 実装待ち
        
        # --- 辞書形式で観測をまとめる ---
        observations = {}

        for agent_id in self.agent_ids:
            observations[agent_id] = {
                "simple_obs": obs_vector,
                "gnn_embeddings": gnn_embeddings
            }
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
