import os
from collections import defaultdict
from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from kazoo.envs.task import Task


class OSSSimpleEnv(gym.Env):
    """
    OSS開発プロセスをシミュレートする、Gymnasium互換の環境。
    時間の進行、タスクの割り当て、開発者の動的な状態を管理する。
    """

    def __init__(
        self, config, backlog, dev_profiles, reward_weights_path=None
    ):
        super().__init__()

        self.config = config
        self.initial_backlog = backlog
        self.dev_profiles = dev_profiles

        # === エージェント(開発者)とIDリストの作成 ===
        self.num_developers = self.config.get("num_developers", len(self.dev_profiles))
        self.developers = self._create_developers()
        self.agent_ids = list(self.developers.keys())

        # ▼▼▼【ここからが修正・拡張箇所】▼▼▼

        # --- 状態変数の初期化 ---
        # Taskオブジェクトのリストを生成
        self.backlog = [Task.from_dict(t) for t in self.initial_backlog]
        self.tasks_in_progress = {}
        self.completed_tasks = []

        # --- 時間関連の状態 ---
        # Noneではない有効な日付だけをリストアップ
        valid_dates = [t.created_at for t in self.backlog if t.created_at is not None]
        # 有効な日付がある場合のみmin()を呼び出し、なければ現在時刻を使う
        self.start_time = min(valid_dates) if valid_dates else datetime.now()
        self.current_time = self.start_time
        self.time_step = timedelta(
            hours=self.config.env.simulation.get("time_step_hours", 8)
        )

        # --- 特徴量計算のための動的状態 ---
        self.dev_action_history = defaultdict(list)
        self.activity_window = timedelta(
            days=self.config.features.get("recent_activity_window_days", 30)
        )
        self.assignments = defaultdict(set)

        # ▲▲▲【ここまでが修正・拡張箇所】▲▲▲

        # === 行動空間・観測空間の定義 ===
        # NOTE: backlogが動的に変わるため、行動空間は本来固定長にすべき。
        # ここでは簡単のため、初期バックログサイズで定義。
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(
                    len(self.initial_backlog) + 1
                )  # +1 for NO_OP
                for agent_id in self.agent_ids
            }
        )
        # 観測空間も同様に、本来は動的に変わらない固定長の特徴量にすべき。
        self.observation_space = spaces.Dict(
            {
                agent_id: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(self.initial_backlog) * 3,),  # 仮のshape
                    dtype=np.float32,
                )
                for agent_id in self.agent_ids
            }
        )

        # === IRL報酬の重み読み込み ===
        self.use_learned_reward = False
        if reward_weights_path and os.path.exists(reward_weights_path):
            self.reward_weights = np.load(reward_weights_path)
            self.use_learned_reward = True
            print(
                f"[OSSSimpleEnv] Loaded learned reward weights from {reward_weights_path}"
            )
        else:
            print("[OSSSimpleEnv] Using default hard-coded reward.")

    def _create_developers(self):
        """開発者プロファイルから開発者辞書を作成する。"""
        developers = {}
        for i, (dev_id, profile) in enumerate(self.dev_profiles.items()):
            if i >= self.num_developers:
                break
            developers[dev_id] = {
                "name": dev_id,
                "profile": profile,
            }
        return developers

    def _calculate_reward(self, agent_id, task, action_type="COMPLETE"):
        """報酬を計算する。単純な例として、タスク完了時に+1。"""
        # ここにIRLの報酬計算ロジックを統合することも可能
        if action_type == "COMPLETE":
            return 1.0
        return 0.0

    def step(self, actions):
        """行動を受け取り、環境を1ステップ進める。"""
        self.current_time += self.time_step
        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        # --- 1. 各エージェントの行動を処理 ---
        for agent_id, action_val in actions.items():
            # action_val は、バックログのタスクインデックス。NO_OPも考慮。
            NO_OP_ACTION = len(self.initial_backlog)
            if action_val == NO_OP_ACTION or action_val >= len(self.backlog):
                continue  # 何もしない

            developer = self.developers[agent_id]
            selected_task = self.backlog.pop(action_val)  # バックログから削除

            # 動的状態の更新
            self.assignments[agent_id].add(selected_task.id)
            selected_task.assigned_to = agent_id
            self.dev_action_history[agent_id].append(self.current_time)
            self.tasks_in_progress[selected_task.id] = selected_task
            selected_task.status = "in_progress"

            print(
                f"Time {self.current_time.date()}: {agent_id} started {selected_task.name}"
            )

        # --- 2. 時間経過によるタスクの進行と完了をシミュレート ---
        completed_this_step = []
        for task in list(self.tasks_in_progress.values()):
            # 仮のロジック：各ステップで一定の確率でタスクが完了する
            if np.random.rand() < 0.1:
                task.status = "done"
                completed_this_step.append(task)

                completing_agent = getattr(task, "assigned_to", None)
                if completing_agent:
                    if task.id in self.assignments[completing_agent]:
                        self.assignments[completing_agent].remove(task.id)
                    rewards[completing_agent] += self._calculate_reward(
                        completing_agent, task, "COMPLETE"
                    )

                print(
                    f"Time {self.current_time.date()}: {task.name} completed by {completing_agent}!"
                )

        # 完了したタスクを進行中リストから削除
        for task in completed_this_step:
            del self.tasks_in_progress[task.id]
            self.completed_tasks.append(task)

        # --- 3. 古い活動履歴をクリーンアップ ---
        for dev_name in self.dev_action_history:
            self.dev_action_history[dev_name] = [
                ts
                for ts in self.dev_action_history[dev_name]
                if self.current_time - ts < self.activity_window
            ]

        # --- 4. gymnasiumのルールに従った戻り値を準備 ---
        observations = self._get_observations()
        is_done = not self.backlog and not self.tasks_in_progress
        terminateds = {agent_id: is_done for agent_id in self.agent_ids}
        max_steps = self.config.env.simulation.get("max_days", 365) * (
            24 // self.config.env.simulation.get("time_step_hours", 8)
        )
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
        """全エージェントの観測を辞書として返す (RLエージェント用)"""
        # この観測はRLエージェントの方策ネットワーク用。
        # IRLの特徴量抽出とは目的が異なる場合があるため、分離して考えるのが良い。
        # ここでは単純なタスクの状態リストを返す。
        task_states = []
        initial_task_ids = {t["id"] for t in self.initial_backlog}

        for task_id in initial_task_ids:
            status_val = 0  # 0: todo
            if task_id in self.tasks_in_progress:
                status_val = 1  # 1: in_progress
            elif any(ct.id == task_id for ct in self.completed_tasks):
                status_val = 2  # 2: done

            # complexity, deadline は固定値と仮定
            task_states.extend([status_val, 0, 0])

        obs_vector = np.array(task_states, dtype=np.float32)
        return {agent_id: obs_vector for agent_id in self.agent_ids}

    def _get_infos(self):
        """特徴量抽出器に渡すための、より完全な状態をinfoに含める。"""
        full_state_info = {
            "current_time": self.current_time,
            "assignments": self.assignments,
            "dev_action_history": self.dev_action_history,
            "backlog": self.backlog,
            "developers": self.developers,
            "tasks_in_progress": self.tasks_in_progress,
        }
        return {agent_id: full_state_info for agent_id in self.agent_ids}
