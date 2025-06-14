import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kazoo.envs.task import Task
from kazoo.features.feature_extractor import get_features

# from kazoo.envs.developer import Developer # Developerクラスがある場合はインポート


# OSSSimpleEnv クラスの定義
class OSSSimpleEnv(gym.Env):
    def __init__(self, config, backlog, dev_profiles, reward_weights_path=None):
        super().__init__()  # gym.Envの初期化を呼び出すことが推奨されます

        self.config = config
        self.dev_profiles = dev_profiles
        self.initial_backlog = backlog

        # ===エージェント(開発者)とIDリストの作成 ===
        self.num_developers = self.config.get("num_developers", 5)
        self.developers = self._create_developers()
        self.agent_ids = list(self.developers.keys())

        # ===行動空間・観測空間の正しい定義 ===
        # agent_ids を使って、各エージェントの空間を定義します。
        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(len(self.initial_backlog) + 1)
                for agent_id in self.agent_ids
            }
        )
        self.observation_space = spaces.Dict(
            # このshapeは環境の状態表現に合わせて調整が必要です
            {
                agent_id: spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(len(self.initial_backlog) * 3,),
                    dtype=np.float32,
                )
                for agent_id in self.agent_ids
            }
        )

        # ===IRL報酬の重み読み込み処理 ===
        # この部分はご自身のコードをそのまま活かしています。
        self.use_learned_reward = False
        if reward_weights_path and os.path.exists(reward_weights_path):
            self.reward_weights = np.load(reward_weights_path)
            self.use_learned_reward = True
            print(
                f"[OSSSimpleEnv] Loaded learned reward weights from {reward_weights_path}"
            )
        else:
            print("[OSSSimpleEnv] Using default hard-coded reward.")

    """
    __init__メソッドの後 (インデントをクラスに合わせる) に、
    以下のヘルパーメソッドを追加してください。
    """

    def _create_developers(self):
        developers = {}
        # dev_profiles.yaml から開発者の情報を読み込む
        for i, (dev_id, profile) in enumerate(self.dev_profiles.items()):
            if i >= self.num_developers:
                break
            # Developerクラスがあればそれを使うのが望ましいですが、
            # ない場合は辞書として情報を保持します。
            developers[dev_id] = {
                "id": dev_id,
                "skills": profile.get("skills", []),
                "efficiency": profile.get("efficiency", 1.0),
            }
        return developers

    """
    以下は、ご自身で実装された _calculate_reward メソッドです。
    このままお使いください。
    """

    def _calculate_reward(self, agent_id, action_enum, action_details):
        # action_detailsなどの引数は、step関数から渡せるように調整が必要
        if self.use_learned_reward:
            # 特徴量ベクトルを取得し、重みとの内積を報酬として返す
            # self.state は step メソッド内で更新される現在の状態を渡す想定
            features = get_features(self.state, action_enum, action_details, agent_id)
            return np.dot(self.reward_weights, features)
        else:
            # from kazoo.consts.actions import Action # 必要に応じてインポート
            # 以下は、従来のハードコーディングされた報酬計算
            # return 1.0 if action_enum == Action.MERGE_PULL_REQUEST else 0.0
            return 0.0  # 仮のデフォルト報酬

    def step(self, actions):
        """
        行動を受け取り、環境を1ステップ進める。
        戻り値は gymnasium のルールに従う必要がある。
        """
        self.time += 1

        # このステップでの各エージェントの報酬を初期化
        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

        # --- 1. 各エージェントの行動を処理 ---
        for agent_id, action_val in actions.items():
            developer = self.developers[agent_id]

            # action_val は、選択したタスクのインデックスを想定
            # 現状の実装では、エージェントは1つのタスクしか持てないと仮定
            # TODO: developerオブジェクトに現在のタスクを保持する属性が必要
            # if developer.get("current_task") is None and action_val < len(self.backlog):
            if action_val < len(self.backlog):  # 仮の割当ロジック
                # 選択されたタスクを取得
                selected_task = self.backlog[action_val]

                # タスクを進行中に移動
                if selected_task.id not in self.tasks_in_progress:
                    self.tasks_in_progress[selected_task.id] = selected_task
                    selected_task.status = "in_progress"
                    # TODO: 開発者にタスクを割り当てる処理
                    # developer["current_task"] = selected_task.id
                    print(f"Time {self.time}: {agent_id} started {selected_task.name}")

        # --- 2. 時間経過によるタスクの進行と完了をシミュレート ---
        # この部分は簡略化されています。本来は開発者のスキルや効率を考慮します。
        completed_this_step = []
        for task in self.tasks_in_progress.values():
            # 仮のロジック：各ステップで一定の確率でタスクが完了する
            if np.random.rand() < 0.1:  # 10%の確率で完了
                task.status = "done"
                self.completed_tasks.append(task)
                completed_this_step.append(task.id)
                # TODO: 完了させたエージェントに報酬を与える
                # rewards[task.assigned_to] += 1.0
                print(f"Time {self.time}: {task.name} completed!")

        # 完了したタスクを進行中リストから削除
        for task_id in completed_this_step:
            del self.tasks_in_progress[task_id]

        # --- 3. gymnasiumのルールに従った戻り値を準備 ---

        # 次の観測を取得
        observations = self._get_observations()

        # 終了判定
        # ここでは単純に最大ステップに達したかどうかで判定
        is_done = self.time >= self.config.get("max_steps", 1000)
        terminateds = {agent_id: is_done for agent_id in self.agent_ids}
        truncateds = {agent_id: is_done for agent_id in self.agent_ids}

        # 補助情報を取得
        infos = self._get_infos()

        # 戻り値は (観測, 報酬, 終了フラグ(terminated), 打ち切りフラグ(truncated), 情報) の5つ組
        return observations, rewards, terminateds, truncateds, infos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.backlog = [Task.from_dict(t) for t in self.initial_backlog]
        self.tasks_in_progress = {}
        self.completed_tasks = []

        # Developerのリセット処理（もしあれば）
        # for dev in self.developers.values():
        #     dev.reset()

        # ★★★ ここからが修正箇所 ★★★
        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos  # ★★★ 初期観測と情報を返す

    # === 以下のヘルパーメソッドをクラス内に追加してください ===

    def _get_observations(self):
        """全エージェントの観測を辞書として返す"""

        task_states = []
        for task in self.backlog:  # ★★★ ここを self.backlog に修正 ★★★
            # これで task は Task オブジェクトになる
            status_val = 0
            if task.id in self.tasks_in_progress:
                status_val = 1
            elif task.status == "done":  # Taskオブジェクトの属性を参照するように変更
                status_val = 2

            task_states.extend([status_val, task.complexity, task.deadline])

        obs_vector = np.array(task_states, dtype=np.float32)

        return {agent_id: obs_vector for agent_id in self.agent_ids}

    def _get_infos(self):
        """全エージェントの補助情報を辞書として返す"""
        # 現状は空の辞書で問題ありません
        return {agent_id: {} for agent_id in self.agent_ids}
