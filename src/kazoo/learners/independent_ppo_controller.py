import os

import numpy as np
import torch
from gymnasium import spaces
from tqdm import tqdm

from kazoo.learners.ppo_agent import PPOAgent


class RolloutStorage:
    """
    PPOのための経験データを一時的に保存し、リターンとアドバンテージを計算する。
    """

    def __init__(self, num_steps, obs_space, act_space, device):
        self.device = device
        self.num_steps = num_steps

        # 複合観測空間の場合は全体の次元を計算
        if hasattr(obs_space, 'spaces'):  # Dict space
            # 全観測空間の合計次元を計算
            total_dim = 0
            for space in obs_space.spaces.values():
                if hasattr(space, 'shape') and space.shape:
                    total_dim += space.shape[0]
            obs_shape = (total_dim,)
        else:
            obs_shape = obs_space.shape
        
        # ストレージを(ステップ数, 次元)の形で初期化
        self.obs = torch.zeros((num_steps,) + obs_shape).to(device)
        self.actions = torch.zeros(num_steps, 1, dtype=torch.long).to(device)
        self.log_probs = torch.zeros(num_steps, 1).to(device)
        self.rewards = torch.zeros(num_steps, 1).to(device)
        self.dones = torch.zeros(num_steps, 1).to(device)
        self.values = torch.zeros(num_steps, 1).to(device)
        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        """
        ステップごとのデータを追加する。
        全てのデータを、保存先の形状である[1]に合わせる。
        """
        # 複合観測の場合は平坦化して結合
        if isinstance(obs, dict):
            obs_tensor = torch.cat([torch.as_tensor(v, device=self.device).flatten() 
                                  for v in obs.values()], dim=0)
        else:
            obs_tensor = torch.as_tensor(obs, device=self.device)
        
        self.obs[self.step].copy_(obs_tensor)
        self.actions[self.step].copy_(
            torch.as_tensor([action], device=self.device, dtype=torch.long)
        )
        self.log_probs[self.step].copy_(log_prob.view(1))
        self.rewards[self.step].copy_(
            torch.as_tensor([reward], device=self.device, dtype=torch.float32)
        )
        self.dones[self.step].copy_(
            torch.as_tensor([done], device=self.device, dtype=torch.float32)
        )
        self.values[self.step].copy_(value.view(1))
        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, next_value, gamma, gae_lambda):
        """GAE (Generalized Advantage Estimation) を使ってリターンとアドバンテージを計算"""
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            self.advantages[t] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )
        self.returns = self.advantages + self.values


class IndependentPPOController:
    """複数のPPOAgentを管理し、マルチエージェント学習を実行する司令塔"""

    def __init__(self, env, config):
        self.env = env
        self.agent_ids = env.agent_ids
        self.config = config
        self.num_agents = len(self.agent_ids)
        self.device = torch.device(config.get("device", "cpu"))

        try:
            self.rl_config = config.rl
        except Exception:
            raise ValueError("Configuration file must contain an 'rl' section.")

        self.agents = {}
        self.storages = {}
        for agent_id in self.agent_ids:
            obs_space = self.env.observation_space[agent_id]
            act_space = self.env.action_space[agent_id]
            
            # デバッグ情報を追加
            print(f"Agent {agent_id}: obs_space = {obs_space}, act_space = {act_space}")
            
            if obs_space is None:
                raise ValueError(f"Observation space for agent {agent_id} is None")
            if act_space is None:
                raise ValueError(f"Action space for agent {agent_id} is None")
            
            # 辞書形式の観測空間から次元数を計算
            if hasattr(obs_space, 'spaces'):  # Dict space
                total_obs_dim = 0
                for space_name, space in obs_space.spaces.items():
                    if hasattr(space, 'shape'):
                        total_obs_dim += space.shape[0]
                    else:
                        print(f"Warning: Space {space_name} has no shape attribute")
                print(f"Total observation dimension for agent {agent_id}: {total_obs_dim}")
            else:  # Box space
                total_obs_dim = obs_space.shape[0]

            self.agents[agent_id] = PPOAgent(
                obs_dim=total_obs_dim,  # 観測次元数を直接渡す
                act_space=act_space,
                lr=self.rl_config.learning_rate,
                gamma=self.rl_config.gamma,
                epochs=self.rl_config.k_epochs,
                eps_clip=self.rl_config.eps_clip,
                device=self.device,
            )
            self.storages[agent_id] = RolloutStorage(
                self.rl_config.rollout_len, obs_space, act_space, self.device
            )

    def learn(self, total_timesteps):
        print("Starting Multi-Agent PPO Training...")
        obs, info = self.env.reset()
        global_step = 0

        # GNNオンライン学習の設定
        gnn_update_frequency = getattr(self.config.irl, "gnn_update_frequency", 50)
        online_gnn_learning = getattr(self.config.irl, "online_gnn_learning", False)

        print(
            f"GNN Online Learning: {'Enabled' if online_gnn_learning else 'Disabled'}"
        )
        if online_gnn_learning:
            print(f"GNN Update Frequency: Every {gnn_update_frequency} steps")

        num_updates = (
            int(total_timesteps / self.rl_config.rollout_len / self.num_agents)
            if self.num_agents > 0
            else 0
        )

        # PPO学習の進捗バー
        update_progress = tqdm(
            range(1, num_updates + 1),
            desc="🤖 PPO 学習",
            unit="update",
            colour='magenta',
            leave=True
        )

        for update in update_progress:
            # ロールアウト収集の進捗バー
            rollout_progress = tqdm(
                range(self.rl_config.rollout_len),
                desc=f"Update {update:4d}/{num_updates}",
                unit="step",
                leave=False,
                colour='yellow'
            )
            
            for step in rollout_progress:
                global_step += self.num_agents
                actions_dict, log_probs_dict, values_dict = {}, {}, {}

                with torch.no_grad():
                    for agent_id, agent_obs in obs.items():
                        action, log_prob, _, value = self.agents[
                            agent_id
                        ].get_action_and_value(agent_obs)
                        actions_dict[agent_id] = action.item()
                        log_probs_dict[agent_id] = log_prob
                        values_dict[agent_id] = value

                next_obs, rewards, terminateds, truncateds, infos = self.env.step(
                    actions_dict
                )
                dones = {
                    agent_id: terminateds[agent_id] or truncateds[agent_id]
                    for agent_id in self.agent_ids
                }

                for agent_id in self.agent_ids:
                    self.storages[agent_id].add(
                        obs[agent_id],
                        actions_dict[agent_id],
                        log_probs_dict[agent_id],
                        rewards[agent_id],
                        dones[agent_id],
                        values_dict[agent_id],
                    )
                obs = next_obs

                # 現在の平均報酬を計算
                current_rewards = list(rewards.values())
                avg_reward = np.mean(current_rewards) if current_rewards else 0.0
                rollout_progress.set_postfix({
                    "Step": f"{global_step:,}",
                    "Avg_Reward": f"{avg_reward:.3f}"
                })

            with torch.no_grad():
                next_values = {
                    agent_id: self.agents[agent_id].get_action_and_value(obs[agent_id])[
                        3
                    ]
                    for agent_id in self.agent_ids
                }

            for agent_id in self.agent_ids:
                self.storages[agent_id].compute_returns(
                    next_values[agent_id],
                    self.rl_config.gamma,
                    self.rl_config.gae_lambda,
                )
                rollout_data = self.storages[agent_id]
                self.agents[agent_id].update(rollout_data)

            # GNNのオンライン学習更新
            if online_gnn_learning and global_step % gnn_update_frequency == 0:
                print(f"\n🔄 [Global Step {global_step}] GNN更新チェック中...")
                self._trigger_gnn_update(global_step)

            # 進捗バーのメイン情報更新
            if self.storages:
                avg_reward = np.mean(
                    [
                        storage.rewards.mean().item()
                        for storage in self.storages.values()
                    ]
                )
                
                update_progress.set_postfix({
                    "Global_Step": f"{global_step:,}",
                    "Avg_Reward": f"{avg_reward:.4f}",
                    "Agents": self.num_agents
                })
                
                # 詳細ログは少ない頻度で出力
                if update % 50 == 0:
                    print(
                        f"\nUpdate {update}/{num_updates}, Global Step: {global_step:,}, Avg Reward: {avg_reward:.3f}"
                    )

        print("\n🎉 Training finished.")
        print(f"🔢 Total Global Steps: {global_step:,}")
        print(f"🤖 Total Agents: {self.num_agents}")
        print(f"🎯 Total Updates: {num_updates}")
        
        # 最終統計
        if self.storages:
            final_avg_reward = np.mean(
                [storage.rewards.mean().item() for storage in self.storages.values()]
            )
            print(f"📊 Final Average Reward: {final_avg_reward:.4f}")

    def _trigger_gnn_update(self, global_step):
        """GNNのオンライン学習更新をトリガー"""
        try:
            # 環境の特徴量抽出器にアクセス
            if hasattr(self.env, "feature_extractor") and hasattr(
                self.env.feature_extractor, "gnn_extractor"
            ):
                gnn_extractor = self.env.feature_extractor.gnn_extractor
                if gnn_extractor and gnn_extractor.online_learning:
                    print(f"\n🔄 [Step {global_step}] GNNオンライン学習更新を実行中...")

                    # バッファの内容をチェック
                    buffer_size = len(gnn_extractor.interaction_buffer)
                    if buffer_size > 0:
                        print(f"  インタラクションバッファサイズ: {buffer_size}")

                        # GNN更新実行
                        gnn_extractor._update_gnn_online()

                        # 統計情報表示
                        gnn_extractor.print_statistics()

                        # 定期的にモデルを保存（例：100ステップごと）
                        if (
                            global_step
                            % (
                                100
                                * getattr(self.config.irl, "gnn_update_frequency", 50)
                            )
                            == 0
                        ):
                            gnn_extractor.save_updated_model(
                                f"data/gnn_model_step_{global_step}.pt"
                            )
                            print(
                                f"  ✅ GNNモデル保存: gnn_model_step_{global_step}.pt"
                            )
                    else:
                        print("  インタラクションバッファが空のため、GNN更新をスキップ")
                else:
                    print(f"  GNNオンライン学習が無効化されています")
        except Exception as e:
            print(f"  ❌ GNN更新中にエラー: {e}")

    def save_models(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for agent_id, agent in self.agents.items():
            agent.save(os.path.join(directory, f"ppo_agent_{agent_id}.pth"))
