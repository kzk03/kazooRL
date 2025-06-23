import os

import numpy as np
import torch
from gymnasium import spaces

from kazoo.learners.ppo_agent import PPOAgent


class RolloutStorage:
    """
    PPOのための経験データを一時的に保存し、リターンとアドバンテージを計算する。
    """

    def __init__(self, num_steps, obs_space, act_space, device):
        self.device = device
        self.num_steps = num_steps

        # ストレージを(ステップ数, 次元)の形で初期化
        self.obs = torch.zeros((num_steps,) + obs_space.shape).to(device)
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
        self.obs[self.step].copy_(torch.as_tensor(obs, device=self.device))
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

            self.agents[agent_id] = PPOAgent(
                obs_space=obs_space,
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

        num_updates = (
            int(total_timesteps / self.rl_config.rollout_len / self.num_agents)
            if self.num_agents > 0
            else 0
        )

        for update in range(1, num_updates + 1):
            for step in range(self.rl_config.rollout_len):
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

            if self.storages:
                avg_reward = np.mean(
                    [
                        storage.rewards.mean().item()
                        for storage in self.storages.values()
                    ]
                )
                print(
                    f"Update {update}/{num_updates}, Global Step: {global_step}, Avg Reward: {avg_reward:.3f}"
                )

        print("Training finished.")

    def save_models(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for agent_id, agent in self.agents.items():
            agent.save(os.path.join(directory, f"ppo_agent_{agent_id}.pth"))