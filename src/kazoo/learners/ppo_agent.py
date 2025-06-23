import os

import torch
import torch.nn as nn
from torch.distributions import Categorical


# Actor-Criticのネットワーク定義
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor（方策を学習）
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic（状態価値を学習）
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(self, obs_space, act_space, lr, gamma, epochs, eps_clip, device):
        self.device = device
        self.gamma = gamma
        self.epochs = epochs
        self.eps_clip = eps_clip
        self.policy = ActorCritic(obs_space.shape[0], act_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(obs_space.shape[0], act_space.n).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action_and_value(self, state):
        """行動選択と状態価値、対数確率を取得する"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs = self.policy.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.policy.critic(state)
        return action.cpu().numpy(), log_prob.cpu(), None, value.cpu()

    def update(self, memory):
        # モンテカルロ法でのリターン計算 (GAEはControllerで計算済みの想定)
        rewards = memory.returns

        # 利点を正規化
        advantages = memory.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 過去のデータを取得
        old_states = memory.obs.detach().to(self.device)
        old_actions = memory.actions.detach().to(self.device).long()
        old_log_probs = memory.log_probs.detach().to(self.device)

        # K epoch分、方策を最適化
        for _ in range(self.epochs):
            # 現在の方策での評価値を取得
            log_probs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions.squeeze()
            )

            # Policy Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs.squeeze().detach())

            # Surrogate Loss (PPOの目的関数)
            surr1 = ratios * advantages
            # ▼▼▼【ここが修正箇所】▼▼▼
            # self.eps_clipを使ってクリッピング
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            # ▲▲▲【ここまで修正箇所】▲▲▲

            # 最終的な損失
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # 勾配を計算して更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 新しい重みを古い方策にコピー
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(
            torch.load(filepath, map_location=lambda storage, loc: storage)
        )
        self.policy_old.load_state_dict(
            torch.load(filepath, map_location=lambda storage, loc: storage)
        )
