import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    """単一エージェントのためのPPO学習器"""
    def __init__(
        self,
        obs_space,
        act_space,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        epochs=4,
        mini_batch_size=32,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.obs_space = obs_space 
        self.act_space = act_space 

        # --- ハイパーパラメータ ---
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        # --- ニューラルネットワークの構築 ---
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.n
        hid_dim = 64

        # 共通ネットワーク
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid_dim), nn.Tanh(),
            nn.Linear(hid_dim, hid_dim), nn.Tanh()
        ).to(self.device)
        # ポリシーヘッド
        self.policy_head = nn.Linear(hid_dim, act_dim).to(self.device)
        # バリューヘッド
        self.value_head = nn.Linear(hid_dim, 1).to(self.device)

        # 最適化アルゴリズム
        self.optimizer = optim.Adam(list(self.net.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()), lr=lr)

    def get_action_and_value(self, obs, action=None):
        """観測から行動と価値を計算する"""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        hidden = self.net(x)
        logits = self.policy_head(hidden)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample() # 新しい行動をサンプリング
            
        log_prob = dist.log_prob(action) # 行動の対数確率
        entropy = dist.entropy() # ポリシーのエントロピー
        value = self.value_head(hidden).squeeze(-1) # 状態価値
        
        return action, log_prob, entropy, value

    # src/kazoo/learners/ppo_agent.py の update メソッド内
    def update(self, storage):
        """収集したデータ（storage）を使ってモデルを更新する"""
        advantages = torch.zeros_like(storage.rewards, device=self.device)
        last_gae_lam = 0
        # ★★★ ここを storage.num_steps に修正 ★★★
        with torch.no_grad():
            for t in reversed(range(storage.num_steps)):
                if t == storage.num_steps - 1:
                    # get_action_and_valueの引数をobsのみにする
                    next_obs_tensor = torch.as_tensor(storage.obs[t], dtype=torch.float32, device=self.device)
                    # 次のステップの観測から価値を推定
                    # next_obsをRolloutStorageに保存するのがより正確ですが、ここでは最後の観測で代用します
                    _, _, _, next_value = self.get_action_and_value(next_obs_tensor)
                else:
                    next_value = storage.values[t + 1]
                
                delta = storage.rewards[t] + self.gamma * next_value * (1.0 - storage.dones[t]) - storage.values[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1.0 - storage.dones[t]) * last_gae_lam
        
        returns = advantages + storage.values

        b_obs = storage.obs.reshape((-1,) + self.obs_space.shape)
        b_log_probs = storage.log_probs.reshape(-1)
        b_actions = storage.actions.reshape((-1,) + self.act_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        for epoch in range(self.epochs):
            # ★★★ ここを storage.num_steps に修正 ★★★
            indices = np.random.permutation(storage.num_steps)
            # ★★★ ここを storage.num_steps に修正 ★★★
            for start in range(0, storage.num_steps, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                _, new_log_prob, entropy, new_value = self.get_action_and_value(
                    b_obs[batch_indices], b_actions[batch_indices]
                )
                
                log_ratio = new_log_prob - b_log_probs[batch_indices]
                ratio = torch.exp(log_ratio)
                
                pg_loss1 = -b_advantages[batch_indices] * ratio
                pg_loss2 = -b_advantages[batch_indices] * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                value_loss = 0.5 * ((new_value - b_returns[batch_indices]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                # パラメータ全体に対して勾配クリッピングを行う
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.policy_head.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value_head.parameters(), 0.5)
                self.optimizer.step()