import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    rollout_len: int = 128
    mini_batch: int = 4
    epochs: int = 4
    device: str = "cpu"

class IndependentPPO:
    def __init__(
        self,
        obs_space,
        act_space,
        lr,
        gamma,
        gae_lambda,
        clip_eps,
        vf_coef,
        ent_coef,
        rollout_len,
        mini_batch,
        epochs,
        device="cpu"
    ):
        self.device = torch.device(device)
        self.obs_space = obs_space
        self.act_space = act_space

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.rollout_len = rollout_len
        self.mini_batch = mini_batch
        self.epochs = epochs

        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.n
        hid_dim = 64

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hid_dim, act_dim)
        self.value_head = nn.Linear(hid_dim, 1)
        self.net.to(self.device)
        self.policy_head.to(self.device)
        self.value_head.to(self.device)

        # ✅ 修正済み：すべてのパラメータを1つのリストにまとめる
        self.optimizer = optim.Adam(
            list(self.net.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters()),
            lr=self.lr,
        )

    def _to_tensor(self, obs):
        # obs をバッチ次元のあるテンソルに変換
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    def act(self, obs):
        x = self._to_tensor(obs)
        h = self.net(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def train(self, env, total_steps: int):
        obs = env.reset()
        for step in range(total_steps):
            # 対象エージェント名（例: agent_0）
            agent_id = env.agents[0]

            # 特定エージェントの観測のみ渡す
            action, logp, value = self.act(obs[agent_id])

            next_obs, reward, done, info = env.step({agent_id: action})

            # obs 更新
            obs = next_obs

            if done.get(agent_id, False):
                obs = env.reset()

        print("Training completed!")
        self.save("models/ppo_agent.pt")
        print("Model saved!")
    

    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "pi": self.policy_head.state_dict(),
            "vf": self.value_head.state_dict(),
        }, path)