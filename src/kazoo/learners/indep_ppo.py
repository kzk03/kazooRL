#IPPOの中身（ニューラルネット，報酬計算，学習処理）などを実装

from __future__ import annotations

"""Independent-PPO (IPPO) — minimalベースライン実装

* すべてのエージェントが同一ネットワークを共有しつつ、
  行動は独立にサンプル（通信・値混合なし）
* Simple multi-agent dict/env support added.
"""

import math
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    rollout_len: int = 256
    mini_batch: int = 4
    epochs: int = 4
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.policy(h), self.value(h)


class IndependentPPO:
    def __init__(
        self,
        obs_space: spaces.Space,
        act_space: spaces.Space,
        n_agents: int,
        cfg: PPOConfig | None = None,
    ):
        self.cfg = cfg or PPOConfig()
        assert isinstance(act_space, spaces.Discrete), "Discrete action only"
        self.n_agents = n_agents

        obs_dim = int(math.prod(obs_space.shape))
        self.net = ActorCritic(obs_dim, act_space.n).to(self.cfg.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        # rollout storage
        self.reset_storage()

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value.squeeze(-1)

    def train(self, env, total_steps: int):
        raw_obs, _ = env.reset()
        multi = isinstance(raw_obs, dict)
        if multi:
            obs_arr = np.array([raw_obs[a] for a in env.agents])
            obs = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.cfg.device)
        else:
            obs = self._to_tensor(raw_obs)

        step = 0
        while step < total_steps:
            self.reset_storage()
            for _ in range(self.cfg.rollout_len):
                action_arr, logp, value = self.act(obs)
                if multi:
                    actions = {a: int(action_arr[i].item()) for i, a in enumerate(env.agents)}
                else:
                    actions = action_arr.cpu().numpy()

                nxt_obs, reward, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated

                self.store(obs, action_arr, logp, value, reward, done)
                if multi:
                    obs_arr = np.array([nxt_obs[a] for a in env.agents])
                    obs = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.cfg.device)
                else:
                    obs = self._to_tensor(nxt_obs)

                step += self.n_agents
                if step >= total_steps:
                    break

            with torch.no_grad():
                _, _, last_val = self.act(obs)
            self.finish_gae(last_val)
            self.update()

    def reset_storage(self):
        self.obs_buf: list[torch.Tensor] = []
        self.act_buf: list[torch.Tensor] = []
        self.logp_buf: list[torch.Tensor] = []
        self.rew_buf: list[torch.Tensor] = []
        self.done_buf: list[torch.Tensor] = []
        self.val_buf: list[torch.Tensor] = []

    def store(
        self,
        obs: Union[torch.Tensor, None],
        act: torch.Tensor,
        logp: torch.Tensor,
        val: torch.Tensor,
        rew: Union[np.ndarray, dict],
        done: Union[torch.Tensor, dict],
    ):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        if isinstance(rew, dict):
            rew_tensor = torch.as_tensor(np.array([rew[a] for a in rew]), dtype=torch.float32, device=self.cfg.device)
            done_tensor = torch.as_tensor(np.array([done[a] for a in done], dtype=float), dtype=torch.float32, device=self.cfg.device)
        else:
            rew_tensor = self._to_tensor(rew)
            done_tensor = self._to_tensor(done)
        self.rew_buf.append(rew_tensor)
        self.done_buf.append(done_tensor)
        self.val_buf.append(val)

    def finish_gae(self, last_value: torch.Tensor):
        R = last_value
        adv_buf: list[torch.Tensor] = []
        ret_buf: list[torch.Tensor] = []
        for t in reversed(range(len(self.rew_buf))):
            mask = 1.0 - self.done_buf[t]
            R = self.rew_buf[t] + self.cfg.gamma * R * mask
            if t < len(self.rew_buf) - 1:
                delta = self.rew_buf[t] + self.cfg.gamma * self.val_buf[t+1] * mask - self.val_buf[t]
                gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * adv_buf[0]
            else:
                delta = R - self.val_buf[t]
                gae = delta
            adv_buf.insert(0, gae)
            ret_buf.insert(0, R)

        self.adv_buf = torch.stack(adv_buf).detach()
        self.ret_buf = torch.stack(ret_buf).detach()
        self.obs_buf = torch.stack(self.obs_buf)
        self.act_buf = torch.stack(self.act_buf)
        self.logp_buf = torch.stack(self.logp_buf)

    def update(self):
        B = self.obs_buf.shape[0] * self.n_agents
        idx = torch.randperm(B)
        mb = B // self.cfg.mini_batch

        obs = self.obs_buf.view(B, -1)
        act = self.act_buf.view(B)
        old_logp = self.logp_buf.view(B).detach()
        ret = self.ret_buf.view(B)
        adv = (self.adv_buf.view(B) - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-8)

        for _ in range(self.cfg.epochs):
            for i in range(0, B, mb):
                j = idx[i:i+mb]
                logits, value = self.net(obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[j])
                ratio = torch.exp(logp - old_logp[j])

                s1 = ratio * adv[j]
                s2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[j]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(value.squeeze(-1), ret[j])
                entropy = dist.entropy().mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def save(self, path: str) -> None:
        """モデルのパラメータを指定パスに保存"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)
        print(f"[PPO] モデル保存: {path}")

    def load(self, path: str) -> None:
        """保存済みパラメータを読み込む"""
        state = torch.load(path, map_location=self.cfg.device)
        self.net.load_state_dict(state)
        print(f"[PPO] モデル読み込み: {path}")

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.cfg.device)
