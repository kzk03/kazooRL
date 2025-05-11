#「OSS開発」を模したマルチエージェント環境を定義するファイル

import random

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

from kazoo.consts.actions import Action  # ← Enum による定義


def env(n_agents: int = 3,
        backlog_size: int = 6,
        seed: int | None = None,
        **kwargs):
    return OSSDevEnv(n_agents, backlog_size, seed)


class OSSDevEnv(ParallelEnv):
    metadata = {"render_modes": []}

    def __init__(self, n_agents: int, backlog_size: int, seed=None):
        super().__init__()
        self.n_agents = n_agents
        self.agents = [f"dev_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.backlog_init = backlog_size
        self.np_random = np.random.default_rng(seed)
        self.skills = {
            a: {
                "review": np.random.uniform(0.5, 1.0),
                "code": np.random.uniform(0.5, 1.0)
            }
            for a in self.agents
        }

        # spaces
        self.observation_spaces = {
            a: spaces.Box(0, 10, shape=(5,), dtype=np.int32)
            for a in self.agents
        }
        self.action_spaces = {a: spaces.Discrete(4) for a in self.agents}

        self.reset(seed=seed)

    # ---------------- core API ----------------
    def reset(self, seed=None, options=None):
        self.backlog = [self.np_random.integers(1, 4)  # complexity 1-3
                        for _ in range(self.backlog_init)]
        self.progress = {a: 0 for a in self.agents}
        self.pending = {a: 0 for a in self.agents}
        self.steps = 0

        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        # 報酬初期化をここに移動
        self.rewards = {a: 0.0 for a in self.agents}

        self._apply_actions(actions)
        self._apply_time_penalty()
        terminations, truncations = self._check_termination()
        observations = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, self.rewards, terminations, truncations, infos

    def _apply_actions(self, actions):
        for a, act in actions.items():
            if act == Action.PICKUP and self.progress[a] == 0 and self.backlog:
                self.progress[a] = self.backlog.pop()
            elif act == Action.CODE and self.progress[a] > 0:
                self.progress[a] -= 1
                if self.progress[a] == 0:
                    reviewer = random.choice([x for x in self.agents if x != a])
                    self.pending[reviewer] += 1
            elif act == Action.REVIEW and self.pending[a] > 0:
                self.pending[a] -= 1
                author = random.choice([x for x in self.agents if x != a])
                reviewer_skill = self.skills[a]["review"]
                self._reward_contribution(a, author, reviewer_skill)

    def _reward_contribution(self, reviewer, author, skill):
        self.rewards[author] += 1.0 * skill
        self.rewards[reviewer] += 0.2 + 0.3 * (skill - 0.5)

    def _apply_time_penalty(self):
        for a in self.agents:
            self.rewards[a] -= 0.01

    def _check_termination(self):
        self.steps += 1
        done_all = (not self.backlog and
                    all(v == 0 for v in self.progress.values()) and
                    all(v == 0 for v in self.pending.values()))
        timeout = self.steps >= 100
        terminations = {a: done_all for a in self.agents}
        truncations = {a: timeout for a in self.agents}
        return terminations, truncations

    # ---------------- helpers ----------------
    def _obs(self, a):
        return np.array([
            len(self.backlog),
            self.progress[a],
            self.pending[a],
            self.skills[a]["code"],
            self.skills[a]["review"]
        ], dtype=np.int32)
