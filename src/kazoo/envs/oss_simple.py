import datetime
import json
import math
import random
import sys
from pathlib import Path
from typing import List
from dateutil import parser
import gym
import numpy as np
import yaml
from gym import spaces
from gym.spaces import Discrete
from data.generate_backlog import load_tasks


class Task:
    def __init__(self, id, title, author, complexity, created_at, labels):
        self.id = id
        self.title = title
        self.author = author
        self.complexity = complexity
        self.created_at = created_at
        self.labels = labels
        self.state = None

class OSSSimpleEnv:
    def __init__(self, tasks: List[Task], profile_file: str, *args, **kwargs):

        with open(profile_file) as f:
            profiles = yaml.safe_load(f)

        self.dev_ids = list(profiles.keys())  # すべての開発者ID（user名）を取得
        self.n_agents = len(self.dev_ids)  # エージェント数を設定
        self.tasks = tasks
        self.current_task = None
        self.index = 0
        self.n_agents = kwargs.get("n_agents", len(self.dev_ids)) 
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        # 各エージェントの観測空間を Box 形式で定義 (複雑度と経過日数)
        self.observation_spaces = {
            agent: spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([5, 365], dtype=np.float32),
                dtype=np.float32
            ) for agent in self.agents
        }
        self.action_spaces = {
            agent: Discrete(self.n_agents) for agent in self.agents
        }

    def reset(self):
        self.index = 0
        self.current_task = self.tasks[self.index]
        return self._get_obs()

    def sample_action(self):
        return random.choice(range(3))

    def step(self, action):
        self.index += 1
        terminated = self.index >= len(self.tasks)
        if not terminated:
            self.current_task = self.tasks[self.index]
        return self._get_obs(), 0.0, terminated, {}

    def _get_obs(self):
        # 現在タスクの複雑度と経過日数をベクトルで返す
        created = parser.isoparse(self.current_task["createdAt"])
        now = datetime.datetime.now(datetime.timezone.utc)
        days = (now - created).days
        days = min(days, 365)
        obs_vector = np.array([
            self.current_task["complexity"],
            days
        ], dtype=np.float32)
        return obs_vector

class OSSDevEnv(OSSSimpleEnv):
    def __init__(self,
                 task_file: str = "data/github_data.json",
                 profile_file: str = "configs/dev_profiles.yaml",
                 *args, **kwargs):
        self.task_file = Path(task_file)
        self.profile_file = Path(profile_file)
        self.update_dev_profiles()
        tasks = load_tasks(task_file)
        super().__init__(tasks=tasks, *args, **kwargs)
        self._original_step = super().step
        return OSSSimpleEnv(tasks, profile_file=profile_file, **kwargs)

    def estimate_complexity(self, pr: dict) -> float:
        msg_len = len(pr.get("body", ""))
        return min(5, max(1, msg_len // 200))

    def load_tasks(self) -> List[Task]:
        with self.task_file.open("r") as f:
            data = json.load(f)
        tasks: List[Task] = []
        for pr in data["prs"]:
            created = datetime.datetime.strptime(pr["createdAt"], "%Y-%m-%dT%H:%M:%SZ")
            task = Task(
                id=pr["number"],
                title=pr["title"],
                author=pr["author"]["login"],
                complexity=self.estimate_complexity(pr),
                created_at=created,
                labels=[l["name"] for l in pr.get("labels", {}).get("nodes", [])] if pr.get("labels") else []
            )
            if pr.get("mergedAt"):
                task.state = "MERGED"
            elif pr.get("state") == "CLOSED":
                task.state = "CLOSED"
            tasks.append(task)
        return tasks

    def update_dev_profiles(self):
        with self.task_file.open("r") as f:
            data = json.load(f)
        try:
            with self.profile_file.open("r") as f:
                profiles = f.read()
        except FileNotFoundError:
            profiles = ""

        existing = set()
        for line in profiles.splitlines():
            if line and not line.startswith(" ") and ":" in line:
                existing.add(line.strip(":"))

        new_profiles = []
        for pr in data["prs"]:
            login = pr["author"]["login"]
            if login not in existing:
                block = (
                    f"{login}:\n"
                    "  lang_emb: [1.0, 0.0, 0.0]\n"
                    "  skill:\n"
                    "    code: 0.5\n"
                    "    review: 0.3\n"
                    "  task_types: [0, 0, 0]\n"
                )
                new_profiles.append(block)
                existing.add(login)

        if new_profiles:
            with self.profile_file.open("a") as f:
                f.write("\n" + "\n".join(new_profiles))

    def step(self, action):
        obs, _, terminated, info = self._original_step(action)
        task = self.current_task
        if task.state == "MERGED":
            reward = 1.0
        elif task.state == "CLOSED":
            reward = -0.5
        elif (datetime.datetime.utcnow() - task.created_at).days > 30:
            reward = -0.2
        else:
            reward = 0.0
        return obs, reward, terminated, info

# 環境生成用関数
def make_oss_env(task_file, profile_file, **kwargs):
    tasks = load_tasks(task_file)  # タスクの読み込み（あなたの実装に合わせて）
    return OSSSimpleEnv(tasks, profile_file=profile_file, **kwargs)

if __name__ == '__main__':
    env_instance = make_oss_env()
    obs = env_instance.reset()
    done = False
    while not done:
        action = env_instance.sample_action()
        obs, reward, done, info = env_instance.step(action)
        print(f"Reward: {reward}")
