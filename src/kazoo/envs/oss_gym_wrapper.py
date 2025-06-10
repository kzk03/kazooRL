import numpy as np
from gymnasium import spaces


class OSSGymWrapper:
    def __init__(self, env):
        self.env = env
        self.agents = env.agents

        # Dict形式のobservation/action spaceを保持
        self.observation_spaces = {
            agent: env.observation_spaces[agent] for agent in self.agents
        }
        self.action_spaces = {agent: env.action_spaces[agent] for agent in self.agents}

    def reset(self):
        obs = self.env.reset()
        return {agent: obs for agent in self.agents}

    def step(self, actions):
        # actions: Dict[agent_id, action]
        rewards = {}
        observations = {}
        terminations = {}
        infos = {}

        for agent_id, action in actions.items():
            obs, reward, done, info = self.env.step(action)
            observations[agent_id] = obs
            rewards[agent_id] = reward
            terminations[agent_id] = done
            infos[agent_id] = info

        return observations, rewards, terminations, infos
