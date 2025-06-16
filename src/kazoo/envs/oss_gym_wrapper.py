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
        action_list = [actions[agent] for agent in self.agents]

        # 2. 全エージェントの行動リストを一度に渡して、stepを1回だけ実行
        observations, rewards, terminations, infos = self.env.step(action_list)

        # 3. 必要に応じて、返り値を辞書形式に変換して返す
        #    (もしself.env.stepの返り値が既に辞書なら、この処理は不要)
        obs_dict = {agent: observations[i] for i, agent in enumerate(self.agents)}
        rew_dict = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        term_dict = {agent: terminations[i] for i, agent in enumerate(self.agents)}
        info_dict = {agent: infos[i] for i, agent in enumerate(self.agents)}

        return obs_dict, rew_dict, term_dict, info_dict
