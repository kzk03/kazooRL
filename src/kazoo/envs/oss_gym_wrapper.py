# src/kazoo/envs/oss_gym_wrapper.py
from gymnasium import Env, spaces
from kazoo.envs.oss_simple import OSSDevEnv

class OSSGymWrapper(Env):
    metadata = {"render_modes":[]}
    def __init__(self, n_agents=3, backlog_size=6, **kwargs):
        self._pe = OSSDevEnv(n_agents, backlog_size, seed=kwargs.get("seed"))
        # Dict でまとめる
        self.observation_space = spaces.Dict({
            a: self._pe.observation_spaces[a] for a in self._pe.agents
        })
        self.action_space = spaces.Dict({
            a: self._pe.action_spaces[a] for a in self._pe.agents
        })
    def reset(self, **kw):
        return self._pe.reset(seed=kw.get("seed"))
    def step(self, acts):
        return self._pe.step(acts)
