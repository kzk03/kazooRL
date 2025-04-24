# scripts/train.py
import gymnasium as gym
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig

def main():
    env = gym.make("GridWorldMA-v0", size=5, n_agents=2)
    agent = IndependentPPO(
        env.observation_space,
        env.action_space,
        n_agents=2,
        cfg=PPOConfig(device="cpu")   # GPU を使うなら "cuda"
    )
    agent.train(env, total_steps=50_000)

if __name__ == "__main__":
    main()
