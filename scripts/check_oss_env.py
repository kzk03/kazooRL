# scripts/check_oss_env.py
from pprint import pprint
from kazoo.envs.oss_simple import env as make_oss_env  # 直接インポート

def main():
    # gym.make は使わず、ファクトリ関数を直接呼ぶ
    env = make_oss_env(n_agents=3, backlog_size=6, seed=0, render_mode=None)
    obs, infos = env.reset()
    pprint(obs)
    for _ in range(5):
        # 各エージェントの action_space は env.action_spaces[agent]
        actions = {
            agent: env.action_spaces[agent].sample()
            for agent in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        pprint(rewards)
        if all(terms.values()) or all(truncs.values()):
            break

if __name__ == "__main__":
    main()
