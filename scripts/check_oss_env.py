# 仮想OSS環境の動作確認


from pprint import pprint

import yaml

from kazoo.envs.oss_simple import env as make_oss_env  # 直接インポート


def main():
    with open("experiments/base.yaml") as f:
        cfg = yaml.safe_load(f)

    env = make_oss_env(
        n_agents     = cfg["n_agents"],
        backlog_size = cfg["backlog_size"],
        seed         = cfg["seed"]
    )
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
