import csv
import os

import numpy as np
import torch
import yaml

from kazoo.envs.oss_simple import env as make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig

ACTION_NAMES = ["idle", "pickup", "code", "review"]


def to_tensor(obs_dict, agents, device):
    arr = np.array([obs_dict[a] for a in agents])
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def main():
    # ① YAML 読み込み ------------------------------------------------------
    with open("experiments/base.yaml") as f:
        cfg = yaml.safe_load(f)

    # ② 環境生成 -----------------------------------------------------------
    env = make_oss_env(
        n_agents=cfg["n_agents"], backlog_size=cfg["backlog_size"], seed=cfg["seed"]
    )

    # ③ エージェント初期化 & モデル読込 ----------------------------------
    obs_space = env.observation_spaces[env.agents[0]]
    act_space = env.action_spaces[env.agents[0]]
    agent = IndependentPPO(obs_space, act_space, n_agents=cfg["n_agents"], cfg=PPOConfig(device=cfg["device"]))
    agent.load(f"{cfg['save_dir']}/indep_ppo_oss.pth")

    # ④ 評価ループ ---------------------------------------------------------
    obs_dict, _ = env.reset()
    obs = to_tensor(obs_dict, env.agents, cfg["device"])
    done = {a: False for a in env.agents}
    total_reward = {a: 0.0 for a in env.agents}

    # CSV ログ準備
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/episode_log.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step"] + env.agents + ["reward_sum"])

    step = 0
    while not all(done.values()):
        with torch.no_grad():
            logits, _ = agent.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy()  # 確率に従ってサンプル

        action_dict = {a: int(actions[i]) for i, a in enumerate(env.agents)}
        printable = {a: ACTION_NAMES[action_dict[a]] for a in env.agents}
        print(f"[step {step:02d}] actions: {printable}")

        obs_dict, reward, term, trunc, _ = env.step(action_dict)
        for a in env.agents:
            total_reward[a] += reward[a]
            done[a] = term[a] or trunc[a]

        writer.writerow([step] + [action_dict[a] for a in env.agents] + [sum(total_reward.values())])
        step += 1
        obs = to_tensor(obs_dict, env.agents, cfg["device"])

    csv_f.close()

    # ⑤ 結果表示 -----------------------------------------------------------
    print("\n◆ episode total reward")
    for a, r in total_reward.items():
        print(f"  {a}: {r:.2f}")
    print(f"▶ 行動ログ CSV: {csv_path}")


if __name__ == "__main__":
    main()
