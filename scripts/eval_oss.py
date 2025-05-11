import csv
import os

import numpy as np
import torch
import yaml

from kazoo.envs.oss_simple import env as make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig

ACTION_NAMES = ["idle", "pickup", "code", "review"]  # 行動の名前

def to_tensor(obs_dict, agents, device):
    """観測辞書をTensorに変換"""
    arr = np.array([obs_dict[a] for a in agents])
    return torch.as_tensor(arr, dtype=torch.float32, device=device)

def run_episode(env, agent, cfg, writer=None, log=False):
    """1エピソード分を実行して報酬を返す"""
    obs_dict, _ = env.reset()
    obs = to_tensor(obs_dict, env.agents, cfg["device"])
    done = {a: False for a in env.agents}
    total_reward = {a: 0.0 for a in env.agents}
    step = 0

    while not all(done.values()):
        with torch.no_grad():
            logits, _ = agent.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy()  # 方策に従って行動を選択

        # 辞書形式の行動に変換
        action_dict = {a: int(actions[i]) for i, a in enumerate(env.agents)}

        if log:
            printable = {a: ACTION_NAMES[action_dict[a]] for a in env.agents}
            print(f"[step {step:02d}] actions: {printable}")

        # 環境を1ステップ進める
        obs_dict, reward, term, trunc, _ = env.step(action_dict)

        # 報酬を累積
        for a in env.agents:
            total_reward[a] += reward[a]
            done[a] = term[a] or trunc[a]

        # ログが有効な場合はCSVにも書き込む
        if writer:
            writer.writerow([step] + [action_dict[a] for a in env.agents] + [sum(total_reward.values())])

        # 次の観測へ
        step += 1
        obs = to_tensor(obs_dict, env.agents, cfg["device"])

    return total_reward

def main():
    # ① 設定ファイルを読み込む -----------------------------------
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    # ② モデルと環境の初期化 -----------------------------------
    env = make_oss_env(
        n_agents=cfg["n_agents"], backlog_size=cfg["backlog_size"], seed=cfg["seed"]
    )
    obs_space = env.observation_spaces[env.agents[0]]
    act_space = env.action_spaces[env.agents[0]]
    agent = IndependentPPO(obs_space, act_space, n_agents=cfg["n_agents"], cfg=PPOConfig(device=cfg["device"]))
    agent.load(f"{cfg['save_dir']}/indep_ppo_oss.pth")

    # ③ CSV ログ設定（最初の1エピソードだけ記録） ----------------
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/episode_log.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step"] + env.agents + ["reward_sum"])

    # ④ 複数エピソードを評価 -------------------------------------
    n_episodes = 10
    all_rewards = {a: [] for a in env.agents}

    for ep in range(n_episodes):
        print(f"\n▶ Evaluating episode {ep + 1}/{n_episodes}...")
        writer_ = writer if ep == 0 else None  # 最初の1回だけCSVに記録
        rewards = run_episode(env, agent, cfg, writer=writer_, log=(ep == 0))
        for a in env.agents:
            all_rewards[a].append(rewards[a])

    csv_f.close()

    # ⑤ 平均報酬の出力 -------------------------------------------
    print("\n◆ average total reward over", n_episodes, "episodes")
    for a in env.agents:
        avg = np.mean(all_rewards[a])
        std = np.std(all_rewards[a])
        print(f"  {a}: {avg:.2f} ± {std:.2f}")
    print(f"▶ 行動ログ CSV: {csv_path}（※エピソード1のみ記録）")

if __name__ == "__main__":
    main()