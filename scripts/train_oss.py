import os

import numpy as np
import torch
import yaml

from kazoo.envs.oss_simple import env as make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig


def main():
    # ---------- ① YAML設定ファイルを読み込み ----------
    with open("configs/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # ---------- ② 開発者プロフィールも読み込み ----------
    with open("configs/dev_profiles.yaml", "r") as f:
        dev_profiles = yaml.safe_load(f)

    # ---------- ③ 環境を生成（プロフィール付き） ----------
    env = make_oss_env(
        n_agents=cfg["n_agents"],
        backlog_size=cfg["backlog_size"],
        seed=cfg["seed"],
        profiles=dev_profiles  # ← これが環境へ渡る
    )

    # ---------- ④ PPOエージェントを初期化 ----------
    first = env.agents[0]
    agent = IndependentPPO(
        obs_space=env.observation_spaces[first],
        act_space=env.action_spaces[first],
        n_agents=cfg["n_agents"],
        cfg=PPOConfig(
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_eps=cfg["clip_eps"],
            vf_coef=cfg["vf_coef"],
            ent_coef=cfg["ent_coef"],
            rollout_len=cfg["rollout_len"],
            mini_batch=cfg["mini_batch"],
            epochs=cfg["epochs"],
            device=cfg["device"],
        )
    )

    # ---------- ⑤ 学習開始 ----------
    agent.train(env, total_steps=cfg["total_steps"])

    # ---------- ⑥ モデルを保存 ----------
    os.makedirs(cfg["save_dir"], exist_ok=True)
    agent.save(f"{cfg['save_dir']}/indep_ppo_oss.pth")

if __name__ == "__main__":
    main()
