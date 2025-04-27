import os  # ★ 追記

import yaml

from kazoo.envs.oss_simple import env as make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig


def main():
    # ---------- ① YAML 読み込み ----------
    with open("experiments/base.yaml") as f:
        cfg = yaml.safe_load(f)

    # ---------- ② 環境生成 ----------
    env = make_oss_env(
        n_agents     = cfg["n_agents"],
        backlog_size = cfg["backlog_size"],
        seed         = cfg["seed"]
    )

    # ---------- ③ エージェント初期化 ----------
    first  = env.agents[0]
    agent  = IndependentPPO(
        obs_space = env.observation_spaces[first],
        act_space = env.action_spaces[first],
        n_agents  = cfg["n_agents"],
        cfg       = PPOConfig(
            lr          = cfg["lr"],
            gamma       = cfg["gamma"],
            gae_lambda  = cfg["gae_lambda"],
            clip_eps    = cfg["clip_eps"],
            vf_coef     = cfg["vf_coef"],
            ent_coef    = cfg["ent_coef"],
            rollout_len = cfg["rollout_len"],
            mini_batch  = cfg["mini_batch"],
            epochs      = cfg["epochs"],
            device      = cfg["device"],
        ),
    )

    # ---------- ④ 学習 ----------
    agent.train(env, total_steps = cfg["total_steps"])

    # ---------- ⑤ モデル保存 ----------
    os.makedirs(cfg["save_dir"], exist_ok=True)
    agent.save(f"{cfg['save_dir']}/indep_ppo_oss.pth")

if __name__ == "__main__":
    main()
