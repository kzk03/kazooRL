# scripts/train_mini.py  ―― サニティチェック用（10 秒以内で完了）

import torch

from kazoo.envs.oss_simple import env as make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig


def main():
    # ── 1) きわめて軽い環境を生成 ────────────────
    env = make_oss_env(
        n_agents=2, backlog_size=2, seed=0  # 人数を最小化  # タスクも最小
    )

    # ── 2) PPO エージェント初期化 ───────────────
    obs_space = env.observation_spaces[env.agents[0]]
    act_space = env.action_spaces[env.agents[0]]
    agent = IndependentPPO(
        obs_space,
        act_space,
        n_agents=2,
        cfg=PPOConfig(  # ハイパラは本番と同じでも OK
            lr=3e-4,
            rollout_len=32,  # すばやく 1 epoch 終わる長さ
            epochs=2,  # 更新も最小回数
            device="cpu",
        ),
    )

    # ── 3) たった 2,000 step だけ学習 ───────────
    agent.train(env, total_steps=2_000)
    print("✅ サニティチェック完了 — 学習ループは生存しています")


if __name__ == "__main__":
    main()
