import os
from kazoo.envs.oss_simple import env as make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig


def main():
    # 1) 環境を直接生成
    env = make_oss_env(n_agents=3, backlog_size=6, seed=0)

    # 2) PPO エージェント初期化
    cfg = PPOConfig(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        rollout_len=128,
        mini_batch=4,
        epochs=4,
        device="cpu"
    )
    first_agent = env.agents[0]
    obs_space = env.observation_spaces[first_agent]
    act_space = env.action_spaces[first_agent]
    agent = IndependentPPO(
        obs_space=obs_space,
        act_space=act_space,
        n_agents=len(env.agents),
        cfg=cfg
    )

    # 3) 学習
    total_steps = 50000
    agent.train(env, total_steps=total_steps)

    # 4) モデル保存先を models/ ディレクトリに変更
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "indep_ppo_oss.pth")
    agent.save(save_path)
    print(f"モデルを保存しました → {save_path}")


if __name__ == "__main__":
    main()
