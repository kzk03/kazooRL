import json
import os
from collections import defaultdict

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    強化学習（PPO）を使って、開発者エージェントを訓練する。
    """
    print("--- Starting OSS Reinforcement Learning Training ---")
    print(OmegaConf.to_yaml(cfg))

    # 1. データの読み込み
    print("1. Loading data...")
    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)

    # 2. 環境の初期化
    print("2. Initializing environment...")
    env = OSSSimpleEnv(
        config=cfg,
        backlog=backlog,
        dev_profiles=dev_profiles,
        reward_weights_path=cfg.irl.output_weights_path,
    )

    # 3. RLコントローラーの初期化
    print("3. Setting up PPO controller...")
    controller = IndependentPPOController(env=env, config=cfg)

    # 4. 強化学習の実行
    print("4. Starting RL training loop...")

    # ▼▼▼【ここが修正箇所】▼▼▼
    # 設定ファイルから 'rl.total_timesteps' を読み込む
    try:
        total_timesteps = cfg.rl.total_timesteps
    except Exception:
        raise ValueError("Config file must have 'rl.total_timesteps' defined.")

    controller.learn(total_timesteps=total_timesteps)
    # ▲▲▲【ここまで修正箇所】▲▲▲

    # 5. 学習済みモデルの保存
    print("5. Training finished. Saving RL agent models...")
    controller.save_models(cfg.rl.output_model_dir)
    print(f"✅ RL models saved to: {cfg.rl.output_model_dir}")


if __name__ == "__main__":
    main()
