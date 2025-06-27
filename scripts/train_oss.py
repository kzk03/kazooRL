import json
import os
from collections import defaultdict

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController
from kazoo.features.feature_extractor import GNNFeatureExtractor


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    強化学習（PPO）を使って、開発者エージェントを訓練する。
    """
    print(OmegaConf.to_yaml(cfg))
     # 1. データの読み込み前にファイルの存在確認
    print("1. Checking file paths...")
    
    if not os.path.exists(cfg.env.backlog_path):
        print(f"Error: Backlog file not found: {cfg.env.backlog_path}")
        return
    
    if not os.path.exists(cfg.env.dev_profiles_path):
        print(f"Error: Dev profiles file not found: {cfg.env.dev_profiles_path}")
        return
    
    print("✅ All required files found")
    

    # 1. データの読み込み
    print("1. Loading data...")
    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)

        # ▼▼▼【デバッグ用の確認コード追加】▼▼▼
    print(f"Loaded {len(backlog)} tasks from backlog")
    print(f"Loaded {len(dev_profiles)} developer profiles")
    print(f"Config num_developers: {cfg.get('num_developers', 'Not set')}")
    
    # 開発者プロファイルの内容確認
    if dev_profiles:
        print("Developer profile keys:", list(dev_profiles.keys()))
    else:
        print("WARNING: No developer profiles loaded!")
        return

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

    try:
        total_timesteps = cfg.rl.total_timesteps
    except Exception:
        raise ValueError("Config file must have 'rl.total_timesteps' defined.")
    
        # PPOのポリシー引数を定義
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor
    )

    # 学習コントローラーの初期化時にpolicy_kwargsを渡すように変更
    # (IndependentPPOControllerとPPOAgentの改造が必要になる場合があります)
    # ここでは、stable-baselines3のPPOを直接使う場合を想定
    
    # from stable_baselines3 import PPO
    # model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(total_timesteps=10000)

    # KazooRLのコントローラーを使う場合は、コントローラーの__init__で
    # policy_kwargsを受け取れるように改造が必要です。
    controller = IndependentPPOController(
        env=env,
        config=cfg.learner,
        policy_kwargs=policy_kwargs # policy_kwargsを渡す
    )

    controller.learn(total_timesteps=total_timesteps)

    # 5. 学習済みモデルの保存
    print("5. Training finished. Saving RL agent models...")
    controller.save_models(cfg.rl.output_model_dir)
    print(f"✅ RL models saved to: {cfg.rl.output_model_dir}")


if __name__ == "__main__":
    main()
