import json
import os
from collections import defaultdict

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import GATFeatureExtractorRL
from kazoo.learners.independent_ppo_controller import IndependentPPOController


@hydra.main(config_path="../../configs", config_name="bot_excluded_production", version_base=None)
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
    try:
        print(f"Reward weights path: {cfg.irl.output_weights_path}")
        if not os.path.exists(cfg.irl.output_weights_path):
            print(f"Warning: Reward weights file not found: {cfg.irl.output_weights_path}")
        
        env = OSSSimpleEnv(
            config=cfg,
            backlog=backlog,
            dev_profiles=dev_profiles,
            reward_weights_path=cfg.irl.output_weights_path,
        )
        print("✅ Environment initialized successfully")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. RLコントローラーの初期化
    print("3. Setting up PPO controller...")
    try:
        print("Creating IndependentPPOController...")
        controller = IndependentPPOController(env=env, config=cfg)
        print("✅ PPO controller initialized successfully")
    except Exception as e:
        print(f"Error initializing PPO controller: {e}")
        import traceback
        traceback.print_exc()
        return
        return

    # 4. 強化学習の実行
    print("4. Starting RL training loop...")

    try:
        total_timesteps = cfg.rl.total_timesteps
        print(f"Total timesteps configured: {total_timesteps}")
    except Exception as e:
        print(f"Error accessing total_timesteps: {e}")
        raise ValueError("Config file must have 'rl.total_timesteps' defined.")

    # PPOのポリシー引数を定義（現在は使用していない）
    policy_kwargs = dict(features_extractor_class=GATFeatureExtractorRL)

    try:
        print("Starting PPO learning...")
        controller.learn(total_timesteps=total_timesteps)
        print("PPO learning completed successfully!")
    except Exception as e:
        print(f"Error during PPO learning: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 学習済みモデルの保存
    print("5. Training finished. Saving RL agent models...")
    try:
        controller.save_models(cfg.rl.output_model_dir)
        print(f"✅ RL models saved to: {cfg.rl.output_model_dir}")
    except Exception as e:
        print(f"Error saving models: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
