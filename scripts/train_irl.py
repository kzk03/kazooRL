import json
import pickle  # ▼▼▼ pickleをインポート ▼▼▼
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.envs.task import Task
from kazoo.features.feature_extractor import FeatureExtractor


def main():
    """逆強化学習のメイン処理."""

    print("1. Loading configuration and data...")
    cfg = OmegaConf.load("configs/base.yaml")

    try:
        with open(cfg.irl.expert_path, "rb") as f:
            trajectories = pickle.load(f)
            # 軌跡は1つしかないので、最初の要素を取得
            expert_trajectory_steps = trajectories[0] if trajectories else []
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading expert trajectories from {cfg.irl.expert_path}: {e}")
        print("Please run 'create_expert_trajectories.py' first.")
        return

    if not expert_trajectory_steps:
        print("No expert steps found in the trajectory file. Exiting.")
        return

    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog_data = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles_data = yaml.safe_load(f)

    print("2. Initializing environment and feature extractor...")
    env = OSSSimpleEnv(cfg, backlog_data, dev_profiles_data)
    feature_extractor = FeatureExtractor(cfg)
    feature_dim = len(feature_extractor.feature_names)

    # 全タスクをIDで検索できるように辞書化
    all_tasks_db = {task.id: task for task in env.backlog}

    print(f"3. Setting up IRL model with {feature_dim} features...")
    reward_weights = torch.randn(feature_dim, requires_grad=True)
    optimizer = optim.Adam([reward_weights], lr=cfg.irl.learning_rate)

    print(
        f"4. Starting training loop with {len(expert_trajectory_steps)} expert steps..."
    )
    for epoch in range(cfg.irl.epochs):
        total_loss = 0

        # .pklファイルから読み込んだ軌跡の各ステップをループ
        for step_data in expert_trajectory_steps:
            optimizer.zero_grad()

            # --- 軌跡データから状態と行動を取得 ---
            state = step_data["state"]
            action_details = step_data["action_details"]

            developer_id = action_details.get("developer")
            expert_task_id = action_details.get("task_id")
            event_timestamp = datetime.fromisoformat(
                action_details.get("timestamp").replace("Z", "+00:00")
            )

            # --- 特徴量計算の準備 ---
            developer_profile = dev_profiles_data.get(developer_id)
            expert_task = all_tasks_db.get(expert_task_id)
            if not developer_profile or not expert_task:
                continue

            developer_obj = {"name": developer_id, "profile": developer_profile}

            # 特徴量計算のために環境の状態を一時的に設定
            env.current_time = event_timestamp

            # --- 特徴量の計算 ---
            expert_features = feature_extractor.get_features(
                expert_task, developer_obj, env
            )
            expert_features = torch.from_numpy(expert_features).float()

            other_features_list = []
            # 「取り得た他の行動」は、その時点でのオープンなタスクリスト
            for other_task_id in state["open_task_ids"]:
                if other_task_id != expert_task_id:
                    other_task = all_tasks_db.get(other_task_id)
                    if other_task:
                        features = feature_extractor.get_features(
                            other_task, developer_obj, env
                        )
                        other_features_list.append(torch.from_numpy(features).float())

            if not other_features_list:
                continue

            # --- 損失の計算と更新 ---
            expert_reward = torch.dot(reward_weights, expert_features)
            other_rewards = torch.stack(
                [torch.dot(reward_weights, f) for f in other_features_list]
            )
            log_sum_exp_other_rewards = torch.logsumexp(other_rewards, dim=0)

            loss = -(expert_reward - log_sum_exp_other_rewards)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if expert_trajectory_steps:
            avg_loss = total_loss / len(expert_trajectory_steps)
            print(f"Epoch {epoch + 1}/{cfg.irl.epochs}, Average Loss: {avg_loss:.4f}")

    print("5. Training finished. Saving learned weights...")
    np.save(cfg.irl.output_weights_path, reward_weights.detach().numpy())
    print(f"✅ Learned reward weights saved to: {cfg.irl.output_weights_path}")


if __name__ == "__main__":
    main()
