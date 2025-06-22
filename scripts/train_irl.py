import json

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import FeatureExtractor


def load_expert_trajectories(filepath):
    """エキスパートの行動データをJSONファイルから読み込む。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading expert trajectories from {filepath}: {e}")
        return []


def main():
    """逆強化学習のメイン処理."""

    print("1. Loading configuration...")
    try:
        cfg = OmegaConf.load("configs/base.yaml")
    except FileNotFoundError:
        print(
            "Error: `configs/base.yaml` not found. Please ensure the config file exists."
        )
        return

    print("2. Loading data...")
    try:
        with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
        with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            dev_profiles_data = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(
            f"Error: Data file not found. {e}. Please run pre-processing scripts first."
        )
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    expert_trajectories = load_expert_trajectories(cfg.irl.expert_path)
    if not expert_trajectories:
        print(f"No expert trajectories found at {cfg.irl.expert_path}. Exiting.")
        return

    print("3. Initializing environment and feature extractor...")
    env = OSSSimpleEnv(cfg, backlog_data, dev_profiles_data)
    feature_extractor = FeatureExtractor(cfg)

    feature_dim = len(feature_extractor.feature_names)

    print(f"4. Setting up IRL model with {feature_dim} features...")
    reward_weights = torch.randn(feature_dim, requires_grad=True)
    optimizer = optim.Adam([reward_weights], lr=cfg.irl.learning_rate)

    print("5. Starting training loop...")
    for epoch in range(cfg.irl.epochs):
        total_loss = 0

        for expert_event in expert_trajectories:
            optimizer.zero_grad()
            env.reset()

            # ▼▼▼【ここがエラーの修正箇所】▼▼▼
            # キーを 'developer_id' から 'developer' に修正します。
            developer_id = expert_event.get("developer")
            task_id = expert_event.get("task_id")

            if not developer_id or not task_id:
                continue

            developer_profile = env.developers.get(developer_id)
            expert_task = next((t for t in env.backlog if t.id == task_id), None)

            if not developer_profile or not expert_task:
                continue

            developer_obj_for_feature = {
                "name": developer_id,
                "profile": developer_profile,
            }
            # ▲▲▲【ここまでがエラーの修正箇所】▲▲▲

            expert_features = feature_extractor.get_features(
                expert_task, developer_obj_for_feature, env
            )
            expert_features = torch.from_numpy(expert_features).float()

            other_features_list = []
            for other_task in env.backlog:
                if other_task.id != expert_task.id:
                    features = feature_extractor.get_features(
                        other_task, developer_obj_for_feature, env
                    )
                    other_features_list.append(torch.from_numpy(features).float())

            if not other_features_list:
                continue

            expert_reward = torch.dot(reward_weights, expert_features)
            other_rewards = torch.stack(
                [torch.dot(reward_weights, f) for f in other_features_list]
            )
            log_sum_exp_other_rewards = torch.logsumexp(other_rewards, dim=0)

            loss = -(expert_reward - log_sum_exp_other_rewards)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if len(expert_trajectories) > 0:
            avg_loss = total_loss / len(expert_trajectories)
            print(f"Epoch {epoch + 1}/{cfg.irl.epochs}, Average Loss: {avg_loss:.4f}")

    print("6. Training finished. Saving learned weights...")
    learned_weights = reward_weights.detach().numpy()
    np.save(cfg.irl.output_weights_path, learned_weights)
    print(f"✅ Learned reward weights saved to: {cfg.irl.output_weights_path}")


if __name__ == "__main__":
    main()
