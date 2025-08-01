#!/usr/bin/env python3
"""
改良版IRL訓練スクリプト
ボット除外データに対応した効率的な逆強化学習訓練
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.envs.task import Task
from kazoo.features.feature_extractor import FeatureExtractor


class EarlyStopping:
    """早期停止クラス"""

    def __init__(self, patience=30, min_improvement=0.0001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, loss, weights):
        if loss < self.best_loss - self.min_improvement:
            self.best_loss = loss
            self.counter = 0
            self.best_weights = weights.clone().detach()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def validate_data_integrity(expert_steps, all_tasks_db, dev_profiles_data):
    """データの整合性を検証"""
    print("🔍 Validating data integrity...")

    valid_steps = 0
    missing_devs = set()
    missing_tasks = set()

    for step in expert_steps[:100]:  # サンプル検証
        action_details = step["action_details"]
        developer_id = action_details.get("developer")
        task_id = action_details.get("task_id")

        if developer_id not in dev_profiles_data:
            missing_devs.add(developer_id)
        if task_id not in all_tasks_db:
            missing_tasks.add(task_id)
        if developer_id in dev_profiles_data and task_id in all_tasks_db:
            valid_steps += 1

    print(f"   ✅ Valid steps: {valid_steps}/100 ({valid_steps}%)")
    if missing_devs:
        print(f"   ⚠️  Missing developers: {len(missing_devs)}")
    if missing_tasks:
        print(f"   ⚠️  Missing tasks: {len(missing_tasks)}")

    return valid_steps >= 50  # 50%以上有効なら続行


def process_expert_step(
    step_data,
    all_tasks_db,
    dev_profiles_data,
    feature_extractor,
    env,
    reward_weights,
    device,
):
    """エキスパートステップの処理"""
    state = step_data["state"]
    action_details = step_data["action_details"]

    developer_id = action_details.get("developer")
    expert_task_id = action_details.get("task_id")
    timestamp_str = action_details.get("timestamp")

    # データ検証
    if not all([developer_id, expert_task_id, timestamp_str]):
        return None

    developer_profile = dev_profiles_data.get(developer_id)
    expert_task = all_tasks_db.get(expert_task_id)

    if not developer_profile or not expert_task:
        return None

    try:
        event_timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        env.current_time = event_timestamp

        developer_obj = {"name": developer_id, "profile": developer_profile}

        # エキスパート行動の特徴量
        expert_features = feature_extractor.get_features(
            expert_task, developer_obj, env
        )
        if expert_features is None:
            return None

        expert_features = torch.from_numpy(expert_features).float().to(device)

        # 他の選択肢の特徴量
        other_features_list = []
        for other_task_id in state["open_task_ids"]:
            if other_task_id != expert_task_id:
                other_task = all_tasks_db.get(other_task_id)
                if other_task:
                    features = feature_extractor.get_features(
                        other_task, developer_obj, env
                    )
                    if features is not None:
                        other_features_list.append(
                            torch.from_numpy(features).float().to(device)
                        )

        if len(other_features_list) == 0:
            return None

        # 損失計算
        expert_reward = torch.dot(reward_weights, expert_features)
        other_rewards = torch.stack(
            [torch.dot(reward_weights, f) for f in other_features_list]
        )
        log_sum_exp_other = torch.logsumexp(other_rewards, dim=0)

        loss = -(expert_reward - log_sum_exp_other)
        return loss

    except Exception as e:
        return None


def main(config_path="configs/bot_excluded_production.yaml"):
    """改良された逆強化学習のメイン処理"""

    print("🚀 Bot-Excluded IRL Training Started")
    print("=" * 60)
    print(f"📁 Config: {config_path}")

    # 設定読み込み
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return

    # 出力ディレクトリ作成
    output_dir = Path(cfg.irl.output_weights_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("📊 Loading training data...")
    try:
        with open(cfg.irl.expert_path, "rb") as f:
            trajectories = pickle.load(f)
            expert_trajectory_steps = trajectories[0] if trajectories else []
        print(f"   ✅ Expert trajectories: {len(expert_trajectory_steps)} steps")
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"   ❌ Error loading expert trajectories: {e}")
        return

    if not expert_trajectory_steps:
        print("   ❌ No expert steps found!")
        return

    try:
        with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
        with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            dev_profiles_data = yaml.safe_load(f)
        print(f"   ✅ Backlog tasks: {len(backlog_data)}")
        print(f"   ✅ Developer profiles: {len(dev_profiles_data)}")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return

    # 環境・特徴量抽出器の初期化
    print("🔧 Initializing environment and feature extractor...")
    try:
        env = OSSSimpleEnv(cfg, backlog_data, dev_profiles_data)
        feature_extractor = FeatureExtractor(cfg)
        feature_dim = len(feature_extractor.feature_names)
        all_tasks_db = {task.id: task for task in env.backlog}
        print(f"   ✅ Feature dimension: {feature_dim}")
        print(f"   ✅ Tasks in database: {len(all_tasks_db)}")
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
        return

    # データ整合性検証
    if not validate_data_integrity(
        expert_trajectory_steps, all_tasks_db, dev_profiles_data
    ):
        print("   ❌ Data integrity check failed!")
        return

    # IRL モデルの設定
    print("🧠 Setting up IRL model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   📱 Device: {device}")

    # 重みの初期化
    reward_weights = torch.randn(feature_dim, requires_grad=True, device=device)
    torch.nn.init.xavier_uniform_(reward_weights.unsqueeze(0))

    # オプティマイザーの設定
    optimizer = optim.Adam(
        [reward_weights], lr=cfg.irl.learning_rate, weight_decay=1e-5
    )

    # 学習率スケジューラー
    scheduler = None
    if cfg.irl.get("lr_scheduler", {}).get("enabled", False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.irl.epochs,
            eta_min=cfg.irl.lr_scheduler.get("min_lr", 1e-6),
        )
        print(f"   ✅ Learning rate scheduler enabled")

    # 早期停止の設定
    early_stopping = None
    if cfg.irl.get("early_stopping", {}).get("enabled", False):
        early_stopping = EarlyStopping(
            patience=cfg.irl.early_stopping.get("patience", 30),
            min_improvement=cfg.irl.early_stopping.get("min_improvement", 0.0001),
        )
        print(f"   ✅ Early stopping enabled (patience: {early_stopping.patience})")

    # 訓練ループ
    print(f"🎯 Starting training for {cfg.irl.epochs} epochs...")
    print("=" * 60)

    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(cfg.irl.epochs):
        epoch_start = time.time()
        total_loss = 0.0
        valid_steps = 0

        # プログレスバー
        pbar = tqdm(
            expert_trajectory_steps,
            desc=f"Epoch {epoch+1:3d}/{cfg.irl.epochs}",
            ncols=100,
            leave=False,
        )

        for step_data in pbar:
            optimizer.zero_grad()

            loss = process_expert_step(
                step_data,
                all_tasks_db,
                dev_profiles_data,
                feature_extractor,
                env,
                reward_weights,
                device,
            )

            if loss is not None:
                total_loss += loss.item()
                valid_steps += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_([reward_weights], max_norm=1.0)
                optimizer.step()

                # プログレスバー更新
                if valid_steps % 10 == 0:
                    avg_loss = total_loss / valid_steps
                    pbar.set_postfix(
                        {"Loss": f"{avg_loss:.4f}", "Valid": f"{valid_steps}"}
                    )

        # エポック終了処理
        epoch_time = time.time() - epoch_start

        if valid_steps > 0:
            avg_loss = total_loss / valid_steps

            print(
                f"Epoch {epoch+1:3d}/{cfg.irl.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Valid: {valid_steps:4d}/{len(expert_trajectory_steps)} | "
                f"Time: {epoch_time:.1f}s"
            )

            # ベストモデル更新
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"   🎉 New best loss: {best_loss:.4f}")

            # 学習率スケジューラー更新
            if scheduler:
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"   📈 Learning rate: {current_lr:.6f}")

            # 早期停止チェック
            if early_stopping and early_stopping(avg_loss, reward_weights):
                print(f"   🛑 Early stopping triggered at epoch {epoch+1}")
                if early_stopping.best_weights is not None:
                    reward_weights.data = early_stopping.best_weights
                break
        else:
            print(f"Epoch {epoch+1:3d}/{cfg.irl.epochs} | No valid steps found!")

    # 訓練完了
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"🎉 Training completed in {total_time:.1f}s")
    print(f"   Best loss: {best_loss:.4f}")

    # 重みの保存
    print("💾 Saving learned weights...")
    weights_np = reward_weights.detach().cpu().numpy()
    np.save(cfg.irl.output_weights_path, weights_np)

    print(f"✅ Learned reward weights saved to: {cfg.irl.output_weights_path}")
    print(f"   Weight shape: {weights_np.shape}")
    print(
        f"   Weight stats: min={weights_np.min():.4f}, max={weights_np.max():.4f}, mean={weights_np.mean():.4f}"
    )

    # 重みの分析
    print("\n📊 Feature importance analysis:")
    feature_names = feature_extractor.feature_names
    weights_abs = np.abs(weights_np)
    top_indices = np.argsort(weights_abs)[-5:][::-1]

    for i, idx in enumerate(top_indices):
        print(f"   {i+1}. {feature_names[idx]}: {weights_np[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IRL model with bot exclusion")
    parser.add_argument(
        "--config",
        default="configs/bot_excluded_production.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    main(args.config)
