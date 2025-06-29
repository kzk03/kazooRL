#!/usr/bin/env python3
"""
Bot除外済みExpert軌跡を使用したRLトレーニングスクリプト

このスクリプトは：
1. Bot除外済みのexpert軌跡ファイル（expert_trajectories_bot_excluded.pkl）を使用
2. Bot開発者とBot関連タスクを完全に除外した学習を実行
3. 人間のみの開発者・タスクでの純粋なRLモデル学習を行う

使用例:
    python scripts/train_oss_with_bot_excluded_expert.py
"""

import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import GNNFeatureExtractor
from kazoo.learners.independent_ppo_controller import IndependentPPOController


def is_bot_user(username):
    """Check if a username indicates a bot user"""
    if not username:
        return False
    username_lower = username.lower()
    bot_indicators = ['bot', 'stale', 'dependabot', 'renovate', 'greenkeeper']
    return any(indicator in username_lower for indicator in bot_indicators)


def filter_bot_tasks_and_developers(backlog, dev_profiles, expert_trajectories=None):
    """
    Filter out bot tasks and bot developers from data.
    Returns filtered data and statistics.
    """
    print("\n=== Filtering Bot Tasks and Developers ===")
    
    # Filter developer profiles first
    human_dev_profiles = {}
    bot_developers = []
    
    for dev_id, profile in dev_profiles.items():
        if is_bot_user(dev_id):
            bot_developers.append(dev_id)
        else:
            human_dev_profiles[dev_id] = profile
    
    print(f"Original developers: {len(dev_profiles)}")
    print(f"Bot developers filtered out: {len(bot_developers)}")
    print(f"Human developers remaining: {len(human_dev_profiles)}")
    
    if bot_developers:
        print(f"Bot developers: {bot_developers[:5]}...")  # Show first 5
    
    # Get expert task assignments (excluding bots)
    expert_assignments = {}
    if expert_trajectories:
        # Process expert trajectories to extract task assignments
        for episode in expert_trajectories:
            for step in episode:
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    if action_details:
                        task_id = action_details.get('task_id')
                        developer = action_details.get('developer')
                        
                        if task_id and developer and not is_bot_user(developer):
                            expert_assignments[task_id] = developer
    
    print(f"Expert assignments found: {len(expert_assignments)}")
    
    # Filter tasks - remove bot-assigned tasks
    human_backlog = []
    bot_tasks_filtered = 0
    
    for task in backlog:
        assigned_to = task.get('assigned_to', '')
        
        # Check if assigned to bot
        if assigned_to and is_bot_user(assigned_to):
            bot_tasks_filtered += 1
            continue
        
        # Check if developer exists in human profiles
        if assigned_to and assigned_to not in human_dev_profiles:
            bot_tasks_filtered += 1
            continue
        
        human_backlog.append(task)
    
    print(f"Original tasks: {len(backlog)}")
    print(f"Bot tasks filtered out: {bot_tasks_filtered}")
    print(f"Human tasks remaining: {len(human_backlog)}")
    
    # Filter expert trajectories
    human_expert_trajectories = []
    if expert_trajectories:
        for episode in expert_trajectories:
            human_episode = []
            for step in episode:
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    developer = action_details.get('developer', '')
                    
                    # Keep only human developer actions
                    if developer and not is_bot_user(developer) and developer in human_dev_profiles:
                        human_episode.append(step)
            
            if human_episode:
                human_expert_trajectories.append(human_episode)
    
    original_episodes = len(expert_trajectories) if expert_trajectories else 0
    print(f"Original expert trajectory episodes: {original_episodes}")
    print(f"Human expert trajectory episodes remaining: {len(human_expert_trajectories)}")
    
    return human_backlog, human_dev_profiles, human_expert_trajectories, expert_assignments


def split_tasks_by_time(tasks, train_ratio=0.7):
    """Split tasks into train/test by time"""
    # Sort tasks by creation time
    tasks_with_dates = []
    
    for task in tasks:
        created_at = task.get('created_at') or task.get('createdAt')
        if created_at:
            try:
                if isinstance(created_at, str):
                    # Parse ISO format datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    dt = created_at
                tasks_with_dates.append((dt, task))
            except:
                continue
    
    if not tasks_with_dates:
        print("Warning: No tasks with valid dates found")
        return tasks, []
    
    # Sort by date
    tasks_with_dates.sort(key=lambda x: x[0])
    
    # Split by time
    split_idx = int(len(tasks_with_dates) * train_ratio)
    train_tasks = [task for _, task in tasks_with_dates[:split_idx]]
    test_tasks = [task for _, task in tasks_with_dates[split_idx:]]
    
    print(f"Time-based split: {len(train_tasks)} train, {len(test_tasks)} test")
    
    return train_tasks, test_tasks


@hydra.main(version_base="1.3", config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    print("🚀 OSS RL Training with Bot-Excluded Expert Trajectories")
    print("=" * 70)
    
    # 設定の表示
    print("Configuration:")
    print(f"  Backlog: {cfg.env.backlog_path}")
    print(f"  Dev profiles: {cfg.env.dev_profiles_path}")
    print(f"  Bot-excluded expert trajectories: data/expert_trajectories_bot_excluded.pkl")
    print(f"  Total timesteps: {cfg.rl.total_timesteps}")
    print()
    
    # データの読み込み
    print("Loading data...")
    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        full_backlog = json.load(f)
    print(f"Loaded {len(full_backlog)} tasks from backlog")
    
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        full_dev_profiles = yaml.safe_load(f)
    print(f"Loaded {len(full_dev_profiles)} developer profiles")
    
    # Bot除外済みexpert軌跡の読み込み
    expert_trajectories_path = "data/expert_trajectories_bot_excluded.pkl"
    expert_trajectories = []
    
    if os.path.exists(expert_trajectories_path):
        try:
            with open(expert_trajectories_path, "rb") as f:
                expert_trajectories = pickle.load(f)
            print(f"Loaded {len(expert_trajectories)} bot-excluded expert trajectory episodes")
            
            # 軌跡の詳細を表示
            total_steps = sum(len(episode) for episode in expert_trajectories)
            print(f"Total expert steps: {total_steps}")
            
            if expert_trajectories and expert_trajectories[0]:
                sample_step = expert_trajectories[0][0]
                if 'action_details' in sample_step:
                    developer = sample_step['action_details'].get('developer', 'Unknown')
                    action = sample_step.get('action', 'Unknown')
                    print(f"Sample expert action: {developer} -> {action}")
            
        except Exception as e:
            print(f"Error loading expert trajectories: {e}")
            expert_trajectories = []
    else:
        print(f"Warning: Bot-excluded expert trajectories file not found: {expert_trajectories_path}")
        print("You can create it by running: python tools/analysis/create_expert_trajectories_bot_excluded.py")
    
    # Bot除外処理
    human_backlog, human_dev_profiles, human_expert_trajectories, expert_assignments = filter_bot_tasks_and_developers(
        full_backlog, full_dev_profiles, expert_trajectories
    )
    
    # 時系列分割
    train_tasks, test_tasks = split_tasks_by_time(human_backlog, train_ratio=0.7)
    
    # 訓練用エキスパート割り当て
    train_task_ids = {task['id'] for task in train_tasks}
    train_expert_assignments = {
        task_id: dev for task_id, dev in expert_assignments.items()
        if task_id in train_task_ids
    }
    
    print(f"\nTraining data summary:")
    print(f"  Train tasks: {len(train_tasks)}")
    print(f"  Test tasks: {len(test_tasks)}")
    print(f"  Human developers: {len(human_dev_profiles)}")
    print(f"  Train expert assignments: {len(train_expert_assignments)}")
    print(f"  Human expert trajectory episodes: {len(human_expert_trajectories)}")
    
    # GNN特徴抽出器の初期化（必要に応じて）
    feature_extractor = None
    if cfg.get("irl", {}).get("use_gnn", False):
        try:
            feature_extractor = GNNFeatureExtractor(
                model_path=cfg.env.gnn.model_path,
                graph_data_path=cfg.env.gnn.graph_data_path,
                device='cpu'
            )
            print("GNN feature extractor initialized")
        except Exception as e:
            print(f"Warning: Could not initialize GNN feature extractor: {e}")
    
    # 環境の初期化（訓練用タスクと人間開発者のみ）
    print("\nInitializing training environment...")
    env = OSSSimpleEnv(cfg, train_tasks, human_dev_profiles)
    print(f"Environment initialized with {len(train_tasks)} tasks and {len(human_dev_profiles)} human developers")
    
    # Get space dimensions from first agent (since all agents have same spaces)
    first_agent = list(env.agent_ids)[0]
    obs_shape = env.observation_space[first_agent].shape
    action_space_n = env.action_space[first_agent].n
    
    print(f"Observation space shape: {obs_shape}")
    print(f"Action space size: {action_space_n}")
    print(f"Agent IDs: {env.agent_ids}")
    
    # PPOコントローラーの初期化
    print("\nInitializing PPO controller...")
    controller = IndependentPPOController(env, cfg)
    print(f"PPO controller initialized for {len(env.agent_ids)} agents")
    
    # トレーニングの実行
    print(f"\n🏋️  Starting RL training...")
    print(f"Total timesteps: {cfg.rl.total_timesteps}")
    print(f"Using bot-excluded expert trajectories: {len(human_expert_trajectories)} episodes")
    
    try:
        controller.learn(total_timesteps=cfg.rl.total_timesteps)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    # モデルの保存
    model_dir = cfg.rl.get("output_model_dir", "models/")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n💾 Saving models to {model_dir}")
    for agent_id in controller.agent_ids:
        model_path = os.path.join(model_dir, f"ppo_agent_{agent_id}.pth")
        controller.agents[agent_id].save(model_path)
        print(f"Saved model for {agent_id}: {model_path}")
    
    # メタデータの保存
    metadata = {
        'training_completed': True,
        'total_timesteps': cfg.rl.total_timesteps,
        'human_developers': list(human_dev_profiles.keys()),
        'train_tasks_count': len(train_tasks),
        'test_tasks_count': len(test_tasks),
        'train_expert_assignments_count': len(train_expert_assignments),
        'human_expert_episodes': len(human_expert_trajectories),
        'bot_excluded': True,
        'expert_trajectories_file': expert_trajectories_path,
        'train_ratio': 0.7
    }
    
    metadata_path = os.path.join(model_dir, "training_metadata_bot_excluded.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved training metadata: {metadata_path}")
    
    print(f"\n🎯 Training Summary:")
    print(f"  Models saved for {len(controller.agent_ids)} human developers")
    print(f"  Training data: {len(train_tasks)} tasks")
    print(f"  Expert data: {len(human_expert_trajectories)} episodes with bot exclusion")
    print(f"  Bot developers excluded: ✅")
    print(f"  All models saved to: {model_dir}")
    
    print("\n✅ Bot-excluded RL training completed successfully!")


if __name__ == "__main__":
    main()
