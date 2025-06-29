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
    bot_task_count = 0
    
    if expert_trajectories:
        for trajectory_episode in expert_trajectories:
            if isinstance(trajectory_episode, list):
                for step in trajectory_episode:
                    if isinstance(step, dict) and 'action_details' in step:
                        action_details = step['action_details']
                        task_id = action_details.get('task_id')
                        developer = action_details.get('developer')
                        
                        if task_id and developer:
                            if is_bot_user(developer):
                                bot_task_count += 1
                                continue
                            expert_assignments[task_id] = developer
    
    print(f"Expert assignments found: {len(expert_assignments)}")
    print(f"Bot expert assignments excluded: {bot_task_count}")
    
    # Filter backlog tasks
    human_tasks = []
    bot_tasks = []
    
    for task in backlog:
        task_id = task.get('id')
        assigned_to = task.get('assigned_to')
        
        # Check if task is assigned to a bot in the task data
        if assigned_to and is_bot_user(assigned_to):
            bot_tasks.append(task)
            continue
            
        # Check if task has expert assignment to a bot
        expert_dev = expert_assignments.get(task_id)
        if expert_dev and is_bot_user(expert_dev):
            bot_tasks.append(task)
            continue
            
        human_tasks.append(task)
    
    print(f"Original tasks: {len(backlog)}")
    print(f"Bot tasks filtered out: {len(bot_tasks)}")
    print(f"Human tasks remaining: {len(human_tasks)}")
    
    # Filter expert trajectories if provided
    filtered_trajectories = None
    if expert_trajectories:
        filtered_trajectories = []
        for trajectory_episode in expert_trajectories:
            if isinstance(trajectory_episode, list):
                filtered_episode = []
                for step in trajectory_episode:
                    if isinstance(step, dict) and 'action_details' in step:
                        action_details = step['action_details']
                        task_id = action_details.get('task_id')
                        developer = action_details.get('developer')
                        
                        # Skip if developer is a bot
                        if developer and is_bot_user(developer):
                            continue
                            
                        # Skip if task is not in human tasks
                        if not any(task.get('id') == task_id for task in human_tasks):
                            continue
                            
                        filtered_episode.append(step)
                        
                if filtered_episode:
                    filtered_trajectories.append(filtered_episode)
        
        print(f"Original expert trajectory episodes: {len(expert_trajectories)}")
        print(f"Human expert trajectory episodes remaining: {len(filtered_trajectories)}")
    
    return human_tasks, human_dev_profiles, filtered_trajectories


def split_tasks_by_time(tasks, train_ratio=0.7):
    """
    Split tasks into train/test based on creation time.
    """
    # Filter tasks with valid dates
    tasks_with_dates = []
    for task in tasks:
        created_at = task.get('created_at')
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
    
    if train_tasks and test_tasks:
        train_start = tasks_with_dates[0][0].strftime('%Y-%m-%d')
        train_end = tasks_with_dates[split_idx-1][0].strftime('%Y-%m-%d')
        test_start = tasks_with_dates[split_idx][0].strftime('%Y-%m-%d')
        test_end = tasks_with_dates[-1][0].strftime('%Y-%m-%d')
        
        print(f"Train period: {train_start} to {train_end}")
        print(f"Test period: {test_start} to {test_end}")
    
    return train_tasks, test_tasks


def filter_bot_tasks(backlog: List[Dict], expert_trajectories: List, train_ratio: float = 0.7) -> tuple:
    """
    botタスクを除外し、時系列順にトレイン・テスト分割を行う
    
    Args:
        backlog: バックログタスクのリスト
        expert_trajectories: エキスパート軌跡データ
        train_ratio: 学習データの割合
    
    Returns:
        (train_backlog, test_backlog, train_assignments, test_assignments)
    """
    # エキスパートデータからbotタスクを除外
    expert_task_ids = set()
    bot_task_count = 0
    
    for trajectory_episode in expert_trajectories:
        for step in trajectory_episode:
            if isinstance(step, dict) and 'action_details' in step:
                task_id = step['action_details'].get('task_id')
                assigned_dev = step['action_details'].get('developer')
                if task_id and assigned_dev:
                    # botタスクを除外
                    if 'bot' in assigned_dev.lower() or assigned_dev == 'stale[bot]':
                        bot_task_count += 1
                        continue
                    expert_task_ids.add(task_id)
    
    print(f"Excluded {bot_task_count} bot tasks from training data")
    
    # エキスパートデータがある人間タスクのみをフィルタリング
    human_tasks = []
    for task in backlog:
        if task['id'] in expert_task_ids and 'created_at' in task:
            task['created_at_dt'] = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
            human_tasks.append(task)
    
    # 時系列順にソート
    human_tasks_sorted = sorted(human_tasks, key=lambda x: x['created_at_dt'])
    
    # 分割点を計算
    split_index = int(len(human_tasks_sorted) * train_ratio)
    
    train_tasks = human_tasks_sorted[:split_index]
    test_tasks = human_tasks_sorted[split_index:]
    
    # created_at_dtフィールドを削除
    train_backlog = []
    test_backlog = []
    
    for task in train_tasks:
        task_copy = task.copy()
        if 'created_at_dt' in task_copy:
            del task_copy['created_at_dt']
        train_backlog.append(task_copy)
    
    for task in test_tasks:
        task_copy = task.copy()
        if 'created_at_dt' in task_copy:
            del task_copy['created_at_dt']
        test_backlog.append(task_copy)
    
    # エキスパート割り当ての分割
    train_task_ids = {task['id'] for task in train_tasks}
    test_task_ids = {task['id'] for task in test_tasks}
    
    train_assignments = {}
    test_assignments = {}
    
    for trajectory_episode in expert_trajectories:
        for step in trajectory_episode:
            if isinstance(step, dict) and 'action_details' in step:
                action_details = step['action_details']
                task_id = action_details.get('task_id')
                assigned_dev = action_details.get('developer')
                if task_id and assigned_dev:
                    # botタスクを除外
                    if 'bot' in assigned_dev.lower() or assigned_dev == 'stale[bot]':
                        continue
                    if task_id in train_task_ids:
                        train_assignments[task_id] = assigned_dev
                    elif task_id in test_task_ids:
                        test_assignments[task_id] = assigned_dev
    
    print(f"Human task split results:")
    print(f"  Total human tasks: {len(human_tasks_sorted)}")
    print(f"  Train tasks: {len(train_backlog)} (up to {train_tasks[-1]['created_at'] if train_tasks else 'N/A'})")
    print(f"  Test tasks: {len(test_backlog)} (from {test_tasks[0]['created_at'] if test_tasks else 'N/A'})")
    print(f"  Train assignments: {len(train_assignments)}")
    print(f"  Test assignments: {len(test_assignments)}")
    
    return train_backlog, test_backlog, train_assignments, test_assignments


def filter_bot_developers(dev_profiles: Dict) -> Dict:
    """
    開発者プロファイルからbotを除外
    
    Args:
        dev_profiles: 開発者プロファイル辞書
    
    Returns:
        botを除外した開発者プロファイル辞書
    """
    human_profiles = {}
    bot_count = 0
    
    for dev_name, profile in dev_profiles.items():
        if 'bot' in dev_name.lower() or dev_name == 'stale[bot]':
            bot_count += 1
            continue
        human_profiles[dev_name] = profile
    
    print(f"Excluded {bot_count} bot developers from profiles")
    print(f"Human developers remaining: {len(human_profiles)}")
    
    return human_profiles


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    Train RL agents using only human tasks and developers (bots excluded).
    """
    print(OmegaConf.to_yaml(cfg))
    print("\n=== Training RL Model with Bot Exclusion ===")
    
    # 1. Load all data
    print("1. Loading data...")
    
    if not os.path.exists(cfg.env.backlog_path):
        print(f"Error: Backlog file not found: {cfg.env.backlog_path}")
        return

    if not os.path.exists(cfg.env.dev_profiles_path):
        print(f"Error: Dev profiles file not found: {cfg.env.dev_profiles_path}")
        return

    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)
    
    # Load expert trajectories if available
    expert_trajectories = None
    expert_path = cfg.env.get('expert_trajectories_path', cfg.irl.get('expert_path', 'data/expert_trajectories.pkl'))
    if os.path.exists(expert_path):
        with open(expert_path, 'rb') as f:
            expert_trajectories = pickle.load(f)
        print(f"Loaded {len(expert_trajectories)} expert trajectories")
    
    print(f"Original data: {len(backlog)} tasks, {len(dev_profiles)} developers")
    
    # 2. Filter out bots using new comprehensive filtering
    human_tasks, human_dev_profiles, filtered_trajectories = filter_bot_tasks_and_developers(
        backlog, dev_profiles, expert_trajectories
    )
    
    if len(human_tasks) == 0:
        print("Error: No human tasks found after filtering")
        return
        
    if len(human_dev_profiles) == 0:
        print("Error: No human developers found after filtering")
        return
    
    # 3. Split human tasks by time for training
    train_tasks, test_tasks = split_tasks_by_time(human_tasks, train_ratio=0.7)
    
    if len(train_tasks) == 0:
        print("Error: No training tasks after split")
        return
    
    print(f"\nFinal training data: {len(train_tasks)} tasks, {len(human_dev_profiles)} developers")
    
    # 4. Save metadata about the split for evaluation
    metadata = {
        'train_task_ids': [task['id'] for task in train_tasks],
        'test_task_ids': [task['id'] for task in test_tasks],
        'human_developer_ids': list(human_dev_profiles.keys()),
        'training_stats': {
            'original_tasks': len(backlog),
            'human_tasks': len(human_tasks),
            'train_tasks': len(train_tasks),
            'test_tasks': len(test_tasks),
            'original_developers': len(dev_profiles),
            'human_developers': len(human_dev_profiles)
        }
    }
    
    metadata_path = os.path.join(cfg.rl.output_model_dir, 'training_metadata.json')
    os.makedirs(cfg.rl.output_model_dir, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved training metadata to: {metadata_path}")
    
    # 5. Update config to use only human developers
    cfg.num_developers = len(human_dev_profiles)
    print(f"Updated num_developers to: {cfg.num_developers}")
    
    # 6. Initialize environment with filtered data
    print("2. Initializing environment with human-only data...")
    env = OSSSimpleEnv(
        config=cfg,
        backlog=train_tasks,  # Use only training tasks
        dev_profiles=human_dev_profiles,  # Use only human developers
        reward_weights_path=cfg.irl.output_weights_path,
    )
    
    print(f"Environment initialized with {len(train_tasks)} tasks and {len(human_dev_profiles)} developers")
    
    # 7. Initialize RL controller
    print("3. Setting up PPO controller...")
    controller = IndependentPPOController(env=env, config=cfg)
    
    # 8. Start training
    print("4. Starting RL training...")
    
    try:
        total_timesteps = cfg.rl.total_timesteps
    except Exception:
        raise ValueError("Config file must have 'rl.total_timesteps' defined.")
    
    print(f"Training for {total_timesteps} timesteps...")
    controller.learn(total_timesteps=total_timesteps)
    
    # 9. Save models
    print("5. Training finished. Saving RL agent models...")
    controller.save_models(cfg.rl.output_model_dir)
    print(f"✅ RL models saved to: {cfg.rl.output_model_dir}")
    
    # 10. Print final summary
    print("\n=== Training Summary ===")
    print(f"✅ Successfully trained on {len(train_tasks)} human tasks")
    print(f"✅ Used {len(human_dev_profiles)} human developers")
    print(f"✅ Models saved to: {cfg.rl.output_model_dir}")
    print(f"✅ Metadata saved to: {metadata_path}")
    print(f"✅ Test set available: {len(test_tasks)} tasks for evaluation")


if __name__ == "__main__":
    main()
