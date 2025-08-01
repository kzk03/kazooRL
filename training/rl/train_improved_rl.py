#!/usr/bin/env python3
"""
改良されたRL訓練スクリプト
PPO 0%精度問題を解決するための統合訓練システム
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# パッケージのパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from kazoo.envs.improved_oss_env import ImprovedOSSEnv
from kazoo.learners.improved_ppo_agent import ImprovedPPOAgent


class ImprovedRLTrainer:
    """改良されたRL訓練器"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # データの読み込み
        self._load_data()
        
        # 環境の初期化
        self._setup_environment()
        
        # エージェントの初期化
        self._setup_agents()
        
        # 統計情報
        self.training_stats = {
            'episode_rewards': defaultdict(list),
            'episode_lengths': [],
            'task_completion_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'learning_rates': []
        }
    
    def _load_data(self):
        """データの読み込み"""
        print("1. Loading data...")
        
        # ファイル存在確認
        if not os.path.exists(self.config.env.backlog_path):
            raise FileNotFoundError(f"Backlog file not found: {self.config.env.backlog_path}")
        
        if not os.path.exists(self.config.env.dev_profiles_path):
            raise FileNotFoundError(f"Dev profiles file not found: {self.config.env.dev_profiles_path}")
        
        # データ読み込み
        with open(self.config.env.backlog_path, "r", encoding="utf-8") as f:
            self.backlog = json.load(f)
        
        with open(self.config.env.dev_profiles_path, "r", encoding="utf-8") as f:
            self.dev_profiles = yaml.safe_load(f)
        
        print(f"✅ Loaded {len(self.backlog)} tasks and {len(self.dev_profiles)} developer profiles")
    
    def _setup_environment(self):
        """環境の設定"""
        print("2. Setting up improved environment...")
        
        self.env = ImprovedOSSEnv(
            config=self.config,
            backlog=self.backlog,
            dev_profiles=self.dev_profiles,
            reward_weights_path=self.config.irl.get('output_weights_path')
        )
        
        print(f"✅ Environment initialized with {len(self.env.agent_ids)} agents")
    
    def _setup_agents(self):
        """エージェントの設定"""
        print("3. Setting up improved PPO agents...")
        
        self.agents = {}
        obs_dim = self.config.get('processed_feature_dim', 64)
        
        for agent_id in self.env.agent_ids:
            agent_config = {
                **self.config.rl,
                **self.config.get('network', {}),
                'total_timesteps': self.config.rl.total_timesteps
            }
            
            self.agents[agent_id] = ImprovedPPOAgent(
                obs_dim=obs_dim,
                act_space=self.env.action_space[agent_id],
                config=agent_config,
                device=str(self.device)
            )
        
        print(f"✅ Initialized {len(self.agents)} improved PPO agents")
    
    def train(self):
        """訓練の実行"""
        print("4. Starting improved RL training...")
        
        total_timesteps = self.config.rl.total_timesteps
        rollout_len = self.config.rl.rollout_len
        eval_frequency = self.config.rl.get('eval_frequency', 10000)
        save_frequency = self.config.rl.get('save_frequency', 50000)
        
        # 訓練ループ
        global_step = 0
        episode = 0
        
        # 進捗バー
        pbar = tqdm(total=total_timesteps, desc="Training Progress")
        
        while global_step < total_timesteps:
            episode += 1
            episode_rewards = defaultdict(float)
            episode_length = 0
            
            # エピソードの実行
            obs, info = self.env.reset()
            
            # ロールアウト収集
            rollout_data = self._collect_rollout(obs, rollout_len)
            
            # エージェント更新
            update_stats = self._update_agents(rollout_data)
            
            # 統計情報の更新
            global_step += rollout_len * len(self.env.agent_ids)
            episode_length = rollout_len
            
            for agent_id, rewards in rollout_data['rewards'].items():
                episode_rewards[agent_id] = sum(rewards)
            
            self._update_training_stats(episode_rewards, episode_length, update_stats)
            
            # 進捗バーの更新
            avg_reward = np.mean(list(episode_rewards.values())) if episode_rewards else 0
            pbar.set_postfix({
                'Episode': episode,
                'Avg_Reward': f'{avg_reward:.3f}',
                'Task_Completions': self.env.episode_stats['task_completions']
            })
            pbar.update(rollout_len * len(self.env.agent_ids))
            
            # 定期評価
            if global_step % eval_frequency == 0:
                self._evaluate_agents(global_step)
            
            # 定期保存
            if global_step % save_frequency == 0:
                self._save_models(global_step)
            
            # ログ出力
            if episode % self.config.rl.get('log_interval', 10) == 0:
                self._log_training_progress(episode, global_step, episode_rewards)
        
        pbar.close()
        
        # 最終保存
        self._save_models(global_step, final=True)
        
        print("🎉 Training completed!")
        self._print_final_stats()
    
    def _collect_rollout(self, initial_obs, rollout_len):
        """ロールアウトデータの収集"""
        rollout_data = {
            'observations': defaultdict(list),
            'actions': defaultdict(list),
            'log_probs': defaultdict(list),
            'rewards': defaultdict(list),
            'values': defaultdict(list),
            'dones': defaultdict(list)
        }
        
        obs = initial_obs
        
        for step in range(rollout_len):
            actions = {}
            log_probs = {}
            values = {}
            
            # 各エージェントの行動選択
            for agent_id, agent_obs in obs.items():
                action, log_prob, _, value = self.agents[agent_id].get_action_and_value(agent_obs)
                actions[agent_id] = action.item() if hasattr(action, 'item') else action
                log_probs[agent_id] = log_prob
                values[agent_id] = value
            
            # 環境ステップ
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            
            # データの記録
            for agent_id in self.env.agent_ids:
                rollout_data['observations'][agent_id].append(obs[agent_id])
                rollout_data['actions'][agent_id].append(actions[agent_id])
                rollout_data['log_probs'][agent_id].append(log_probs[agent_id])
                rollout_data['rewards'][agent_id].append(rewards[agent_id])
                rollout_data['values'][agent_id].append(values[agent_id])
                rollout_data['dones'][agent_id].append(terminateds[agent_id] or truncateds[agent_id])
            
            obs = next_obs
            
            # エピソード終了チェック
            if any(terminateds.values()) or any(truncateds.values()):
                obs, _ = self.env.reset()
        
        return rollout_data
    
    def _update_agents(self, rollout_data):
        """エージェントの更新"""
        update_stats = defaultdict(dict)
        
        for agent_id in self.env.agent_ids:
            # ロールアウトデータの準備
            agent_rollout = self._prepare_agent_rollout(rollout_data, agent_id)
            
            # エージェント更新
            stats = self.agents[agent_id].update(agent_rollout)
            update_stats[agent_id] = stats
        
        return update_stats
    
    def _prepare_agent_rollout(self, rollout_data, agent_id):
        """エージェント用ロールアウトデータの準備"""
        class RolloutBuffer:
            def __init__(self):
                pass
        
        buffer = RolloutBuffer()
        
        # データの変換
        observations = torch.stack([torch.FloatTensor(obs) for obs in rollout_data['observations'][agent_id]])
        actions = torch.LongTensor(rollout_data['actions'][agent_id])
        log_probs = torch.stack(rollout_data['log_probs'][agent_id])
        rewards = torch.FloatTensor(rollout_data['rewards'][agent_id])
        values = torch.stack(rollout_data['values'][agent_id])
        dones = torch.FloatTensor(rollout_data['dones'][agent_id])
        
        # GAEの計算
        returns, advantages = self._compute_gae(rewards, values, dones)
        
        # バッファに設定
        buffer.obs = observations
        buffer.actions = actions.unsqueeze(-1)
        buffer.log_probs = log_probs.unsqueeze(-1)
        buffer.returns = returns.unsqueeze(-1)
        buffer.advantages = advantages.unsqueeze(-1)
        
        return buffer
    
    def _compute_gae(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """GAE (Generalized Advantage Estimation) の計算"""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0  # 最後のステップ
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        return returns, advantages
    
    def _evaluate_agents(self, global_step):
        """エージェントの評価"""
        print(f"\n📊 Evaluation at step {global_step}")
        
        # 評価エピソードの実行
        eval_episodes = self.config.rl.get('eval_episodes', 5)
        eval_rewards = defaultdict(list)
        eval_task_completions = []
        
        for episode in range(eval_episodes):
            obs, _ = self.env.reset()
            episode_rewards = defaultdict(float)
            
            for step in range(100):  # 最大100ステップ
                actions = {}
                
                # 決定論的行動選択
                for agent_id, agent_obs in obs.items():
                    action, _, _, _ = self.agents[agent_id].get_action_and_value(
                        agent_obs, deterministic=True
                    )
                    actions[agent_id] = action.item() if hasattr(action, 'item') else action
                
                obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
                
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                if any(terminateds.values()) or any(truncateds.values()):
                    break
            
            for agent_id, reward in episode_rewards.items():
                eval_rewards[agent_id].append(reward)
            
            eval_task_completions.append(self.env.episode_stats['task_completions'])
        
        # 評価結果の表示
        avg_rewards = {agent_id: np.mean(rewards) for agent_id, rewards in eval_rewards.items()}
        avg_task_completions = np.mean(eval_task_completions)
        
        print(f"  Average Reward: {np.mean(list(avg_rewards.values())):.3f}")
        print(f"  Average Task Completions: {avg_task_completions:.1f}")
    
    def _save_models(self, global_step, final=False):
        """モデルの保存"""
        save_dir = Path(self.config.rl.output_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            save_path = save_dir / "final_models"
        else:
            save_path = save_dir / f"checkpoint_{global_step}"
        
        save_path.mkdir(exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            model_path = save_path / f"agent_{agent_id}.pth"
            agent.save(str(model_path))
        
        # 統計情報の保存
        stats_path = save_path / "training_stats.json"
        with open(stats_path, 'w') as f:
            # NumPy配列をリストに変換
            serializable_stats = {}
            for key, value in self.training_stats.items():
                if isinstance(value, dict):
                    serializable_stats[key] = {k: v for k, v in value.items()}
                else:
                    serializable_stats[key] = value
            
            json.dump(serializable_stats, f, indent=2)
        
        print(f"💾 Models saved to {save_path}")
    
    def _update_training_stats(self, episode_rewards, episode_length, update_stats):
        """訓練統計の更新"""
        for agent_id, reward in episode_rewards.items():
            self.training_stats['episode_rewards'][agent_id].append(reward)
        
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['task_completion_rates'].append(
            self.env.episode_stats['task_completions']
        )
        
        # エージェント更新統計の平均
        if update_stats:
            avg_policy_loss = np.mean([stats.get('policy_loss', 0) for stats in update_stats.values()])
            avg_value_loss = np.mean([stats.get('value_loss', 0) for stats in update_stats.values()])
            avg_lr = np.mean([stats.get('learning_rate', 0) for stats in update_stats.values()])
            
            self.training_stats['policy_losses'].append(avg_policy_loss)
            self.training_stats['value_losses'].append(avg_value_loss)
            self.training_stats['learning_rates'].append(avg_lr)
    
    def _log_training_progress(self, episode, global_step, episode_rewards):
        """訓練進捗のログ出力"""
        avg_reward = np.mean(list(episode_rewards.values())) if episode_rewards else 0
        task_completions = self.env.episode_stats['task_completions']
        
        print(f"Episode {episode:4d} | Step {global_step:6d} | "
              f"Avg Reward: {avg_reward:6.3f} | "
              f"Task Completions: {task_completions:2d}")
    
    def _print_final_stats(self):
        """最終統計の表示"""
        print("\n" + "="*60)
        print("🎯 Final Training Statistics")
        print("="*60)
        
        if self.training_stats['episode_rewards']:
            all_rewards = []
            for agent_rewards in self.training_stats['episode_rewards'].values():
                all_rewards.extend(agent_rewards)
            
            if all_rewards:
                print(f"Average Episode Reward: {np.mean(all_rewards):.3f}")
                print(f"Max Episode Reward: {np.max(all_rewards):.3f}")
                print(f"Min Episode Reward: {np.min(all_rewards):.3f}")
        
        if self.training_stats['task_completion_rates']:
            avg_completions = np.mean(self.training_stats['task_completion_rates'])
            print(f"Average Task Completions per Episode: {avg_completions:.1f}")
        
        print("="*60)


@hydra.main(config_path="../../configs", config_name="improved_rl_training", version_base=None)
def main(cfg: DictConfig):
    """メイン関数"""
    print("🚀 Starting Improved RL Training")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60)
    
    # 訓練器の初期化と実行
    trainer = ImprovedRLTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()