#!/usr/bin/env python3
"""
改良されたPPOエージェント
0%精度問題を解決するための改良版
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ImprovedActorCritic(nn.Module):
    """改良されたActor-Criticネットワーク"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Dict):
        super(ImprovedActorCritic, self).__init__()
        
        self.config = config
        hidden_layers = config.get('hidden_layers', [128, 128, 64])
        activation = config.get('activation', 'relu')
        dropout_rate = config.get('dropout_rate', 0.1)
        layer_norm = config.get('layer_norm', True)
        
        # 活性化関数の選択
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()
        
        # 共通特徴量抽出器
        feature_layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_layers[:-1]:
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                feature_layers.append(nn.LayerNorm(hidden_dim))
            feature_layers.append(self.activation)
            feature_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Actor（方策ネットワーク）
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            nn.LayerNorm(hidden_layers[-1]) if layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[-1], action_dim)
        )
        
        # Critic（価値ネットワーク）
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            nn.LayerNorm(hidden_layers[-1]) if layer_norm else nn.Identity(),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """フォワードパス"""
        features = self.feature_extractor(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """行動選択"""
        action_logits, _ = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)))
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動評価"""
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # 価値関数の出力を適切な形状に調整
        value = value.squeeze(-1)
        
        return log_prob, value, entropy


class ImprovedPPOAgent:
    """改良されたPPOエージェント"""
    
    def __init__(self, obs_dim: int, act_space, config: Dict, device: str = 'cpu'):
        self.device = torch.device(device)
        self.config = config
        
        # ハイパーパラメータ
        self.lr = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.k_epochs = config.get('k_epochs', 10)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # ネットワーク設定
        network_config = config.get('network', {})
        
        # ネットワークの初期化
        self.policy = ImprovedActorCritic(obs_dim, act_space.n, network_config).to(self.device)
        self.policy_old = ImprovedActorCritic(obs_dim, act_space.n, network_config).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # オプティマイザ
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 学習率スケジューラ
        lr_schedule = config.get('lr_schedule', 'constant')
        if lr_schedule == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, 
                total_iters=config.get('total_timesteps', 500000) // config.get('rollout_len', 256)
            )
        else:
            self.scheduler = None
        
        # 統計情報
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'grad_norm': [],
            'learning_rate': []
        }
    
    def get_action_and_value(self, state: Any, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, None, torch.Tensor]:
        """行動選択と価値推定"""
        with torch.no_grad():
            # 状態の前処理
            state_tensor = self._preprocess_state(state)
            
            # 行動選択
            action, log_prob = self.policy.act(state_tensor, deterministic)
            
            # 価値推定
            _, value = self.policy.forward(state_tensor)
            value = value.squeeze(-1)  # 適切な形状に調整
            
            return action.cpu().numpy(), log_prob.cpu(), None, value.cpu()
    
    def _preprocess_state(self, state: Any) -> torch.Tensor:
        """状態の前処理"""
        if isinstance(state, dict):
            # 辞書形式の状態を結合
            state_list = []
            for key in sorted(state.keys()):  # 順序を固定
                value = state[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.FloatTensor(value)
                state_list.append(value.flatten())
            
            state_tensor = torch.cat(state_list, dim=0)
        else:
            state_tensor = torch.FloatTensor(state)
        
        return state_tensor.to(self.device)
    
    def update(self, memory) -> Dict[str, float]:
        """ポリシー更新"""
        # データの準備
        states = memory.obs.detach().to(self.device)
        actions = memory.actions.detach().to(self.device).long().squeeze()
        old_log_probs = memory.log_probs.detach().to(self.device).squeeze()
        rewards = memory.returns.detach().to(self.device).squeeze()
        advantages = memory.advantages.detach().to(self.device).squeeze()
        
        # アドバンテージの正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # バッチサイズの設定
        batch_size = self.config.get('batch_size', len(states))
        minibatch_size = self.config.get('minibatch_size', batch_size // 4)
        
        # 複数エポック更新
        epoch_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'grad_norm': []
        }
        
        for epoch in range(self.k_epochs):
            # ミニバッチでの更新
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), minibatch_size):
                end_idx = min(start_idx + minibatch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # ミニバッチデータ
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 現在のポリシーでの評価
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # 重要度比率
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO損失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 価値関数損失
                values_flat = values.squeeze(-1)
                
                # デバッグ情報（形状確認）
                if values_flat.shape != batch_rewards.shape:
                    print(f"警告: 価値関数出力形状 {values_flat.shape} と報酬形状 {batch_rewards.shape} が一致しません")
                    # 形状を強制的に合わせる
                    if values_flat.dim() > 1:
                        values_flat = values_flat.view(-1)
                    if batch_rewards.dim() > 1:
                        batch_rewards = batch_rewards.view(-1)
                
                value_loss = F.mse_loss(values_flat, batch_rewards)
                
                # エントロピー損失
                entropy_loss = -entropy.mean()
                
                # 総損失
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 勾配更新
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 勾配クリッピング
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # 統計情報の記録
                epoch_stats['policy_loss'].append(policy_loss.item())
                epoch_stats['value_loss'].append(value_loss.item())
                epoch_stats['entropy_loss'].append(entropy_loss.item())
                epoch_stats['total_loss'].append(total_loss.item())
                epoch_stats['grad_norm'].append(grad_norm.item())
        
        # 学習率スケジューラの更新
        if self.scheduler:
            self.scheduler.step()
        
        # 古いポリシーの更新
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 統計情報の更新
        for key in epoch_stats:
            if epoch_stats[key]:
                self.training_stats[key].append(np.mean(epoch_stats[key]))
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.training_stats['learning_rate'].append(current_lr)
        
        return {
            'policy_loss': np.mean(epoch_stats['policy_loss']),
            'value_loss': np.mean(epoch_stats['value_loss']),
            'entropy_loss': np.mean(epoch_stats['entropy_loss']),
            'total_loss': np.mean(epoch_stats['total_loss']),
            'grad_norm': np.mean(epoch_stats['grad_norm']),
            'learning_rate': current_lr
        }
    
    def save(self, filepath: str):
        """モデルの保存"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }, filepath)
    
    def load(self, filepath: str):
        """モデルの読み込み"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)