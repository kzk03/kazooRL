#!/usr/bin/env python3
"""
高度なPPOエージェント実装
本格的な強化学習システム用
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


class AttentionNetwork(nn.Module):
    """アテンション機構付きネットワーク"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # マルチヘッドアテンション
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 入力投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 正規化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # フィードフォワード
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播"""
        # 入力投影
        x = self.input_projection(x)
        
        # アテンション
        if x.dim() == 2:
            x = x.unsqueeze(1)  # バッチ次元を追加
        
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # フィードフォワード
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x.squeeze(1) if x.size(1) == 1 else x


class AdvancedActorCritic(nn.Module):
    """高度なActor-Criticネットワーク"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        use_attention: bool = True,
        use_residual: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 共有特徴抽出器
        self.feature_extractor = self._build_feature_extractor(
            obs_dim, hidden_dims[0], dropout_rate
        )
        
        # アテンション層（オプション）
        if use_attention:
            self.attention = AttentionNetwork(
                hidden_dims[0], hidden_dims[0], num_heads=4
            )
        
        # Actor（ポリシー）ネットワーク
        self.actor = self._build_actor(hidden_dims, action_dim, dropout_rate)
        
        # Critic（価値）ネットワーク
        self.critic = self._build_critic(hidden_dims, dropout_rate)
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _build_feature_extractor(
        self, input_dim: int, output_dim: int, dropout_rate: float
    ) -> nn.Module:
        """特徴抽出器の構築"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def _build_actor(
        self, hidden_dims: List[int], action_dim: int, dropout_rate: float
    ) -> nn.Module:
        """Actorネットワークの構築"""
        layers = []
        
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # 出力層
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        
        return nn.Sequential(*layers)
    
    def _build_critic(
        self, hidden_dims: List[int], dropout_rate: float
    ) -> nn.Module:
        """Criticネットワークの構築"""
        layers = []
        
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # 出力層（価値は1次元）
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module: nn.Module):
        """重みの初期化"""
        if isinstance(module, nn.Linear):
            # Xavier初期化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播"""
        # 特徴抽出
        features = self.feature_extractor(obs)
        
        # アテンション（オプション）
        if self.use_attention:
            attended_features = self.attention(features)
            if self.use_residual:
                features = features + attended_features
            else:
                features = attended_features
        
        # Actor（ポリシー）
        action_logits = self.actor(features)
        
        # Critic（価値）
        value = self.critic(features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動と価値の取得"""
        action_logits, value = self.forward(obs)
        
        # 行動分布
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class AdvancedPPOAgent:
    """高度なPPOエージェント"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Dict,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.config = config
        
        # ハイパーパラメータ
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.k_epochs = config.get("k_epochs", 4)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        
        # 適応的学習率
        self.use_adaptive_lr = config.get("adaptive_lr", True)
        self.lr_decay_rate = config.get("lr_decay_rate", 0.99)
        
        # カリキュラム学習
        self.use_curriculum = config.get("use_curriculum_learning", True)
        self.curriculum_stage = 0
        self.curriculum_threshold = config.get("curriculum_threshold", 0.7)
        
        # ネットワーク構築
        network_config = config.get("network", {})
        self.actor_critic = AdvancedActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=network_config.get("hidden_dims", [256, 256, 128]),
            use_attention=network_config.get("use_attention", True),
            use_residual=network_config.get("use_residual", True),
            dropout_rate=network_config.get("dropout_rate", 0.1)
        ).to(self.device)
        
        # オプティマイザー
        self.optimizer = Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # 学習率スケジューラー
        if self.use_adaptive_lr:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("total_timesteps", 100000) // config.get("rollout_len", 2048),
                eta_min=self.learning_rate * 0.1
            )
        
        # 統計情報
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "kl_divergence": [],
            "explained_variance": []
        }
        
        # 早期停止
        self.early_stopping = config.get("early_stopping", {})
        self.best_performance = float('-inf')
        self.patience_counter = 0
    
    def get_action_and_value(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動と価値の取得"""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.actor_critic(obs)
            
            if deterministic:
                # 決定論的行動選択
                action = torch.argmax(action_logits, dim=-1)
                probs = Categorical(logits=action_logits)
                log_prob = probs.log_prob(action)
                entropy = probs.entropy()
            else:
                # 確率的行動選択
                probs = Categorical(logits=action_logits)
                action = probs.sample()
                log_prob = probs.log_prob(action)
                entropy = probs.entropy()
        
        return action, log_prob, entropy, value
    
    def update(self, rollout_buffer) -> Dict[str, float]:
        """エージェントの更新"""
        # データの準備
        obs = rollout_buffer.obs.to(self.device)
        actions = rollout_buffer.actions.to(self.device)
        old_log_probs = rollout_buffer.log_probs.to(self.device)
        returns = rollout_buffer.returns.to(self.device)
        advantages = rollout_buffer.advantages.to(self.device)
        
        # アドバンテージの正規化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 複数エポック更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        
        for epoch in range(self.k_epochs):
            # ミニバッチ学習
            batch_size = len(obs)
            minibatch_size = self.config.get("minibatch_size", batch_size)
            
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # ミニバッチデータ
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # 現在のポリシーでの評価
                _, new_log_probs, entropy, values = self.actor_critic.get_action_and_value(
                    mb_obs, mb_actions.squeeze(-1)
                )
                
                # 重要度比
                ratio = torch.exp(new_log_probs - mb_old_log_probs.squeeze(-1))
                
                # PPOクリッピング
                surr1 = ratio * mb_advantages.squeeze(-1)
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages.squeeze(-1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 価値関数損失
                value_loss = F.mse_loss(values, mb_returns.squeeze(-1))
                
                # エントロピー損失
                entropy_loss = -entropy.mean()
                
                # 総損失
                total_loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # 勾配更新
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 勾配クリッピング
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # 統計情報の蓄積
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                
                # KLダイバージェンス
                with torch.no_grad():
                    kl_div = (mb_old_log_probs.squeeze(-1) - new_log_probs).mean()
                    total_kl_div += kl_div.item()
        
        # 学習率スケジューリング
        if self.use_adaptive_lr:
            self.scheduler.step()
        
        # 統計情報の更新
        num_updates = self.k_epochs * math.ceil(batch_size / self.config.get("minibatch_size", batch_size))
        
        stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy_loss": total_entropy_loss / num_updates,
            "total_loss": (total_policy_loss + total_value_loss + total_entropy_loss) / num_updates,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "kl_divergence": total_kl_div / num_updates,
            "explained_variance": self._explained_variance(returns, values.detach())
        }
        
        # 統計情報の記録
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def _explained_variance(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """説明分散の計算"""
        var_y = torch.var(y_true)
        return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
    
    def update_curriculum(self, performance_metric: float):
        """カリキュラム学習の更新"""
        if not self.use_curriculum:
            return
        
        if performance_metric > self.curriculum_threshold:
            self.curriculum_stage += 1
            print(f"📚 カリキュラム学習: ステージ {self.curriculum_stage} に進行")
    
    def should_early_stop(self, performance_metric: float) -> bool:
        """早期停止の判定"""
        if not self.early_stopping.get("enabled", False):
            return False
        
        improvement = performance_metric - self.best_performance
        min_improvement = self.early_stopping.get("min_improvement", 0.001)
        
        if improvement > min_improvement:
            self.best_performance = performance_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            patience = self.early_stopping.get("patience", 50)
            
            if self.patience_counter >= patience:
                print(f"🛑 早期停止: {patience}エピソード改善なし")
                return True
        
        return False
    
    def save(self, path: str):
        """モデルの保存"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'curriculum_stage': self.curriculum_stage
        }, path)
    
    def load(self, path: str):
        """モデルの読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', {})
        self.curriculum_stage = checkpoint.get('curriculum_stage', 0)