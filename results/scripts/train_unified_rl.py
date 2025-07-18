#!/usr/bin/env python3
"""
統合強化学習システム - IRL重みを使用したRL訓練
train_oss.pyとtrain_rl_agent.pyの統合改良版

特徴:
- Hydra設定管理 + IRL重み統合
- 既存のOSSSimpleEnvを活用しつつカスタム報酬関数を追加
- 性能評価とCSV出力機能
- 自動化されたパイプライン実行
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# パス設定
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import FeatureExtractor
from kazoo.learners.independent_ppo_controller import IndependentPPOController


class UnifiedTaskAssignmentEnv(OSSSimpleEnv):
    """
    IRL重みを統合した統一タスク割り当て環境
    既存のOSSSimpleEnvを継承し、IRL重みベースの報酬を追加
    """
    
    def __init__(self, config, backlog, dev_profiles, irl_weights_path=None):
        super().__init__(config, backlog, dev_profiles)
        
        self.feature_extractor = FeatureExtractor(config)
        self.irl_weights = self._load_irl_weights(irl_weights_path)
        
        # デバッグ情報
        print(f"🎮 統合環境初期化完了")
        print(f"   開発者数: {len(dev_profiles)}")
        print(f"   タスク数: {len(backlog)}")
        print(f"   特徴量次元: {len(self.feature_extractor.feature_names)}")
        print(f"   IRL重み形状: {self.irl_weights.shape}")
    
    def _load_irl_weights(self, weights_path):
        """IRL学習済み重みを読み込み"""
        if weights_path and Path(weights_path).exists():
            try:
                weights = np.load(weights_path)
                print(f"✅ IRL重みを読み込み: {weights_path} ({weights.shape})")
                return torch.tensor(weights, dtype=torch.float32)
            except Exception as e:
                print(f"⚠️ IRL重み読み込みエラー: {e}")
        
        # フォールバック: ランダム重み
        feature_dim = len(self.feature_extractor.feature_names)
        weights = torch.randn(feature_dim, dtype=torch.float32)
        print(f"⚠️ ランダム重みを使用: {weights.shape}")
        return weights
    
    def calculate_irl_reward(self, task, developer) -> float:
        """IRL重みを使用した報酬計算"""
        try:
            # 特徴量を抽出
            features = self.feature_extractor.get_features(task, developer, self)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # IRL重みとの内積で報酬を計算
            irl_reward = torch.dot(self.irl_weights, features_tensor).item()
            
            # 報酬を正規化
            irl_reward = np.clip(irl_reward, -10.0, 10.0)
            
            return irl_reward
            
        except Exception as e:
            print(f"⚠️ IRL報酬計算エラー: {e}")
            return 0.0
    
    def step(self, action):
        """ステップ実行時にIRL報酬を追加"""
        # 元の環境のステップを実行
        obs, original_reward, terminated, truncated, info = super().step(action)
        
        # IRL報酬を計算して追加
        if hasattr(self, '_last_assignment'):
            task, developer = self._last_assignment
            irl_reward = self.calculate_irl_reward(task, developer)
            
            # 元の報酬とIRL報酬を組み合わせ
            combined_reward = 0.5 * original_reward + 0.5 * irl_reward
            
            info['original_reward'] = original_reward
            info['irl_reward'] = irl_reward
            info['combined_reward'] = combined_reward
            
            return obs, combined_reward, terminated, truncated, info
        
        return obs, original_reward, terminated, truncated, info


@hydra.main(config_path="../configs", config_name="base_training", version_base=None)
def main(cfg: DictConfig):
    """メイン実行関数"""
    
    print("🚀 統合強化学習システム開始")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    # 1. ファイル存在確認
    print("1. ファイル存在確認...")
    required_files = [
        cfg.env.backlog_path,
        cfg.env.dev_profiles_path,
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 必要ファイルが見つかりません: {file_path}")
            return
    
    print("✅ 全ての必要ファイルを確認")
    
    # 2. データ読み込み
    print("2. データ読み込み...")
    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)
    
    print(f"   バックログ: {len(backlog)} タスク")
    print(f"   開発者: {len(dev_profiles)} 人")
    
    # 3. 統合環境の初期化
    print("3. 統合環境初期化...")
    irl_weights_path = getattr(cfg.irl, 'output_weights_path', None)
    
    env = UnifiedTaskAssignmentEnv(
        config=cfg,
        backlog=backlog,
        dev_profiles=dev_profiles,
        irl_weights_path=irl_weights_path
    )
    
    # 4. 訓練方法の選択
    training_method = cfg.get('training_method', 'unified')
    
    if training_method == 'original':
        # 元のOSSSimpleEnv + IndependentPPOControllerを使用
        print("4. 元のシステムで訓練...")
        train_original_system(cfg, env)
        
    elif training_method == 'stable_baselines':
        # Stable-Baselines3を直接使用
        print("4. Stable-Baselines3で訓練...")
        train_with_stable_baselines(cfg, env)
        
    else:
        # 統合システム（デフォルト）
        print("4. 統合システムで訓練...")
        train_unified_system(cfg, env)
    
    # 5. 評価とレポート生成
    print("5. 評価とレポート生成...")
    generate_evaluation_report(cfg, env)
    
    print("✅ 統合強化学習システム完了")


def train_original_system(cfg: DictConfig, env: UnifiedTaskAssignmentEnv):
    """元のシステムでの訓練"""
    controller = IndependentPPOController(env=env, config=cfg)
    
    total_timesteps = cfg.rl.get('total_timesteps', 50000)
    controller.learn(total_timesteps=total_timesteps)
    
    # モデル保存
    output_dir = cfg.rl.get('output_model_dir', 'models/original_rl')
    controller.save_models(output_dir)
    print(f"✅ 元システムモデル保存: {output_dir}")


def train_with_stable_baselines(cfg: DictConfig, env: UnifiedTaskAssignmentEnv):
    """Stable-Baselines3での直接訓練"""
    
    # 環境をVectorizedに変換
    def make_env():
        return env
    
    vec_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    # PPOモデル作成
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=cfg.rl.get('learning_rate', 3e-4),
        n_steps=cfg.rl.get('n_steps', 2048),
        batch_size=cfg.rl.get('batch_size', 64),
        n_epochs=cfg.rl.get('n_epochs', 10),
        gamma=cfg.rl.get('gamma', 0.99),
        device="auto"
    )
    
    # 評価コールバック
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/unified_best/",
        log_path="./logs/unified_eval/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # 訓練実行
    total_timesteps = cfg.rl.get('total_timesteps', 50000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # モデル保存
    model_path = "models/unified_rl_agent.zip"
    model.save(model_path)
    print(f"✅ Stable-Baselines3モデル保存: {model_path}")


def train_unified_system(cfg: DictConfig, env: UnifiedTaskAssignmentEnv):
    """統合システムでの訓練（両方の手法を組み合わせ）"""
    
    print("📊 統合訓練: 元システム + Stable-Baselines3")
    
    # 1. 元システムでの事前訓練
    print("   1) 元システムでの事前訓練...")
    train_original_system(cfg, env)
    
    # 2. Stable-Baselines3での微調整
    print("   2) Stable-Baselines3での微調整...")
    train_with_stable_baselines(cfg, env)
    
    print("✅ 統合訓練完了")


def generate_evaluation_report(cfg: DictConfig, env: UnifiedTaskAssignmentEnv):
    """評価レポートとCSVファイルの生成"""
    
    print("📊 評価レポート生成中...")
    
    # 利用可能なモデルを検索
    model_paths = [
        "models/unified_rl_agent.zip",
        "models/original_rl/",
        "models/unified_best/"
    ]
    
    results = []
    
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                if model_path.endswith('.zip'):
                    # Stable-Baselines3モデル
                    model = PPO.load(model_path)
                    model_name = Path(model_path).stem
                    
                    # 評価実行
                    rewards = evaluate_sb3_model(model, env, num_episodes=10)
                    
                else:
                    # 元システムモデル（評価方法は別途実装が必要）
                    model_name = Path(model_path).name
                    rewards = [0.0]  # プレースホルダー
                
                # 結果を記録
                results.append({
                    'model_name': model_name,
                    'model_path': model_path,
                    'avg_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'max_reward': np.max(rewards),
                    'min_reward': np.min(rewards),
                    'num_episodes': len(rewards)
                })
                
                print(f"   ✅ {model_name}: 平均報酬 {np.mean(rewards):.4f}")
                
            except Exception as e:
                print(f"   ❌ {model_path} 評価エラー: {e}")
    
    # CSVレポート生成
    if results:
        df = pd.DataFrame(results)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"outputs/unified_rl_evaluation_{timestamp}.csv"
        
        df.to_csv(csv_path, index=False)
        print(f"✅ 評価レポート保存: {csv_path}")
        
        # サマリー表示
        print("\n📈 評価サマリー:")
        print(df.to_string(index=False))
    
    # 特徴量重要度分析
    analyze_feature_importance(env)


def evaluate_sb3_model(model, env, num_episodes=10) -> List[float]:
    """Stable-Baselines3モデルの評価"""
    rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
    
    return rewards


def analyze_feature_importance(env: UnifiedTaskAssignmentEnv):
    """特徴量重要度分析"""
    print("🔍 特徴量重要度分析...")
    
    feature_names = env.feature_extractor.feature_names
    irl_weights = env.irl_weights.numpy()
    
    # 重要度データフレーム作成
    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'irl_weight': irl_weights,
        'abs_weight': np.abs(irl_weights),
        'importance_rank': range(1, len(feature_names) + 1)
    })
    
    # 重要度順にソート
    importance_df = importance_df.sort_values('abs_weight', ascending=False)
    importance_df['importance_rank'] = range(1, len(importance_df) + 1)
    
    # CSV保存
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"outputs/unified_feature_importance_{timestamp}.csv"
    importance_df.to_csv(csv_path, index=False)
    
    print(f"✅ 特徴量重要度保存: {csv_path}")
    print("\n🏆 TOP10 重要特徴量:")
    print(importance_df.head(10)[['feature_name', 'irl_weight', 'abs_weight']].to_string(index=False))


if __name__ == "__main__":
    main()
