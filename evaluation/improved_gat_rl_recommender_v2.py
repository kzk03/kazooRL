#!/usr/bin/env python3
"""
改善版GAT特徴量統合強化学習推薦システム v2

改善点:
1. 正解報酬を圧倒的に重要に (100.0)
2. GAT特徴量は補助的役割 (最大5.5)
3. 間違いに対する明確なペナルティ (-20.0)
4. 無効行動に対する重いペナルティ (-50.0)
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# プロジェクトのモジュール
sys.path.append('/Users/kazuki-h/rl/kazoo')
sys.path.append('/Users/kazuki-h/rl/kazoo/src')

from src.kazoo.envs.task import Task
from src.kazoo.features.feature_extractor import FeatureExtractor


class ImprovedGATRLEnvironmentV2(gym.Env):
    """
    改善版GAT特徴量統合強化学習環境 v2
    - 正解報酬を圧倒的に重要に
    - GAT特徴量は補助的役割
    """
    
    def __init__(self, config, training_data, dev_profiles):
        super().__init__()
        
        self.config = config
        self.training_data = training_data
        self.dev_profiles = dev_profiles
        
        # 1. 特徴量抽出器の初期化
        print("🔧 GAT特徴量抽出器初期化中...")
        # 設定をDictConfig風にラップして必要な設定を追加
        from omegaconf import DictConfig
        if isinstance(config, dict):
            # 必要な設定を追加
            if 'features' not in config:
                config['features'] = {
                    'recent_activity_window_days': 30,
                    'all_labels': ["bug", "enhancement", "documentation", "question", "help wanted"],
                    'label_to_skills': {
                        'bug': ["debugging", "analysis"],
                        'enhancement': ["python", "design"],
                        'documentation': ["writing"],
                        'question': ["communication"],
                        'help wanted': ["collaboration"]
                    }
                }
            if 'gat' not in config:
                config['gat'] = {
                    'model_path': "data/gnn_model_collaborative.pt",
                    'graph_data_path': "data/graph_collaborative.pt"
                }
            config = DictConfig(config)
        self.feature_extractor = FeatureExtractor(config)
        
        # 2. IRL重み読み込み
        irl_weights_path = "data/learned_weights_training.npy"
        if Path(irl_weights_path).exists():
            self.irl_weights = torch.tensor(np.load(irl_weights_path), dtype=torch.float32)
            print(f"✅ IRL重み読み込み: {irl_weights_path} ({tuple(self.irl_weights.shape)})")
        else:
            self.irl_weights = torch.zeros(62, dtype=torch.float32)
            print(f"⚠️ IRL重みファイルが見つかりません: {irl_weights_path}")
        
        # 3. アクティブ開発者抽出（訓練データに基づく）
        training_pairs = []
        developer_stats = Counter()
        
        for task_data in training_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            assignee = None
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            
            if assignee:
                training_pairs.append({
                    'task_data': task_data,
                    'developer': assignee,
                    'task_id': task_id
                })
                developer_stats[assignee] += 1
        
        print(f"📋 アクティブ開発者抽出: {len(developer_stats)}人")
        top_devs = developer_stats.most_common(6)  # 上位6人に制限
        print("   上位アクティブ開発者:")
        for dev, count in top_devs:
            print(f"     {dev}: {count} タスク")
        
        self.active_developers = [dev for dev, _ in top_devs]
        self.num_developers = len(self.active_developers)
        
        print(f"🎯 行動空間を実訓練開発者に制限: {self.num_developers}人")
        
        # 4. 環境設定
        feature_dim = len(self.feature_extractor.feature_names)
        
        self.action_space = gym.spaces.Discrete(self.num_developers)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(feature_dim,),  # GAT特徴量のみ
            dtype=np.float32
        )
        
        # 学習ペア
        self.training_pairs = [pair for pair in training_pairs 
                             if pair['developer'] in self.active_developers]
        
        print(f"🤖 改善版GAT-RL環境v2初期化完了")
        print(f"   アクティブ開発者: {self.num_developers}")
        print(f"   観測次元: {feature_dim} (GAT特徴量のみ)")
        print(f"   学習ペア数: {len(self.training_pairs)}")
        
        # エピソード管理
        self.current_pair_idx = 0
        self.current_pair = None
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        if not self.training_pairs:
            # フォールバック: 空の観測
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, {}
        
        # ランダムなペアを選択
        self.current_pair_idx = np.random.randint(0, len(self.training_pairs))
        self.current_pair = self.training_pairs[self.current_pair_idx]
        
        # 初期観測を取得
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """アクションを実行"""
        if action >= self.num_developers:
            # 無効なアクションに対する重いペナルティ
            reward = -50.0
            terminated = True
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, reward, terminated, False, {}
        
        selected_dev = self.active_developers[action]
        actual_dev = self.current_pair['developer']
        
        # 報酬計算（改善版v2）
        reward = self._calculate_improved_reward_v2(selected_dev, actual_dev)
        
        # エピソード終了
        terminated = True
        
        # 次の観測（エピソード終了なので現在の観測）
        obs = self._get_observation()
        
        # 詳細情報
        info = {
            'selected_dev': selected_dev,
            'actual_dev': actual_dev,
            'correct': selected_dev == actual_dev,
            'task_id': self.current_pair['task_id'],
            'reward_breakdown': getattr(self, 'last_reward_breakdown', {})
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self):
        """現在のタスクと選択肢の観測を取得"""
        if not self.current_pair:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        try:
            # タスクオブジェクト作成
            task_obj = Task(self.current_pair['task_data'])
            
            # ダミー開発者（実際の開発者プロファイルを使用）
            actual_dev = self.current_pair['developer']
            dev_profile = self.dev_profiles.get(actual_dev, {})
            developer_obj = {"name": actual_dev, "profile": dev_profile}
            
            # ダミー環境
            dummy_env = type('DummyEnv', (), {
                'backlog': [task_obj],
                'dev_profiles': self.dev_profiles,
                'assignments': {},
                'dev_action_history': {}
            })()
            
            # GAT特徴量抽出
            features = self.feature_extractor.get_features(
                task_obj, developer_obj, dummy_env
            )
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"⚠️ 観測取得エラー: {e}")
            # フォールバック: ゼロベクトル
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _calculate_improved_reward_v2(self, selected_dev, actual_dev):
        """改善版報酬計算v2 - 正解報酬を圧倒的に重要に"""
        total_reward = 0.0
        reward_breakdown = {}
        
        # 1. 正解報酬 (圧倒的に重要)
        if selected_dev == actual_dev:
            correct_reward = 100.0  # 圧倒的な正解報酬
            total_reward += correct_reward
            reward_breakdown['correct'] = correct_reward
        else:
            # 間違いに対する明確なペナルティ
            wrong_penalty = -20.0
            total_reward += wrong_penalty
            reward_breakdown['correct'] = wrong_penalty
        
        # 2. GAT特徴量報酬 (補助的役割のみ)
        try:
            # 選択された開発者でのGAT特徴量抽出
            task_obj = Task(self.current_pair['task_data'])
            selected_dev_profile = self.dev_profiles.get(selected_dev, {})
            selected_dev_obj = {"name": selected_dev, "profile": selected_dev_profile}
            
            dummy_env = type('DummyEnv', (), {
                'backlog': [task_obj],
                'dev_profiles': self.dev_profiles,
                'assignments': {},
                'dev_action_history': {}
            })()
            
            # GAT特徴量抽出
            features = self.feature_extractor.get_features(
                task_obj, selected_dev_obj, dummy_env
            )
            
            # GAT類似度特徴量 (補助的)
            gat_similarity_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                                if name == 'gat_similarity']
            if gat_similarity_idx:
                similarity_score = features[gat_similarity_idx[0]]
                similarity_reward = similarity_score * 2.0  # 補助的役割
                total_reward += similarity_reward
                reward_breakdown['gat_similarity'] = similarity_reward
            
            # GAT専門性報酬 (補助的)
            expertise_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                           if name == 'gat_dev_expertise']
            if expertise_idx:
                expertise_score = features[expertise_idx[0]]
                expertise_reward = expertise_score * 1.5  # 補助的
                total_reward += expertise_reward
                reward_breakdown['gat_expertise'] = expertise_reward
            
            # GAT協力強度報酬 (補助的)
            collaboration_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                               if name == 'gat_collaboration_strength']
            if collaboration_idx:
                collab_score = features[collaboration_idx[0]]
                collab_reward = collab_score * 1.0  # 補助的
                total_reward += collab_reward
                reward_breakdown['gat_collaboration'] = collab_reward
            
            # IRL重みとの統合報酬 (補助的)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            irl_score = torch.dot(self.irl_weights, features_tensor).item()
            irl_reward = np.tanh(irl_score) * 1.0  # 補助的
            total_reward += irl_reward
            reward_breakdown['irl_compatibility'] = irl_reward
            
        except Exception as e:
            print(f"⚠️ GAT報酬計算エラー: {e}")
            # エラー時はGAT報酬を0に
            reward_breakdown['gat_similarity'] = 0.0
            reward_breakdown['gat_expertise'] = 0.0
            reward_breakdown['gat_collaboration'] = 0.0
            reward_breakdown['irl_compatibility'] = 0.0
        
        # 報酬詳細を保存
        self.last_reward_breakdown = reward_breakdown
        
        return total_reward


class ImprovedGATRLRecommenderV2:
    """改善版GAT特徴量統合強化学習推薦システムv2"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🚀 改善版GAT統合強化学習推薦システムv2初期化完了")
    
    def load_data(self, data_path):
        """データを読み込んで時系列分割"""
        print("📊 データ読み込み中...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # 時系列分割
        training_data = []  # 2022年以前
        test_data = []      # 2023年
        
        for task in all_data:
            created_at = task.get('created_at', '')
            if created_at.startswith('2022'):
                training_data.append(task)
            elif created_at.startswith('2023'):
                test_data.append(task)
        
        print(f"   学習データ: {len(training_data):,} タスク (2022年)")
        print(f"   テストデータ: {len(test_data):,} タスク (2023年)")
        
        return training_data, test_data
    
    def load_dev_profiles(self, profiles_path="configs/dev_profiles.yaml"):
        """開発者プロファイル読み込み"""
        try:
            with open(profiles_path, 'r', encoding='utf-8') as f:
                profiles = yaml.safe_load(f)
            print(f"📋 開発者プロファイル読み込み: {len(profiles)} 人")
            return profiles
        except Exception as e:
            print(f"⚠️ プロファイル読み込みエラー: {e}")
            return {}
    
    def train_improved_rl_agent_v2(self, training_data, dev_profiles, total_timesteps=100000):
        """改善版GAT統合強化学習エージェント訓練v2"""
        print("🎓 改善版GAT統合強化学習エージェントv2訓練開始...")
        
        # 環境作成
        def make_env():
            return ImprovedGATRLEnvironmentV2(self.config, training_data, dev_profiles)
        
        env = DummyVecEnv([make_env])
        
        # PPOエージェント（最適化版）
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0001,  # 低い学習率で安定化
            n_steps=256,           # 小さなバッチサイズ
            batch_size=64,         # さらに小さく
            n_epochs=30,           # エポック数増加
            gamma=0.98,            # 割引率を高く
            gae_lambda=0.95,       # GAE
            clip_range=0.1,        # 保守的なクリップ
            ent_coef=0.01,         # エントロピー係数
            vf_coef=1.0,           # 価値関数係数
            max_grad_norm=0.5,     # 勾配クリッピング
            device="auto",
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # より大きなネットワーク
                activation_fn=torch.nn.ReLU
            )
        )
        
        # 評価コールバック
        eval_env = DummyVecEnv([make_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/improved_gat_rl_v2_best/",
            log_path="./logs/improved_gat_rl_v2_eval/",
            eval_freq=2000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # 訓練実行
        print(f"   訓練ステップ: {total_timesteps:,}")
        print(f"   推定時間: {total_timesteps / 1000:.1f}分")
        
        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        print("✅ 改善版GAT統合強化学習エージェントv2訓練完了")
    
    def predict_with_improved_gat_rl_v2(self, test_data, dev_profiles):
        """改善版GAT強化学習v2で予測"""
        print("🤖 改善版GAT強化学習v2予測実行中...")
        
        # テスト環境作成
        test_env = ImprovedGATRLEnvironmentV2(self.config, test_data, dev_profiles)
        
        predictions = {}
        prediction_scores = {}
        
        # テスト用のアクティブ開発者抽出
        test_developer_stats = Counter()
        test_assignments = {}
        
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            assignee = None
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            
            if assignee:
                test_assignments[task_id] = assignee
                test_developer_stats[assignee] += 1
        
        # アクティブ開発者に限定
        active_test_tasks = {
            task_id: dev for task_id, dev in test_assignments.items()
            if dev in test_env.active_developers
        }
        
        print(f"   予測対象: {len(active_test_tasks)} タスク（アクティブ開発者のみ）")
        
        progress_count = 0
        total_tasks = len(active_test_tasks) * len(test_env.active_developers)
        
        # 予測実行
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if task_id not in active_test_tasks:
                continue
            
            try:
                # タスクをセット
                test_env.current_pair = {
                    'task_data': task_data,
                    'task_id': task_id,
                    'developer': active_test_tasks[task_id]
                }
                
                # 観測取得
                obs = test_env._get_observation()
                if obs is None:
                    continue
                
                # numpy配列をPyTorchテンサーに変換
                import torch
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # PPOポリシーから行動確率分布を取得
                with torch.no_grad():
                    action_probs = self.rl_agent.policy.get_distribution(obs_tensor).distribution.probs.squeeze().cpu().numpy()
                
                # 各開発者のスコアを取得
                action_scores = []
                for dev_idx in range(test_env.num_developers):
                    dev_name = test_env.active_developers[dev_idx]
                    score = float(action_probs[dev_idx])
                    action_scores.append((dev_name, score))
                
                # スコア順にソート
                action_scores.sort(key=lambda x: x[1], reverse=True)
                
                # 最高スコアの開発者を選択
                if action_scores:
                    predicted_dev = action_scores[0][0]
                    best_score = action_scores[0][1]
                    
                    predictions[task_id] = predicted_dev
                    prediction_scores[task_id] = {
                        'predicted_dev': predicted_dev,
                        'score': best_score,
                        'all_scores': dict(action_scores)  # 全開発者のスコア（ソート済み）
                    }
                
                progress_count += len(test_env.active_developers)
                if progress_count % 100 == 0:
                    print(f"   進捗: {progress_count // len(test_env.active_developers)}/{len(active_test_tasks)}")
                
            except Exception as e:
                print(f"⚠️ 予測エラー (task {task_id}): {e}")
                continue
        
        print(f"   改善GAT-RLv2予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores, active_test_tasks
    
    def run_full_pipeline_v2(self, data_path, total_timesteps=100000):
        """完全パイプラインの実行v2"""
        print("🚀 改善版GAT統合強化学習推薦システムv2実行開始")
        print("=" * 80)
        
        # 1. データ読み込み
        training_data, test_data = self.load_data(data_path)
        
        # 2. 開発者プロファイル読み込み
        dev_profiles = self.load_dev_profiles()
        
        # 3. 改善版GAT統合強化学習エージェント訓練
        self.train_improved_rl_agent_v2(training_data, dev_profiles, total_timesteps)
        
        # 4. 予測実行
        predictions, prediction_scores, test_assignments = self.predict_with_improved_gat_rl_v2(
            test_data, dev_profiles
        )
        
        # 5. 評価
        metrics = self.evaluate_predictions_v2(predictions, test_assignments, prediction_scores)
        
        # 6. モデル保存
        self.save_model_v2()
        
        print("✅ 改善版GAT統合強化学習推薦システムv2完了")
        return metrics
    
    def evaluate_predictions_v2(self, predictions, test_assignments, prediction_scores=None):
        """予測結果の評価v2（Top-K評価含む）"""
        print("📊 改善版GAT-RLv2評価中...")
        
        # 共通タスクで評価
        common_tasks = set(predictions.keys()) & set(test_assignments.keys())
        print(f"   評価対象: {len(common_tasks)} タスク")
        
        if not common_tasks:
            print("⚠️ 評価可能なタスクがありません")
            return {}
        
        # Top-1 (従来の)正確性評価
        correct_predictions = 0
        for task_id in common_tasks:
            if predictions[task_id] == test_assignments[task_id]:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(common_tasks)
        
        # Top-K評価
        topk_metrics = self._evaluate_topk_accuracy_v2(common_tasks, test_assignments, prediction_scores)
        
        metrics = {
            'improved_gat_rl_v2_accuracy': accuracy,
            'improved_gat_rl_v2_top1_accuracy': accuracy,
            'improved_gat_rl_v2_top3_accuracy': topk_metrics['top3_accuracy'],
            'improved_gat_rl_v2_top5_accuracy': topk_metrics['top5_accuracy'],
            'improved_gat_rl_v2_correct': correct_predictions,
            'improved_gat_rl_v2_total': len(common_tasks)
        }
        
        print(f"🤖 改善版GAT強化学習システムv2:")
        print(f"   Top-1精度: {accuracy:.3f} ({correct_predictions}/{len(common_tasks)})")
        print(f"   Top-3精度: {topk_metrics['top3_accuracy']:.3f} ({topk_metrics['top3_correct']}/{len(common_tasks)})")
        print(f"   Top-5精度: {topk_metrics['top5_accuracy']:.3f} ({topk_metrics['top5_correct']}/{len(common_tasks)})")
        
        return metrics
    
    def _evaluate_topk_accuracy_v2(self, common_tasks, test_assignments, prediction_scores):
        """Top-K accuracy評価v2"""
        top3_correct = 0
        top5_correct = 0
        
        for task_id in common_tasks:
            actual_dev = test_assignments[task_id]
            
            if prediction_scores and task_id in prediction_scores:
                # 全スコアを取得
                all_scores = prediction_scores[task_id].get('all_scores', {})
                if not all_scores:
                    continue
                
                # スコア順に開発者を取得
                sorted_devs = list(all_scores.keys())
                
                # Top-3評価
                if len(sorted_devs) >= 3:
                    top3_devs = sorted_devs[:3]
                    if actual_dev in top3_devs:
                        top3_correct += 1
                elif actual_dev in sorted_devs:
                    top3_correct += 1
                
                # Top-5評価
                if len(sorted_devs) >= 5:
                    top5_devs = sorted_devs[:5]
                    if actual_dev in top5_devs:
                        top5_correct += 1
                elif actual_dev in sorted_devs:
                    top5_correct += 1
        
        total_tasks = len(common_tasks)
        return {
            'top3_accuracy': top3_correct / total_tasks if total_tasks > 0 else 0.0,
            'top5_accuracy': top5_correct / total_tasks if total_tasks > 0 else 0.0,
            'top3_correct': top3_correct,
            'top5_correct': top5_correct
        }
    
    def save_model_v2(self, output_dir="models"):
        """モデルの保存v2"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"improved_gat_rl_recommender_v2_{timestamp}.zip"
        
        self.rl_agent.save(model_path)
        print(f"✅ 改善版GAT統合RLモデルv2保存: {model_path}")
        return model_path


def main():
    parser = argparse.ArgumentParser(description='改善版GAT統合強化学習推薦システムv2')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--data', default='data/backlog.json',
                       help='統合データパス')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='訓練ステップ数')
    
    args = parser.parse_args()
    
    # 推薦システム実行
    recommender = ImprovedGATRLRecommenderV2(args.config)
    metrics = recommender.run_full_pipeline_v2(args.data, args.timesteps)
    
    print("\n🎯 最終結果:")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
