#!/usr/bin/env python3
"""
マルチメソッド強化学習タスク推薦システム

3つの異なる開発者抽出方法で独立した強化学習エージェントを訓練し、
それぞれの性能を比較する：

1. assignees_agent: assigneesフィールドのみ使用
2. creators_agent: Issue/PR作成者（user）のみ使用  
3. all_agent: すべての方法を統合（assignees + creators）

これにより、データリークや偏りの影響を最小化し、
より現実的な推薦システムの性能評価を行う。
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from simple_similarity_recommender import SimpleSimilarityRecommender
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


class MultiMethodRLEnvironment(gym.Env):
    """
    マルチメソッド強化学習環境
    
    各抽出方法に特化した独立環境で学習を行う
    """
    
    def __init__(self, extraction_method='assignees'):
        super().__init__()
        
        self.extraction_method = extraction_method
        self.recommender = SimpleSimilarityRecommender('configs/unified_rl.yaml')
        
        # データ読み込み
        self.training_data, self.test_data = self.recommender.load_data('data/backlog.json')
        
        # 抽出方法に応じて学習データを準備
        self.training_pairs, self.developer_stats = self.recommender.extract_training_pairs(
            self.training_data, extraction_method=self.extraction_method
        )
        
        print(f"🤖 {extraction_method}メソッドRL環境初期化:")
        print(f"   抽出方法: {extraction_method}")
        print(f"   学習ペア: {len(self.training_pairs)} ペア")
        print(f"   開発者数: {len(self.developer_stats)} 人")
        
        # 開発者リスト（この抽出方法で見つかった開発者のみ）
        self.developers = list(self.developer_stats.keys())
        self.dev_to_idx = {dev: idx for idx, dev in enumerate(self.developers)}
        
        # 学習データのインデックス管理
        self.current_task_idx = 0
        self.time_step = 0
        
        # 履歴とパフォーマンス統計
        self.prediction_history = []
        self.success_rate = 0.0
        self.recent_successes = []
        
        # アクション・観測空間定義（開発者数が0の場合の対策）
        num_developers = max(1, len(self.developers))  # 最低1つのアクション
        self.action_space = spaces.Discrete(num_developers)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(20,),  # タスク特徴量(10) + 開発者統計(5) + 履歴統計(5)
            dtype=np.float32
        )
        
        # 基本モデル訓練
        self._train_base_models()
    
    def _train_base_models(self):
        """ベースとなる類似度モデルを訓練"""
        print(f"📚 {self.extraction_method}用ベースモデル訓練中...")
        
        # 開発者プロファイル構築
        self.developer_profiles = self.recommender.build_developer_profiles(self.training_pairs)
        
        # テキスト類似度モデル
        self.recommender.train_text_similarity_model(self.developer_profiles)
        
        # 特徴量モデル  
        self.recommender.train_feature_model(self.training_pairs)
        
        print(f"   ベースモデル訓練完了: TF-IDF行列 {getattr(self.recommender, 'tfidf_matrix', 'N/A')}")
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        self.current_task_idx = 0
        self.time_step = 0
        self.prediction_history = []
        self.recent_successes = []
        self.success_rate = 0.0
        
        obs = self._get_observation()
        info = {
            'extraction_method': self.extraction_method,
            'total_developers': len(self.developers),
            'total_tasks': len(self.training_pairs)
        }
        
        return obs, info
    
    def step(self, action):
        """アクションを実行して次の状態へ"""
        if self.current_task_idx >= len(self.training_pairs):
            # エピソード終了
            obs = np.zeros(20, dtype=np.float32)
            return obs, 0.0, True, False, {'episode_complete': True}
        
        # 現在のタスクデータ
        current_pair = self.training_pairs[self.current_task_idx]
        current_task = current_pair['task_data']
        actual_developer = current_pair['developer']
        
        # アクションから予測開発者を取得
        if action < len(self.developers):
            predicted_developer = self.developers[action]
        else:
            # 無効なアクション
            predicted_developer = self.developers[0]
        
        # 報酬計算
        reward = self._calculate_reward(predicted_developer, actual_developer, current_task)
        
        # 履歴更新
        success = (predicted_developer == actual_developer)
        self.prediction_history.append({
            'task_idx': self.current_task_idx,
            'predicted': predicted_developer,
            'actual': actual_developer,
            'success': success,
            'reward': reward
        })
        
        self.recent_successes.append(success)
        if len(self.recent_successes) > 50:  # 直近50件の成功率
            self.recent_successes.pop(0)
        
        self.success_rate = np.mean(self.recent_successes) if self.recent_successes else 0.0
        
        # 次のタスクへ
        self.current_task_idx += 1
        self.time_step += 1
        
        # 次の観測
        obs = self._get_observation()
        
        # 終了判定
        terminated = (self.current_task_idx >= len(self.training_pairs))
        truncated = False
        
        info = {
            'success': success,
            'predicted_dev': predicted_developer,
            'actual_dev': actual_developer,
            'success_rate': self.success_rate,
            'extraction_method': self.extraction_method
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, predicted_dev, actual_dev, task_data):
        """報酬を計算"""
        base_reward = 1.0 if predicted_dev == actual_dev else 0.0
        
        # 抽出方法特有のボーナス
        method_bonus = 0.0
        if self.extraction_method == 'assignees' and base_reward > 0:
            # assigneesは高品質だがカバレッジが狭い
            method_bonus = 0.5
        elif self.extraction_method == 'creators' and base_reward > 0:
            # creatorsは幅広いカバレッジ
            method_bonus = 0.3
        elif self.extraction_method == 'all' and base_reward > 0:
            # allはバランス型
            method_bonus = 0.4
        
        # 開発者の活動レベルによる調整
        dev_activity = self.developer_stats.get(actual_dev, 1)
        activity_factor = min(1.0, dev_activity / 10)  # 正規化
        
        # 最終報酬
        final_reward = base_reward + (method_bonus * activity_factor)
        
        return final_reward
    
    def _get_observation(self):
        """現在の観測を取得"""
        if self.current_task_idx >= len(self.training_pairs):
            return np.zeros(20, dtype=np.float32)
        
        current_pair = self.training_pairs[self.current_task_idx]
        current_task = current_pair['task_data']
        
        # 1. タスク特徴量 (10次元)
        task_features = self._extract_task_features(current_task)
        
        # 2. 開発者プール統計 (5次元)
        pool_stats = self._extract_pool_statistics()
        
        # 3. 履歴統計 (5次元)
        history_stats = self._extract_history_statistics()
        
        # 全特徴量を結合
        obs = np.concatenate([
            task_features,
            pool_stats,
            history_stats
        ]).astype(np.float32)
        
        return obs
    
    def _extract_task_features(self, task_data):
        """タスクから特徴量を抽出"""
        features = self.recommender.extract_basic_features(task_data)
        
        # 正規化された特徴量ベクトル (10次元)
        normalized_features = np.array([
            min(1.0, features.get('title_length', 0) / 100),
            min(1.0, features.get('body_length', 0) / 1000),
            min(1.0, features.get('comments_count', 0) / 20),
            features.get('is_bug', 0),
            features.get('is_enhancement', 0),
            features.get('is_documentation', 0),
            features.get('is_question', 0),
            features.get('is_help_wanted', 0),
            min(1.0, features.get('label_count', 0) / 10),
            features.get('is_open', 0)
        ], dtype=np.float32)
        
        return normalized_features
    
    def _extract_pool_statistics(self):
        """開発者プールの統計を抽出 (5次元)"""
        total_devs = len(self.developers)
        avg_activity = np.mean(list(self.developer_stats.values())) if self.developer_stats else 0
        max_activity = max(self.developer_stats.values()) if self.developer_stats else 0
        
        return np.array([
            min(1.0, total_devs / 100),        # 正規化された開発者数
            min(1.0, avg_activity / 50),       # 正規化された平均活動度
            min(1.0, max_activity / 200),      # 正規化された最大活動度
            self.time_step / max(1, len(self.training_pairs)),  # 進捗率
            min(1.0, self.current_task_idx / 100)  # 正規化されたタスクインデックス
        ], dtype=np.float32)
    
    def _extract_history_statistics(self):
        """履歴統計を抽出 (5次元)"""
        if not self.prediction_history:
            return np.zeros(5, dtype=np.float32)
        
        recent_history = self.prediction_history[-20:]  # 直近20件
        
        success_rate = np.mean([h['success'] for h in recent_history])
        avg_reward = np.mean([h['reward'] for h in recent_history])
        
        # 開発者の多様性（直近で何人の異なる開発者を予測したか）
        predicted_devs = set([h['predicted'] for h in recent_history])
        diversity = len(predicted_devs) / len(recent_history) if recent_history else 0
        
        return np.array([
            success_rate,
            avg_reward,
            diversity,
            self.success_rate,  # 全体の成功率
            min(1.0, len(self.prediction_history) / 100)  # 履歴の長さ
        ], dtype=np.float32)


class MultiMethodRLRecommender:
    """マルチメソッド強化学習推薦システム"""
    
    def __init__(self):
        self.extraction_methods = ['assignees', 'creators', 'all']
        self.agents = {}
        self.environments = {}
        self.results = {}
        
        print("🎯 マルチメソッド強化学習推薦システム初期化")
    
    def train_all_methods(self, timesteps=10000):
        """全ての抽出方法で独立してエージェントを訓練"""
        print("🚀 マルチメソッド訓練開始")
        print("=" * 60)
        
        for method in self.extraction_methods:
            print(f"\n📚 {method.upper()}メソッド訓練開始...")
            
            # 環境作成
            env = MultiMethodRLEnvironment(extraction_method=method)
            self.environments[method] = env
            
            # PPOエージェント作成
            agent = PPO(
                "MlpPolicy",
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=None  # ローカルログのみ
            )
            
            # 訓練実行
            print(f"🏋️ {method}エージェント訓練中... ({timesteps:,} ステップ)")
            agent.learn(total_timesteps=timesteps)
            
            self.agents[method] = agent
            
            print(f"✅ {method}エージェント訓練完了")
        
        print("\n🎉 全メソッド訓練完了")
    
    def evaluate_all_methods(self):
        """全ての方法でテストデータを評価"""
        print("📊 マルチメソッド評価開始")
        print("=" * 60)
        
        # テストデータ準備
        recommender = SimpleSimilarityRecommender('configs/unified_rl.yaml')
        training_data, test_data = recommender.load_data('data/backlog.json')
        
        for method in self.extraction_methods:
            print(f"\n🔍 {method.upper()}メソッド評価中...")
            
            # テスト用の環境とエージェント
            env = self.environments[method]
            agent = self.agents[method]
            
            # テストデータでの評価
            test_results = self._evaluate_method(method, agent, env, test_data)
            self.results[method] = test_results
            
            print(f"   精度: {test_results['accuracy']:.3f}")
            print(f"   予測数: {test_results['total_predictions']}")
            print(f"   対象開発者数: {test_results['unique_developers']}")
    
    def _evaluate_method(self, method, agent, env, test_data):
        """特定の方法でテストデータを評価"""
        # テストデータから該当する抽出方法でのペアを作成
        recommender = SimpleSimilarityRecommender('configs/unified_rl.yaml')
        test_assignments = {}
        
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            # この抽出方法で開発者を抽出
            developers = recommender.extract_developers_from_task(task_data, method=method)
            if developers:
                developers.sort(key=lambda x: x['priority'])
                selected_dev = developers[0]
                test_assignments[task_id] = selected_dev['login']
        
        print(f"   {method}でのテスト割り当て: {len(test_assignments)} タスク")
        
        # 実際の予測実行
        predictions = {}
        correct_predictions = 0
        
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if task_id not in test_assignments:
                continue
            
            # エージェントによる予測
            obs = self._create_test_observation(task_data, env)
            action, _ = agent.predict(obs, deterministic=True)
            
            if action < len(env.developers):
                predicted_dev = env.developers[action]
                predictions[task_id] = predicted_dev
                
                actual_dev = test_assignments[task_id]
                if predicted_dev == actual_dev:
                    correct_predictions += 1
        
        # 結果集計
        accuracy = correct_predictions / len(test_assignments) if test_assignments else 0.0
        unique_developers = len(set(test_assignments.values()))
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': len(test_assignments),
            'unique_developers': unique_developers,
            'predictions': predictions,
            'test_assignments': test_assignments
        }
    
    def _create_test_observation(self, task_data, env):
        """テストタスク用の観測を作成"""
        # タスク特徴量
        task_features = env._extract_task_features(task_data)
        
        # 環境統計（訓練時の統計を使用）
        pool_stats = env._extract_pool_statistics()
        
        # 履歴統計（テスト時は初期値）
        history_stats = np.zeros(5, dtype=np.float32)
        
        obs = np.concatenate([task_features, pool_stats, history_stats])
        return obs
    
    def compare_results(self):
        """全メソッドの結果を比較"""
        print("\n📈 マルチメソッド比較結果")
        print("=" * 60)
        
        comparison_data = []
        
        for method in self.extraction_methods:
            if method in self.results:
                result = self.results[method]
                comparison_data.append({
                    'method': method,
                    'accuracy': result['accuracy'],
                    'predictions': result['total_predictions'],
                    'developers': result['unique_developers'],
                    'coverage': result['total_predictions'] / max(1, sum(r['total_predictions'] for r in self.results.values())) * 100
                })
        
        # ソート（精度順）
        comparison_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print("方法         | 精度   | 予測数 | 開発者数 | カバレッジ")
        print("-" * 55)
        for data in comparison_data:
            print(f"{data['method']:12} | {data['accuracy']:.3f} | {data['predictions']:6} | {data['developers']:8} | {data['coverage']:6.1f}%")
        
        # 詳細分析
        print(f"\n🔍 詳細分析:")
        for method in self.extraction_methods:
            if method in self.results:
                result = self.results[method]
                dev_counts = Counter(result['test_assignments'].values())
                print(f"\n{method.upper()}:")
                print(f"  上位開発者:")
                for dev, count in dev_counts.most_common(5):
                    print(f"    {dev}: {count} タスク")
        
        return comparison_data
    
    def save_results(self, output_dir="outputs"):
        """結果を保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 各方法の結果保存
        for method in self.extraction_methods:
            if method in self.results:
                # メトリクス保存
                metrics_path = output_dir / f"multi_method_{method}_metrics_{timestamp}.json"
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    # numpy型を変換
                    result = self.results[method].copy()
                    result.pop('predictions', None)  # 予測結果は別ファイル
                    result.pop('test_assignments', None)
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # モデル保存
                agent_path = output_dir / f"multi_method_{method}_agent_{timestamp}.zip"
                self.agents[method].save(agent_path)
                
                print(f"✅ {method}結果保存: {metrics_path}")
                print(f"✅ {method}モデル保存: {agent_path}")


def main():
    parser = argparse.ArgumentParser(description='マルチメソッド強化学習タスク推薦システム')
    parser.add_argument('--timesteps', type=int, default=10000,
                       help='各メソッドの訓練ステップ数')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # マルチメソッド強化学習システム実行
    recommender = MultiMethodRLRecommender()
    
    # 全メソッドで訓練
    recommender.train_all_methods(timesteps=args.timesteps)
    
    # 全メソッドで評価
    recommender.evaluate_all_methods()
    
    # 結果比較
    comparison_results = recommender.compare_results()
    
    # 結果保存
    recommender.save_results(args.output)
    
    print(f"\n🎯 最終結果:")
    best_method = max(comparison_results, key=lambda x: x['accuracy'])
    print(f"   最高精度: {best_method['method']} ({best_method['accuracy']:.3f})")
    print(f"   最大カバレッジ: {max(comparison_results, key=lambda x: x['predictions'])['method']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
