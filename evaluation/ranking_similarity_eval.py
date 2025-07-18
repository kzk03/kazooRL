#!/usr/bin/env python3
"""
ランキング類似度評価システム

開発者プールが異なっても、特徴ベースの類似度でランキング評価を実行
"""

import argparse
import json

# プロジェクトのモジュール
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Stable-Baselines3
from stable_baselines3 import PPO

sys.path.append('/Users/kazuki-h/rl/kazoo')
sys.path.append('/Users/kazuki-h/rl/kazoo/src')

from scripts.train_simple_unified_rl import SimpleTaskAssignmentEnv
from src.kazoo.envs.task import Task


class RankingSimilarityEvaluator:
    """ランキング類似度での評価システム"""
    
    def __init__(self, config_path, model_path, test_data_path):
        """
        Args:
            config_path: 設定ファイルパス
            model_path: 学習済みモデルパス
            test_data_path: テストデータパス
        """
        self.config_path = config_path
        self.model_path = model_path
        self.test_data_path = test_data_path
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # テストデータ読み込み（2023年のみ）
        with open(test_data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        self.test_data = []
        for task in all_data:
            created_at = task.get('created_at', '')
            if created_at.startswith('2023'):
                self.test_data.append(task)
        
        print(f"📊 全データ: {len(all_data):,} タスク")
        print(f"📊 2023年テストデータ: {len(self.test_data):,} タスク")
        
        # 環境初期化
        self._setup_environment()
        
        # モデル読み込み
        self.model = PPO.load(model_path)
        print(f"✅ モデル読み込み完了: {model_path}")
    
    def _setup_environment(self):
        """環境の初期化"""
        print("🎮 環境初期化中...")
        
        # 開発者プロファイル読み込み
        dev_profiles_path = self.config['env']['dev_profiles_path']
        with open(dev_profiles_path, 'r', encoding='utf-8') as f:
            self.dev_profiles = yaml.safe_load(f)
        
        # 学習に使用したバックログデータも読み込み
        backlog_path = self.config['env']['backlog_path']
        with open(backlog_path, 'r', encoding='utf-8') as f:
            training_backlog = json.load(f)
        
        # 設定オブジェクトの作成
        class DictObj:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, DictObj(v))
                    else:
                        setattr(self, k, v)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        cfg = DictObj(self.config)
        
        # 環境作成（学習時のデータ構成を使用）
        self.env = SimpleTaskAssignmentEnv(
            cfg=cfg,
            backlog_data=training_backlog,
            dev_profiles_data=self.dev_profiles
        )
        
        print(f"   開発者数: {self.env.num_developers}")
        print(f"   タスク数: {len(self.env.tasks)}")
        print(f"   特徴量次元: {self.env.observation_space.shape[0]}")
    
    def extract_actual_assignments(self):
        """実際の開発者割り当てを抽出"""
        print("📋 実際の割り当て抽出中...")
        
        actual_assignments = {}
        actual_dev_profiles = {}
        assignment_stats = defaultdict(int)
        developer_stats = Counter()
        
        for task_data in self.test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            # 実際の担当者を抽出
            assignee = None
            
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            
            if assignee:
                actual_assignments[task_id] = assignee
                assignment_stats['assigned'] += 1
                developer_stats[assignee] += 1
                
                # 実際の開発者のプロファイル（簡易版）を作成
                if assignee not in actual_dev_profiles:
                    actual_dev_profiles[assignee] = {
                        'name': assignee,
                        'task_count': 0,
                        'skill_areas': set(),
                        'collaboration_history': []
                    }
                
                actual_dev_profiles[assignee]['task_count'] += 1
                
                # ラベルからスキル推定
                for label in task_data.get('labels', []):
                    if isinstance(label, dict):
                        label_name = label.get('name', '')
                    else:
                        label_name = str(label)
                    actual_dev_profiles[assignee]['skill_areas'].add(label_name)
            else:
                assignment_stats['unassigned'] += 1
        
        print(f"   割り当て済み: {assignment_stats['assigned']:,} タスク")
        print(f"   未割り当て: {assignment_stats['unassigned']:,} タスク")
        print(f"   ユニーク開発者: {len(developer_stats)} 人")
        
        # 上位開発者表示
        top_devs = developer_stats.most_common(10)
        print("   上位開発者:")
        for dev, count in top_devs:
            print(f"     {dev}: {count} タスク")
        
        return actual_assignments, actual_dev_profiles
    
    def calculate_developer_similarity(self, actual_dev_profiles):
        """実際の開発者と環境の開発者間の類似度を計算"""
        print("🔍 開発者類似度計算中...")
        
        # 実際の開発者のスキルベクトルを作成
        all_skills = set()
        for profile in actual_dev_profiles.values():
            all_skills.update(profile['skill_areas'])
        
        skill_list = sorted(list(all_skills))
        print(f"   識別されたスキル: {len(skill_list)} 種類")
        
        # 実際の開発者のスキルベクトル
        actual_vectors = {}
        for dev_name, profile in actual_dev_profiles.items():
            vector = np.zeros(len(skill_list))
            for i, skill in enumerate(skill_list):
                if skill in profile['skill_areas']:
                    vector[i] = 1.0
            actual_vectors[dev_name] = vector
        
        # 環境の開発者のスキルベクトル（プロファイルから）
        env_vectors = {}
        for dev_name in self.env.developers:
            dev_profile = self.dev_profiles.get(dev_name, {})
            vector = np.zeros(len(skill_list))
            
            # プロファイルからスキルを抽出
            dev_skills = set()
            if 'skills' in dev_profile:
                dev_skills.update(dev_profile['skills'])
            if 'expertise' in dev_profile:
                dev_skills.update(dev_profile['expertise'])
            
            for i, skill in enumerate(skill_list):
                if skill in dev_skills:
                    vector[i] = 1.0
            
            env_vectors[dev_name] = vector
        
        # 類似度マトリックス計算
        similarity_matrix = {}
        for actual_dev, actual_vec in actual_vectors.items():
            similarities = {}
            for env_dev, env_vec in env_vectors.items():
                # コサイン類似度
                if np.linalg.norm(actual_vec) > 0 and np.linalg.norm(env_vec) > 0:
                    cosine_sim = np.dot(actual_vec, env_vec) / (
                        np.linalg.norm(actual_vec) * np.linalg.norm(env_vec)
                    )
                else:
                    cosine_sim = 0.0
                
                # ジャッカード類似度
                actual_skills = set(skill for i, skill in enumerate(skill_list) if actual_vec[i] > 0)
                env_skills = set(skill for i, skill in enumerate(skill_list) if env_vec[i] > 0)
                
                if len(actual_skills) > 0 or len(env_skills) > 0:
                    jaccard_sim = len(actual_skills & env_skills) / len(actual_skills | env_skills)
                else:
                    jaccard_sim = 0.0
                
                # 複合類似度（コサイン + ジャッカード）
                combined_sim = (cosine_sim + jaccard_sim) / 2.0
                similarities[env_dev] = {
                    'cosine': cosine_sim,
                    'jaccard': jaccard_sim,
                    'combined': combined_sim
                }
            
            similarity_matrix[actual_dev] = similarities
        
        return similarity_matrix
    
    def predict_with_ranking(self, actual_assignments, actual_dev_profiles):
        """各タスクに対して開発者をランキング予測"""
        print("🤖 ランキング予測中...")
        
        predictions = {}
        ranking_results = {}
        
        # 類似度マトリックス計算
        similarity_matrix = self.calculate_developer_similarity(actual_dev_profiles)
        
        # 実際に割り当てられた2023年のタスクを対象に予測
        test_tasks_with_assignments = []
        for task in self.test_data:
            task_id = task.get('id') or task.get('number')
            if task_id and task_id in actual_assignments:
                test_tasks_with_assignments.append(task)
        
        print(f"   予測対象: {len(test_tasks_with_assignments)} タスク（2023年）")
        
        for task_data in test_tasks_with_assignments:
            try:
                task_obj = Task(task_data)
                task_id = task_obj.id if hasattr(task_obj, 'id') else task_data.get('id', task_data.get('number'))
                actual_dev = actual_assignments[task_id]
                
                # ダミー環境
                dummy_env = type('DummyEnv', (), {
                    'backlog': self.env.tasks,
                    'dev_profiles': self.dev_profiles,
                    'assignments': {},
                    'dev_action_history': {}
                })()
                
                # 各環境開発者に対する予測確率を計算
                dev_predictions = []
                for dev_idx, env_dev_name in enumerate(self.env.developers):
                    env_dev_profile = self.dev_profiles.get(env_dev_name, {})
                    dev_obj = {"name": env_dev_name, "profile": env_dev_profile}
                    
                    # 特徴量を抽出
                    features = self.env.feature_extractor.get_features(task_obj, dev_obj, dummy_env)
                    obs = features.astype(np.float32)
                    
                    # モデルで予測確率を取得
                    obs_tensor = torch.tensor(obs).unsqueeze(0).float()
                    with torch.no_grad():
                        logits = self.model.policy.mlp_extractor.policy_net(
                            self.model.policy.features_extractor(obs_tensor)
                        )
                        probs = torch.softmax(logits, dim=-1).numpy()[0]
                        dev_prob = probs[dev_idx] if dev_idx < len(probs) else 0.0
                    
                    dev_predictions.append((dev_idx, env_dev_name, dev_prob))
                
                # 予測確率でソート
                dev_predictions.sort(key=lambda x: x[2], reverse=True)
                
                # ランキング類似度計算
                ranking_scores = {}
                
                if actual_dev in similarity_matrix:
                    # 実際の開発者に最も類似した環境開発者のランキング位置
                    similarities = similarity_matrix[actual_dev]
                    
                    # 最も類似した開発者を見つける
                    best_match = max(similarities.items(), key=lambda x: x[1]['combined'])
                    best_env_dev = best_match[0]
                    best_similarity = best_match[1]['combined']
                    
                    # ランキング位置を見つける
                    for rank, (idx, env_dev, prob) in enumerate(dev_predictions):
                        if env_dev == best_env_dev:
                            ranking_position = rank + 1  # 1-based
                            ranking_scores['best_match_rank'] = ranking_position
                            ranking_scores['best_match_similarity'] = best_similarity
                            ranking_scores['best_match_dev'] = best_env_dev
                            break
                    
                    # トップ5内の類似度重み付きスコア
                    top5_score = 0.0
                    for rank, (idx, env_dev, prob) in enumerate(dev_predictions[:5]):
                        if env_dev in similarities:
                            weight = 1.0 / (rank + 1)  # 位置による重み
                            similarity = similarities[env_dev]['combined']
                            top5_score += weight * similarity
                    
                    ranking_scores['top5_weighted_similarity'] = top5_score
                
                predictions[task_id] = dev_predictions[0][1]  # トップ予測
                ranking_results[task_id] = {
                    'actual_developer': actual_dev,
                    'predicted_ranking': [(env_dev, prob) for _, env_dev, prob in dev_predictions[:10]],
                    'ranking_scores': ranking_scores
                }
                
            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測でエラー: {e}")
                continue
        
        print(f"   予測完了: {len(predictions)} タスク")
        return predictions, ranking_results
    
    def calculate_ranking_metrics(self, ranking_results):
        """ランキング類似度メトリクスを計算"""
        print("📊 ランキング類似度評価中...")
        
        metrics = {}
        
        # ランキング位置の統計
        best_match_ranks = []
        best_match_similarities = []
        top5_weighted_similarities = []
        
        for task_id, result in ranking_results.items():
            scores = result.get('ranking_scores', {})
            
            if 'best_match_rank' in scores:
                best_match_ranks.append(scores['best_match_rank'])
                best_match_similarities.append(scores['best_match_similarity'])
            
            if 'top5_weighted_similarity' in scores:
                top5_weighted_similarities.append(scores['top5_weighted_similarity'])
        
        if best_match_ranks:
            metrics['avg_best_match_rank'] = np.mean(best_match_ranks)
            metrics['median_best_match_rank'] = np.median(best_match_ranks)
            metrics['top5_rank_ratio'] = sum(1 for r in best_match_ranks if r <= 5) / len(best_match_ranks)
            metrics['top10_rank_ratio'] = sum(1 for r in best_match_ranks if r <= 10) / len(best_match_ranks)
            
            print(f"   最類似開発者の平均ランク: {metrics['avg_best_match_rank']:.2f}")
            print(f"   最類似開発者の中央値ランク: {metrics['median_best_match_rank']:.1f}")
            print(f"   Top-5ランク率: {metrics['top5_rank_ratio']:.3f}")
            print(f"   Top-10ランク率: {metrics['top10_rank_ratio']:.3f}")
        
        if best_match_similarities:
            metrics['avg_best_match_similarity'] = np.mean(best_match_similarities)
            metrics['similarity_std'] = np.std(best_match_similarities)
            
            print(f"   平均最類似度: {metrics['avg_best_match_similarity']:.3f} ± {metrics['similarity_std']:.3f}")
        
        if top5_weighted_similarities:
            metrics['avg_top5_weighted_similarity'] = np.mean(top5_weighted_similarities)
            print(f"   Top-5重み付き類似度: {metrics['avg_top5_weighted_similarity']:.3f}")
        
        # ランキング類似度総合スコア
        if best_match_ranks and best_match_similarities:
            # 正規化されたランクスコア（1位なら1.0、下位ほど0に近づく）
            normalized_ranks = [1.0 - (r - 1) / 149 for r in best_match_ranks]  # 150人中
            combined_scores = [r * s for r, s in zip(normalized_ranks, best_match_similarities)]
            
            metrics['ranking_similarity_score'] = np.mean(combined_scores)
            print(f"   ランキング類似度総合スコア: {metrics['ranking_similarity_score']:.3f}")
        
        return metrics
    
    def run_evaluation(self, output_dir="outputs"):
        """評価実行"""
        print("🚀 ランキング類似度評価開始")
        print("=" * 70)
        
        # 1. 実際の割り当て抽出
        actual_assignments, actual_dev_profiles = self.extract_actual_assignments()
        
        # 2. ランキング予測
        predictions, ranking_results = self.predict_with_ranking(actual_assignments, actual_dev_profiles)
        
        # 3. ランキング類似度メトリクス計算
        ranking_metrics = self.calculate_ranking_metrics(ranking_results)
        
        # 4. 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # メトリクス保存
        metrics_path = output_dir / f"ranking_similarity_metrics_{timestamp}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(ranking_metrics, f, indent=2, ensure_ascii=False)
        
        # 詳細結果保存
        results_path = output_dir / f"ranking_similarity_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # numpy配列を対応
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            
            json.dump(ranking_results, f, indent=2, ensure_ascii=False, default=convert_numpy)
        
        print(f"✅ メトリクス保存: {metrics_path}")
        print(f"✅ 詳細結果保存: {results_path}")
        
        return ranking_metrics


def main():
    parser = argparse.ArgumentParser(description='ランキング類似度評価')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--model', default='models/simple_unified_rl_agent.zip',
                       help='学習済みモデルパス')
    parser.add_argument('--test-data', default='data/backlog.json',
                       help='テストデータパス（統合済み）')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 評価実行
    evaluator = RankingSimilarityEvaluator(args.config, args.model, args.test_data)
    metrics = evaluator.run_evaluation(args.output)
    
    if metrics:
        print("\n🎯 主要結果:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
