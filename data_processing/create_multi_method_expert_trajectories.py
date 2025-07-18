#!/usr/bin/env python3
"""
マルチメソッドエキスパート軌跡生成

3つの抽出方法（assignees, creators, all）でそれぞれ独立した
エキスパート軌跡を生成し、逆強化学習で使用する。

これにより、各抽出方法に特化した報酬関数を学習できる。
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.simple_similarity_recommender import SimpleSimilarityRecommender


class MultiMethodExpertTrajectoryGenerator:
    """マルチメソッドエキスパート軌跡生成器"""
    
    def __init__(self, config_path="configs/multi_method_training.yaml"):
        self.config_path = config_path
        
        # SimpleSimilarityRecommenderを使用してデータ処理
        self.recommender = SimpleSimilarityRecommender('configs/unified_rl.yaml')
        
        # 抽出方法
        self.extraction_methods = ['assignees', 'creators', 'all']
        
        print("🎯 マルチメソッドエキスパート軌跡生成器初期化完了")
    
    def generate_all_expert_trajectories(self, data_path="data/backlog.json"):
        """全ての抽出方法でエキスパート軌跡を生成"""
        print("🚀 マルチメソッドエキスパート軌跡生成開始")
        print("=" * 70)
        
        # データ読み込み
        training_data, test_data = self.recommender.load_data(data_path)
        
        results = {}
        
        for method in self.extraction_methods:
            print(f"\n📚 {method.upper()}メソッドのエキスパート軌跡生成...")
            
            # 各方法でエキスパート軌跡を生成
            expert_trajectories = self.generate_expert_trajectories(
                training_data, method
            )
            
            # 保存
            output_path = f"data/expert_trajectories_{method}.pkl"
            self.save_expert_trajectories(expert_trajectories, output_path)
            
            results[method] = {
                'trajectories': len(expert_trajectories),
                'output_path': output_path,
                'unique_developers': len(set([traj['developer'] for traj in expert_trajectories])),
                'total_actions': sum([len(traj['actions']) for traj in expert_trajectories])
            }
            
            print(f"   軌跡数: {results[method]['trajectories']}")
            print(f"   開発者数: {results[method]['unique_developers']}")
            print(f"   総アクション数: {results[method]['total_actions']}")
            print(f"   保存先: {output_path}")
        
        # 結果比較
        self.compare_trajectories(results)
        
        return results
    
    def generate_expert_trajectories(self, training_data, extraction_method):
        """特定の抽出方法でエキスパート軌跡を生成"""
        
        # 学習ペアを抽出
        training_pairs, developer_stats = self.recommender.extract_training_pairs(
            training_data, extraction_method=extraction_method
        )
        
        if not training_pairs:
            print(f"⚠️ {extraction_method}で学習ペアが見つかりませんでした")
            return []
        
        # 開発者別に軌跡を構築
        developer_trajectories = defaultdict(list)
        
        for pair in training_pairs:
            developer = pair['developer']
            task_data = pair['task_data']
            
            # タスクの状態ベクトルを作成
            state_vector = self._create_state_vector(task_data, extraction_method)
            
            # 開発者のアクション（このタスクを選択）
            action = {
                'task_id': pair['task_id'],
                'task_data': task_data,
                'state': state_vector,
                'timestamp': task_data.get('created_at', ''),
                'extraction_source': pair.get('extraction_source', extraction_method)
            }
            
            developer_trajectories[developer].append(action)
        
        # 軌跡形式に変換
        expert_trajectories = []
        
        for developer, actions in developer_trajectories.items():
            if len(actions) >= 2:  # 最低2つのアクションが必要
                # 時系列順にソート
                actions.sort(key=lambda x: x['timestamp'])
                
                trajectory = {
                    'developer': developer,
                    'extraction_method': extraction_method,
                    'actions': actions,
                    'total_tasks': len(actions),
                    'timespan': self._calculate_timespan(actions),
                    'activity_score': len(actions) / max(1, developer_stats.get(developer, 1))
                }
                
                expert_trajectories.append(trajectory)
        
        # アクティビティスコア順にソート
        expert_trajectories.sort(key=lambda x: x['activity_score'], reverse=True)
        
        return expert_trajectories
    
    def _create_state_vector(self, task_data, extraction_method):
        """タスクの状態ベクトルを作成"""
        
        # 基本特徴量
        basic_features = self.recommender.extract_basic_features(task_data)
        
        # 正規化された特徴量ベクトル
        state_vector = np.array([
            min(1.0, basic_features.get('title_length', 0) / 100),
            min(1.0, basic_features.get('body_length', 0) / 1000),
            min(1.0, basic_features.get('comments_count', 0) / 20),
            basic_features.get('is_bug', 0),
            basic_features.get('is_enhancement', 0),
            basic_features.get('is_documentation', 0),
            basic_features.get('is_question', 0),
            basic_features.get('is_help_wanted', 0),
            min(1.0, basic_features.get('label_count', 0) / 10),
            basic_features.get('is_open', 0),
            # 抽出方法別の特徴量
            self._get_extraction_method_features(extraction_method)
        ], dtype=np.float32)
        
        return state_vector
    
    def _get_extraction_method_features(self, extraction_method):
        """抽出方法別の特徴量"""
        if extraction_method == 'assignees':
            return 1.0  # 高品質・狭いカバレッジ
        elif extraction_method == 'creators':
            return 0.5  # 広いカバレッジ・中品質
        elif extraction_method == 'all':
            return 0.8  # バランス型
        else:
            return 0.0
    
    def _calculate_timespan(self, actions):
        """アクション列の時間スパンを計算"""
        if len(actions) < 2:
            return 0
        
        try:
            first_time = datetime.fromisoformat(actions[0]['timestamp'].replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(actions[-1]['timestamp'].replace('Z', '+00:00'))
            return (last_time - first_time).days
        except:
            return 0
    
    def save_expert_trajectories(self, expert_trajectories, output_path):
        """エキスパート軌跡を保存"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(expert_trajectories, f)
        
        print(f"✅ エキスパート軌跡保存: {output_path}")
    
    def compare_trajectories(self, results):
        """軌跡の比較分析"""
        print("\n📈 エキスパート軌跡比較分析")
        print("=" * 70)
        
        print("方法         | 軌跡数 | 開発者数 | 総アクション | 平均アクション/軌跡")
        print("-" * 70)
        
        for method in self.extraction_methods:
            if method in results:
                result = results[method]
                avg_actions = result['total_actions'] / max(1, result['trajectories'])
                print(f"{method:12} | {result['trajectories']:6} | {result['unique_developers']:8} | {result['total_actions']:12} | {avg_actions:15.1f}")
        
        # 品質分析
        print(f"\n🔍 品質分析:")
        for method in self.extraction_methods:
            if method in results:
                result = results[method]
                
                # カバレッジ分析
                coverage = result['total_actions'] / max(1, sum(r['total_actions'] for r in results.values())) * 100
                
                # 密度分析
                density = result['total_actions'] / max(1, result['unique_developers'])
                
                print(f"\n{method.upper()}:")
                print(f"  カバレッジ: {coverage:.1f}%")
                print(f"  開発者あたり軌跡密度: {density:.1f}")
                
                if method == 'assignees':
                    print(f"  特徴: 高品質・狭いカバレッジ（正式割り当てのみ）")
                elif method == 'creators':
                    print(f"  特徴: 広いカバレッジ・中品質（Issue/PR作成者も含む）")
                elif method == 'all':
                    print(f"  特徴: バランス型（assignees優先 + creators補完）")
    
    def load_and_analyze_trajectories(self, method):
        """保存された軌跡を読み込んで分析"""
        trajectory_path = f"data/expert_trajectories_{method}.pkl"
        
        try:
            with open(trajectory_path, 'rb') as f:
                trajectories = pickle.load(f)
            
            print(f"\n📊 {method.upper()}軌跡詳細分析:")
            print(f"   軌跡数: {len(trajectories)}")
            
            # 開発者別統計
            dev_stats = Counter([traj['developer'] for traj in trajectories])
            print(f"   上位開発者:")
            for dev, count in dev_stats.most_common(5):
                avg_tasks = np.mean([traj['total_tasks'] for traj in trajectories if traj['developer'] == dev])
                print(f"     {dev}: {count} 軌跡, 平均 {avg_tasks:.1f} タスク/軌跡")
            
            # アクティビティ分析
            activity_scores = [traj['activity_score'] for traj in trajectories]
            print(f"   アクティビティスコア: 平均 {np.mean(activity_scores):.3f}, 標準偏差 {np.std(activity_scores):.3f}")
            
            # 時間スパン分析
            timespans = [traj['timespan'] for traj in trajectories if traj['timespan'] > 0]
            if timespans:
                print(f"   時間スパン: 平均 {np.mean(timespans):.1f} 日, 最大 {max(timespans)} 日")
            
            return trajectories
            
        except FileNotFoundError:
            print(f"⚠️ 軌跡ファイルが見つかりません: {trajectory_path}")
            return None
        except Exception as e:
            print(f"⚠️ 軌跡読み込みエラー: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='マルチメソッドエキスパート軌跡生成')
    parser.add_argument('--data', default='data/backlog.json',
                       help='バックログデータパス')
    parser.add_argument('--config', default='configs/multi_method_training.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--analyze', action='store_true',
                       help='既存軌跡の分析のみ実行')
    parser.add_argument('--method', choices=['assignees', 'creators', 'all'],
                       help='特定の方法のみ処理')
    
    args = parser.parse_args()
    
    # エキスパート軌跡生成器
    generator = MultiMethodExpertTrajectoryGenerator(args.config)
    
    if args.analyze:
        # 既存軌跡の分析
        if args.method:
            generator.load_and_analyze_trajectories(args.method)
        else:
            for method in ['assignees', 'creators', 'all']:
                generator.load_and_analyze_trajectories(method)
    else:
        # 軌跡生成
        if args.method:
            # 特定の方法のみ
            training_data, _ = generator.recommender.load_data(args.data)
            trajectories = generator.generate_expert_trajectories(training_data, args.method)
            output_path = f"data/expert_trajectories_{args.method}.pkl"
            generator.save_expert_trajectories(trajectories, output_path)
        else:
            # 全ての方法
            results = generator.generate_all_expert_trajectories(args.data)
    
    return 0


if __name__ == "__main__":
    exit(main())
