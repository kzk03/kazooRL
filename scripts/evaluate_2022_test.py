#!/usr/bin/env python3
"""
学習済みモデルを2022年テストデータで評価するスクリプト
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# パッケージのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import FeatureExtractor


class SimpleConfig:
    """辞書をオブジェクトのように扱うためのクラス"""
    def __init__(self, config_dict):
        self._dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)
    
    def get(self, key, default=None):
        """辞書のgetメソッドと同様の動作"""
        return self._dict.get(key, default)


def load_config(config_path):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return SimpleConfig(config_dict)


def evaluate_recommendations(backlog_data, dev_profiles_data, feature_extractor, learned_weights, num_recommendations=5):
    """
    学習済み重みを使って推薦システムを評価する（生のJSONデータを使用）
    
    Returns:
        dict: 評価結果（accuracy, precision, recall, etc.）
    """
    # タスクオブジェクトの簡易版を作成
    class MockTask:
        def __init__(self, task_data):
            self.id = task_data.get('id')
            self.title = task_data.get('title', '')
            self.body = task_data.get('body', '')
            self.labels = [label.get('name') for label in task_data.get('labels', [])]
            self.comments = task_data.get('comments', 0)
            self.updated_at = task_data.get('updated_at', '2022-01-01T00:00:00Z')
            self.user = task_data.get('user', {})
            self.assignees = task_data.get('assignees', [])
            
            # 日付文字列をdatetimeオブジェクトに変換
            from datetime import datetime
            if isinstance(self.updated_at, str):
                try:
                    if self.updated_at.endswith('Z'):
                        self.updated_at = self.updated_at[:-1] + '+00:00'
                    self.updated_at = datetime.fromisoformat(self.updated_at)
                except:
                    self.updated_at = datetime(2022, 1, 1)

    # 仮の環境オブジェクトを作成（特徴量抽出に必要）
    class MockEnv:
        def __init__(self, dev_profiles, backlog_data):
            self.dev_profiles = dev_profiles
            self.assignments = defaultdict(set)
            self.dev_action_history = defaultdict(list)
            self.backlog = [MockTask(task_data) for task_data in backlog_data]
    
    results = {
        'total_tasks': 0,
        'tasks_with_assignees': 0,
        'correct_recommendations': 0,
        'top_k_hits': defaultdict(int),
        'recommendation_details': []
    }
    
    print(f"🔍 評価開始: {len(backlog_data)} タスクで評価")
    
    # 担当者情報があるタスクのみを抽出
    tasks_with_assignees = []
    for task in backlog_data:
        if task.get('assignees') and len(task['assignees']) > 0:
            # 担当者がdev_profiles_dataに含まれているかチェック
            assignees = [a.get('login') for a in task['assignees'] if a.get('login')]
            if any(assignee in dev_profiles_data for assignee in assignees):
                tasks_with_assignees.append(task)
    
    print(f"📊 担当者情報があるタスク: {len(tasks_with_assignees)}/{len(backlog_data)}")
    
    # 本格的な評価のため全タスクを使用（高速化のための制限を削除）
    eval_tasks = tasks_with_assignees  # 全てのタスクで評価
    
    print(f"🎯 本格評価: {len(eval_tasks)} タスクで評価実行")
    
    # 環境オブジェクトを作成
    mock_env = MockEnv(dev_profiles_data, backlog_data)
    
    # 評価タスクの進捗バー
    task_progress = tqdm(
        enumerate(eval_tasks),
        total=len(eval_tasks),
        desc="📊 評価進行",
        unit="task",
        colour='green',
        leave=True
    )
    
    for task_idx, task in task_progress:
        # タスクの実際の担当者を取得（Ground Truth）
        actual_assignees = [assignee.get('login') for assignee in task['assignees'] 
                          if assignee.get('login')]
        
        if not actual_assignees:
            task_progress.set_postfix({"Status": "担当者なし (スキップ)"})
            continue
            
        # 各開発者に対するスコアを計算（本格評価のため全開発者を対象）
        developer_scores = []
        available_developers = list(dev_profiles_data.keys())  # 全開発者で評価
        
        mock_task = MockTask(task)
        
        # 開発者評価の進捗バー（内部）
        dev_progress = tqdm(
            available_developers,
            desc=f"Task {task_idx+1:2d}",
            unit="dev",
            leave=False,
            colour='blue'
        )
        
        for dev_name in dev_progress:
            try:
                # 開発者プロファイルを辞書形式で取得
                dev_profile = dev_profiles_data[dev_name]
                developer = {'name': dev_name, 'profile': dev_profile}
                
                # 特徴量を抽出
                features = feature_extractor.get_features(mock_task, developer, mock_env)
                
                # 学習済み重みで重み付けスコアを計算
                score = np.dot(features, learned_weights)
                developer_scores.append((dev_name, score))
                
            except Exception as e:
                # 特徴量抽出でエラーが発生した場合はスキップ
                if task_idx == 0:  # 最初のタスクでのみエラーを表示
                    print(f"⚠️ 開発者 {dev_name} の特徴量抽出でエラー: {e}")
                continue
        
        if not developer_scores:
            task_progress.set_postfix({"Status": "スコア計算失敗"})
            continue
            
        # スコア順にソート（降順）
        developer_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 上位N人の推薦リストを作成
        recommendations = [dev_name for dev_name, score in developer_scores[:num_recommendations]]
        
        # 正解率を計算
        correct_in_top_k = []
        for k in [1, 3, 5]:
            top_k_recs = recommendations[:k]
            hit = any(assignee in top_k_recs for assignee in actual_assignees)
            if hit:
                results['top_k_hits'][f'top_{k}'] += 1
            correct_in_top_k.append(hit)
        
        # 詳細結果を記録
        results['recommendation_details'].append({
            'task_id': task.get('id'),
            'task_title': task.get('title', 'Unknown')[:50],
            'actual_assignees': actual_assignees,
            'recommendations': recommendations,
            'top_scores': [(dev, float(score)) for dev, score in developer_scores[:5]],
            'correct_in_top_1': correct_in_top_k[0],
            'correct_in_top_3': correct_in_top_k[1],
            'correct_in_top_5': correct_in_top_k[2]
        })
        
        results['total_tasks'] += 1
        results['tasks_with_assignees'] += 1
        
        # 進捗バーの情報更新
        if results['total_tasks'] > 0:
            top1_acc = results['top_k_hits']['top_1'] / results['total_tasks']
            top3_acc = results['top_k_hits']['top_3'] / results['total_tasks']
            task_progress.set_postfix({
                "Top-1": f"{top1_acc:.3f}",
                "Top-3": f"{top3_acc:.3f}",
                "完了": f"{results['total_tasks']}/{len(eval_tasks)}"
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='学習済みモデルを2022年テストデータで評価')
    parser.add_argument('--config', required=True, help='設定ファイルのパス')
    parser.add_argument('--learned-weights', required=True, help='学習済み重みファイルのパス')
    parser.add_argument('--output', default='evaluation_results_2022.json', help='結果出力ファイル')
    
    args = parser.parse_args()
    
    print("🚀 2022年テストデータでの評価開始")
    print(f"📝 設定: {args.config}")
    print(f"🎯 学習済み重み: {args.learned_weights}")
    
    # 設定読み込み
    config = load_config(args.config)
    
    # 学習済み重み読み込み
    learned_weights = np.load(args.learned_weights)
    print(f"💪 学習済み重み形状: {learned_weights.shape}")
    
    # バックログとプロファイルデータ読み込み
    print("📚 テストデータ読み込み中...")
    with open(config.env.backlog_path, 'r', encoding='utf-8') as f:
        backlog_data = json.load(f)
    with open(config.env.dev_profiles_path, 'r', encoding='utf-8') as f:
        dev_profiles_data = yaml.safe_load(f)
    
    # 特徴量抽出器初期化
    print("🔧 特徴量抽出器初期化中...")
    feature_extractor = FeatureExtractor(config)
    
    print(f"📊 テストデータ: {len(backlog_data)} タスク, {len(dev_profiles_data)} 開発者")
    
    # 評価実行
    print("🎯 推薦評価実行中...")
    results = evaluate_recommendations(backlog_data, dev_profiles_data, feature_extractor, learned_weights)
    
    # 結果計算
    total_tasks = results['total_tasks']
    if total_tasks > 0:
        accuracy_top_1 = results['top_k_hits']['top_1'] / total_tasks
        accuracy_top_3 = results['top_k_hits']['top_3'] / total_tasks
        accuracy_top_5 = results['top_k_hits']['top_5'] / total_tasks
        
        print("\n" + "="*60)
        print("📈 評価結果")
        print("="*60)
        print(f"評価タスク数: {total_tasks}")
        print(f"Top-1 Accuracy: {accuracy_top_1:.3f} ({results['top_k_hits']['top_1']}/{total_tasks})")
        print(f"Top-3 Accuracy: {accuracy_top_3:.3f} ({results['top_k_hits']['top_3']}/{total_tasks})")
        print(f"Top-5 Accuracy: {accuracy_top_5:.3f} ({results['top_k_hits']['top_5']}/{total_tasks})")
        print("="*60)
        
        # 結果をまとめ
        final_results = {
            'evaluation_config': args.config,
            'learned_weights_path': args.learned_weights,
            'total_tasks_evaluated': total_tasks,
            'tasks_with_assignees': results['tasks_with_assignees'],
            'top_1_accuracy': float(accuracy_top_1),
            'top_3_accuracy': float(accuracy_top_3),
            'top_5_accuracy': float(accuracy_top_5),
            'detailed_results': results['recommendation_details']
        }
        
        # 結果保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 詳細結果を保存: {output_path}")
        
        # サンプル結果表示
        print("\n📋 サンプル推薦結果:")
        for i, detail in enumerate(results['recommendation_details'][:3]):
            print(f"\nタスク {i+1}: {detail['task_title']}")
            print(f"  実際の担当者: {detail['actual_assignees']}")
            print(f"  推薦Top-5: {detail['recommendations']}")
            print(f"  Top-1正解: {'✅' if detail['correct_in_top_1'] else '❌'}")
            print(f"  Top-3正解: {'✅' if detail['correct_in_top_3'] else '❌'}")
    
    else:
        print("⚠️ 評価できるタスクが見つかりませんでした")


if __name__ == "__main__":
    main()
