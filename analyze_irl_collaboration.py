#!/usr/bin/env python3
"""
逆強化学習（IRL）で学習した報酬重みを分析し、
協力関係特徴量の影響を詳しく調べるスクリプト
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def analyze_irl_weights():
    """IRL学習済み重みの分析"""
    
    print("=== IRL報酬重み分析（協力関係特徴量中心） ===\n")
    
    # 学習済み重みを読み込み
    weights_files = [
        "data/learned_weights.npy",
        "data/learned_reward_weights.npy"
    ]
    
    latest_weights = None
    for weights_file in weights_files:
        if Path(weights_file).exists():
            weights = np.load(weights_file)
            print(f"✅ 重みファイル読み込み: {weights_file}")
            print(f"   重みベクトル形状: {weights.shape}")
            print(f"   重みの範囲: [{weights.min():.4f}, {weights.max():.4f}]")
            latest_weights = weights
            break
    
    if latest_weights is None:
        print("❌ 学習済み重みファイルが見つかりません")
        return
    
    # 特徴量名の定義（FeatureExtractorと同じ順序）
    all_labels = ["bug", "enhancement", "documentation", "question", "help wanted"]
    
    feature_names = []
    feature_names.extend([
        "task_days_since_last_activity",
        "task_discussion_activity", 
        "task_text_length",
        "task_code_block_count"
    ])
    feature_names.extend([f"task_label_{label}" for label in all_labels])
    feature_names.extend([
        "dev_recent_activity_count",
        "dev_current_workload", 
        "dev_total_lines_changed"
    ])
    # 社会的つながり特徴量
    feature_names.extend([
        "dev_collaboration_network_size",
        "dev_comment_interactions", 
        "dev_cross_issue_activity"
    ])
    # 協力関係特徴量
    feature_names.extend([
        "match_collaborated_with_task_author",
        "match_collaborator_overlap_count",
        "match_has_prior_collaboration"
    ])
    feature_names.extend([
        "match_skill_intersection_count",
        "match_file_experience_count"
    ])
    feature_names.extend([f"match_affinity_for_{label}" for label in all_labels])
    
    # GNN特徴量（協力ネットワーク対応）
    feature_names.extend([
        "gnn_similarity",
        "gnn_dev_expertise", 
        "gnn_task_popularity",
        "gnn_collaboration_strength",
        "gnn_network_centrality"
    ])
    
    print(f"\\n特徴量数: {len(feature_names)}, 重み数: {len(latest_weights)}")
    
    if len(feature_names) != len(latest_weights):
        print(f"❌ 特徴量数と重み数が一致しません")
        # 利用可能な分だけ分析
        min_len = min(len(feature_names), len(latest_weights))
        feature_names = feature_names[:min_len]
        latest_weights = latest_weights[:min_len]
        print(f"   最初の{min_len}個の特徴量で分析を続行")
    
    # 協力関係特徴量のインデックスを特定
    collaboration_indices = []
    collaboration_names = []
    for i, name in enumerate(feature_names):
        if any(keyword in name for keyword in [
            "collaboration_network_size", "comment_interactions", "cross_issue_activity",
            "collaborated_with_task_author", "collaborator_overlap_count", "has_prior_collaboration"
        ]):
            collaboration_indices.append(i)
            collaboration_names.append(name)
    
    print(f"\\n1. 協力関係特徴量の重み分析:")
    print(f"   協力関係特徴量数: {len(collaboration_indices)}")
    
    for idx, name in zip(collaboration_indices, collaboration_names):
        weight = latest_weights[idx]
        print(f"   {name}: {weight:.6f}")
    
    # 重みの絶対値でランキング
    print(f"\\n2. 全特徴量の重要度ランキング（絶対値）:")
    weight_abs = np.abs(latest_weights)
    sorted_indices = np.argsort(weight_abs)[::-1]
    
    print("   TOP 10:")
    for rank, idx in enumerate(sorted_indices[:10], 1):
        weight = latest_weights[idx]
        is_collab = idx in collaboration_indices
        collab_mark = " 🤝" if is_collab else ""
        print(f"   {rank:2d}. {feature_names[idx]:<40} {weight:8.6f}{collab_mark}")
    
    # 協力関係特徴量の順位
    print(f"\\n3. 協力関係特徴量の順位:")
    for idx, name in zip(collaboration_indices, collaboration_names):
        rank = np.where(sorted_indices == idx)[0][0] + 1
        weight = latest_weights[idx]
        print(f"   {rank:2d}位: {name:<40} {weight:8.6f}")
    
    # カテゴリ別重み分析
    analyze_weights_by_category(latest_weights, feature_names)
    
    # 可視化
    create_weight_visualization(latest_weights, feature_names, collaboration_indices)

def analyze_weights_by_category(weights, feature_names):
    """カテゴリ別の重み分析"""
    
    print(f"\\n4. カテゴリ別重み分析:")
    
    categories = {
        "タスク特徴": [],
        "開発者特徴": [],
        "社会的つながり": [],
        "協力関係": [],
        "マッチング": []
    }
    
    for i, name in enumerate(feature_names):
        if name.startswith("task_"):
            categories["タスク特徴"].append(weights[i])
        elif name.startswith("dev_") and not any(keyword in name for keyword in [
            "collaboration_network_size", "comment_interactions", "cross_issue_activity"
        ]):
            categories["開発者特徴"].append(weights[i])
        elif any(keyword in name for keyword in [
            "collaboration_network_size", "comment_interactions", "cross_issue_activity"
        ]):
            categories["社会的つながり"].append(weights[i])
        elif any(keyword in name for keyword in [
            "collaborated_with_task_author", "collaborator_overlap_count", "has_prior_collaboration"
        ]):
            categories["協力関係"].append(weights[i])
        elif name.startswith("match_"):
            categories["マッチング"].append(weights[i])
    
    for category, cat_weights in categories.items():
        if cat_weights:
            mean_weight = np.mean(np.abs(cat_weights))
            max_weight = np.max(np.abs(cat_weights))
            print(f"   {category:<12}: 平均重み={mean_weight:.6f}, 最大重み={max_weight:.6f}, 個数={len(cat_weights)}")

def create_weight_visualization(weights, feature_names, collaboration_indices):
    """重みの可視化"""
    
    print(f"\\n5. 重みの可視化を作成中...")
    
    # 協力関係特徴量をハイライト
    colors = ['red' if i in collaboration_indices else 'blue' for i in range(len(weights))]
    
    plt.figure(figsize=(15, 8))
    
    # 重みの棒グラフ
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(weights)), weights, color=colors, alpha=0.7)
    plt.title('IRL学習済み重み（協力関係特徴量を赤でハイライト）')
    plt.xlabel('特徴量インデックス')
    plt.ylabel('重み')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 凡例
    plt.bar([], [], color='red', alpha=0.7, label='協力関係特徴量')
    plt.bar([], [], color='blue', alpha=0.7, label='その他の特徴量')
    plt.legend()
    
    # 協力関係特徴量のみの拡大表示
    plt.subplot(2, 1, 2)
    collab_weights = [weights[i] for i in collaboration_indices]
    collab_names = [feature_names[i] for i in collaboration_indices]
    
    bars = plt.bar(range(len(collab_weights)), collab_weights, color='red', alpha=0.7)
    plt.title('協力関係特徴量の重み詳細')
    plt.xlabel('協力関係特徴量')
    plt.ylabel('重み')
    plt.xticks(range(len(collab_names)), [name.replace('match_', '').replace('dev_', '') for name in collab_names], 
               rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 重みの値をバーの上に表示
    for i, (bar, weight) in enumerate(zip(bars, collab_weights)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 * np.sign(weight), 
                f'{weight:.4f}', ha='center', va='bottom' if weight >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('irl_collaboration_weights_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   💾 グラフを保存: irl_collaboration_weights_analysis.png")
    
    # 統計サマリー
    print(f"\\n6. 協力関係特徴量の統計:")
    collab_weights = [weights[i] for i in collaboration_indices]
    print(f"   平均重み: {np.mean(collab_weights):.6f}")
    print(f"   重み標準偏差: {np.std(collab_weights):.6f}")
    print(f"   正の重みの数: {sum(1 for w in collab_weights if w > 0)}")
    print(f"   負の重みの数: {sum(1 for w in collab_weights if w < 0)}")
    print(f"   最大重み: {np.max(collab_weights):.6f}")
    print(f"   最小重み: {np.min(collab_weights):.6f}")

def analyze_collaboration_impact():
    """協力関係特徴量の実際の影響を分析"""
    
    print(f"\\n=== 協力関係特徴量の実際の影響分析 ===\\n")
    
    # 開発者プロファイルを読み込み
    with open('configs/dev_profiles.yaml', 'r') as f:
        profiles = yaml.safe_load(f)
    
    # バックログを読み込み
    with open('data/backlog.json', 'r') as f:
        backlog = json.load(f)
    
    print(f"1. データ統計:")
    print(f"   開発者数: {len(profiles)}")
    print(f"   タスク数: {len(backlog)}")
    
    # 協力関係が推薦に与える影響をシミュレート
    collaboration_impact_examples = []
    
    # いくつかの開発者とタスクの組み合わせで協力関係特徴量を計算
    sample_devs = list(profiles.keys())[:5]  # 最初の5人の開発者
    sample_tasks = backlog[:3]  # 最初の3つのタスク
    
    print(f"\\n2. 協力関係特徴量の実例計算:")
    
    for dev_name in sample_devs:
        dev_profile = profiles[dev_name]
        collaborators = set(dev_profile.get('collaborators', []))
        
        print(f"\\n  開発者: {dev_name}")
        print(f"    協力者数: {len(collaborators)}")
        print(f"    協力者: {', '.join(list(collaborators)[:3])}{'...' if len(collaborators) > 3 else ''}")
        
        for task_idx, task in enumerate(sample_tasks):
            task_author = task.get('user', {}).get('login') if 'user' in task else None
            task_assignees = task.get('assignees', []) if 'assignees' in task else []
            assignee_logins = {assignee.get('login') for assignee in task_assignees if assignee.get('login')}
            
            # 協力関係特徴量を計算
            collaborated_with_author = 1.0 if task_author and task_author in collaborators else 0.0
            collaborator_overlap_count = len(assignee_logins.intersection(collaborators))
            
            task_related_devs = assignee_logins.copy()
            if task_author:
                task_related_devs.add(task_author)
            has_prior_collaboration = 1.0 if len(task_related_devs.intersection(collaborators)) > 0 else 0.0
            
            print(f"    タスク{task_idx+1}:")
            print(f"      作成者との協力履歴: {collaborated_with_author}")
            print(f"      担当者との重複数: {collaborator_overlap_count}")
            print(f"      関連者との協力有無: {has_prior_collaboration}")

if __name__ == "__main__":
    analyze_irl_weights()
    analyze_collaboration_impact()
