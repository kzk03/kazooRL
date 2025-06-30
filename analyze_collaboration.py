#!/usr/bin/env python3
"""
協力関係の判定メカニズムを詳しく分析するスクリプト
"""

import json
from collections import defaultdict

import yaml


def analyze_collaboration_detection():
    """協力関係の検出メカニズムを分析"""
    
    print("=== 協力関係検出メカニズムの分析 ===\n")
    
    # 開発者プロファイルを読み込み
    with open('configs/dev_profiles.yaml', 'r') as f:
        profiles = yaml.safe_load(f)
    
    print("1. 協力関係の統計:")
    total_devs = len(profiles)
    devs_with_collaborators = 0
    total_collaborations = 0
    
    collab_distribution = defaultdict(int)
    
    for dev, profile in profiles.items():
        collaborators = profile.get('collaborators', [])
        collab_count = len(collaborators)
        
        if collab_count > 0:
            devs_with_collaborators += 1
            total_collaborations += collab_count
            
        collab_distribution[collab_count] += 1
    
    print(f"  - 総開発者数: {total_devs}")
    print(f"  - 協力者がいる開発者数: {devs_with_collaborators}")
    print(f"  - 協力者がいる割合: {devs_with_collaborators/total_devs*100:.1f}%")
    print(f"  - 平均協力者数: {total_collaborations/total_devs:.2f}")
    
    print(f"\n2. 協力者数の分布:")
    for collab_count in sorted(collab_distribution.keys()):
        if collab_count <= 10:  # 0-10人まで表示
            print(f"  - {collab_count}人: {collab_distribution[collab_count]}名")
        elif collab_count > 10:
            if collab_count == 11:  # 初回のみ表示
                print(f"  - 11人以上: {sum(collab_distribution[c] for c in collab_distribution if c > 10)}名")
            break
    
    print(f"\n3. 協力関係の具体例:")
    examples_shown = 0
    for dev, profile in profiles.items():
        collaborators = profile.get('collaborators', [])
        if len(collaborators) >= 2 and examples_shown < 3:  # 2人以上の協力者がいる例を3つ表示
            print(f"\n  開発者: {dev}")
            print(f"    協力者数: {len(collaborators)}")
            print(f"    マージされたPR数: {profile.get('total_merged_prs', 0)}")
            print(f"    コメント相互作用数: {profile.get('comment_interactions', 0)}")
            print(f"    協力者: {', '.join(collaborators[:5])}{'...' if len(collaborators) > 5 else ''}")
            examples_shown += 1
    
    print(f"\n4. 判定基準の詳細:")
    print("  協力関係は以下の場合に記録されます:")
    print("  A. Pull Request レベル:")
    print("    - PR作成者 ↔ PR担当者 (assignees)")
    print("    - PR作成者 ↔ PRレビュー依頼者 (requested_reviewers)")
    print("  B. Issue/コメント レベル:")
    print("    - Issue作成者 ↔ そのIssueにコメントした人")
    print("    - PR作成者 ↔ そのPRにコメントした人")
    
    print(f"\n5. 特徴量での活用:")
    print("  現在の特徴量は以下のように協力関係を使用します:")
    print("  - match_collaborated_with_task_author: タスク作成者との過去の協力履歴 (0 or 1)")
    print("  - match_collaborator_overlap_count: タスク担当者との協力履歴重複数")
    print("  - match_has_prior_collaboration: タスク関連開発者との協力履歴の有無 (0 or 1)")

def demonstrate_feature_calculation():
    """特徴量計算の実例を示す"""
    
    print(f"\n=== 特徴量計算の実例 ===\n")
    
    # サンプル開発者とタスクの例
    sample_dev_profile = {
        'collaborators': ['alice', 'bob', 'charlie', 'dave']
    }
    
    # ケース1: タスク作成者との協力履歴あり
    task1_author = 'alice'
    collaborated_with_author = 1.0 if task1_author in sample_dev_profile['collaborators'] else 0.0
    print(f"ケース1: タスク作成者が'{task1_author}'の場合")
    print(f"  match_collaborated_with_task_author = {collaborated_with_author}")
    
    # ケース2: タスク作成者との協力履歴なし
    task2_author = 'eve'
    collaborated_with_author = 1.0 if task2_author in sample_dev_profile['collaborators'] else 0.0
    print(f"\\nケース2: タスク作成者が'{task2_author}'の場合")
    print(f"  match_collaborated_with_task_author = {collaborated_with_author}")
    
    # ケース3: タスク担当者との重複
    task_assignees = ['bob', 'frank', 'charlie']
    assignee_set = set(task_assignees)
    collaborator_set = set(sample_dev_profile['collaborators'])
    overlap_count = len(assignee_set.intersection(collaborator_set))
    print(f"\\nケース3: タスク担当者が{task_assignees}の場合")
    print(f"  協力履歴のある担当者: {list(assignee_set.intersection(collaborator_set))}")
    print(f"  match_collaborator_overlap_count = {overlap_count}")
    
    # ケース4: 総合的な協力履歴
    task_related_devs = set([task1_author] + task_assignees)
    has_prior_collaboration = 1.0 if len(task_related_devs.intersection(collaborator_set)) > 0 else 0.0
    print(f"\\nケース4: タスク関連開発者との総合判定")
    print(f"  タスク関連開発者: {list(task_related_devs)}")
    print(f"  協力履歴のある人: {list(task_related_devs.intersection(collaborator_set))}")
    print(f"  match_has_prior_collaboration = {has_prior_collaboration}")

if __name__ == "__main__":
    analyze_collaboration_detection()
    demonstrate_feature_calculation()
