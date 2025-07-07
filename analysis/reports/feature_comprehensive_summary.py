#!/usr/bin/env python3
"""
特徴量要約レポート
=================

IRLで使用される全特徴量の詳細解説を簡潔にまとめたレポートです。
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def generate_comprehensive_feature_summary():
    """包括的な特徴量要約を生成"""
    
    print("="*80)
    print("🎯 Kazoo プロジェクト - 特徴量完全解説書")
    print("="*80)
    print(f"📅 生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    
    # IRL重みを読み込み
    try:
        project_root = Path(__file__).resolve().parents[2]
        weights_path = project_root / "data" / "learned_weights_training.npy"
        if weights_path.exists():
            weights = np.load(weights_path)
            print(f"✅ IRL重み読み込み: {len(weights)}次元")
        else:
            # デフォルト重み（実際のサマリーレポートから）
            weights = np.array([
                -0.006133, -0.027890, 0.000034, 0.213454, -0.759014, -0.417597, -0.585529, 1.732911, 0.595309,
                0.882315, -0.598743, 0.084734, 0.765557, 1.401790, 0.659460,
                0.111267, 0.260951, -0.643749, -1.295156, 1.417670,
                -0.265101, -1.153890, 0.989552, 0.606813, 0.501730,
                -1.134863, 0.541512, -0.137711, 1.838603, 1.235597
            ] + [np.random.randn() for _ in range(32)])  # GAT埋め込み32次元
            print(f"⚠️  ダミー重みを使用: {len(weights)}次元")
    except Exception as e:
        weights = np.random.randn(62)
        print(f"⚠️  重み読み込みエラー: {e}")
    
    print(f"\n📊 【概要】")
    print(f"・総特徴量数: 62次元")
    print(f"・基本特徴量: 25次元 (タスク9 + 開発者6 + マッチング10)")
    print(f"・GAT特徴量: 37次元 (統計5 + 埋め込み32)")
    print(f"・有効重み数: {np.sum(np.abs(weights) > 0.01)}")
    print(f"・正の重み: {np.sum(weights > 0)} / 負の重み: {np.sum(weights < 0)}")
    
    print("\n" + "="*80)
    print("📝 【特徴量詳細解説】")
    print("="*80)
    
    # 特徴量定義
    features_info = [
        # タスク特徴量 (9次元)
        {
            "category": "🎯 タスク特徴量",
            "features": [
                {
                    "name": "task_days_since_last_activity",
                    "japanese": "タスク放置日数",
                    "description": "最後の活動からの経過日数",
                    "calculation": "(最新日時 - task.updated_at) ÷ (24時間)",
                    "data_source": "GitHub Issue/PR 更新日時",
                    "meaning": "古いタスクほど大きい値。緊急度の指標",
                    "weight_idx": 0
                },
                {
                    "name": "task_discussion_activity", 
                    "japanese": "議論活発度",
                    "description": "タスクのコメント数",
                    "calculation": "task.comments の値",
                    "data_source": "GitHub コメント数",
                    "meaning": "議論が活発なタスクほど大きい値",
                    "weight_idx": 1
                },
                {
                    "name": "task_text_length",
                    "japanese": "説明文長",
                    "description": "タスク本文の文字数",
                    "calculation": "len(task.body) の文字数",
                    "data_source": "GitHub Issue/PR 本文",
                    "meaning": "詳細に説明されたタスクほど大きい値",
                    "weight_idx": 2
                },
                {
                    "name": "task_code_block_count",
                    "japanese": "コード例数",
                    "description": "タスク内のコードブロック数",
                    "calculation": "task.body内の```マーカー数 ÷ 2",
                    "data_source": "GitHub マークダウン",
                    "meaning": "コード例が多いタスクほど大きい値",
                    "weight_idx": 3
                },
                {
                    "name": "task_label_bug",
                    "japanese": "バグラベル",
                    "description": "バグ修正タスクか",
                    "calculation": "'bug' in task.labels ? 1 : 0",
                    "data_source": "GitHub ラベル",
                    "meaning": "バグ修正タスクなら1、そうでなければ0",
                    "weight_idx": 4
                },
                {
                    "name": "task_label_enhancement",
                    "japanese": "機能強化ラベル", 
                    "description": "新機能追加タスクか",
                    "calculation": "'enhancement' in task.labels ? 1 : 0",
                    "data_source": "GitHub ラベル",
                    "meaning": "機能強化タスクなら1、そうでなければ0",
                    "weight_idx": 5
                },
                {
                    "name": "task_label_documentation",
                    "japanese": "ドキュメントラベル",
                    "description": "ドキュメント作成タスクか", 
                    "calculation": "'documentation' in task.labels ? 1 : 0",
                    "data_source": "GitHub ラベル",
                    "meaning": "ドキュメント作業なら1、そうでなければ0",
                    "weight_idx": 6
                },
                {
                    "name": "task_label_question",
                    "japanese": "質問ラベル",
                    "description": "質問・相談タスクか",
                    "calculation": "'question' in task.labels ? 1 : 0",
                    "data_source": "GitHub ラベル", 
                    "meaning": "質問対応なら1、そうでなければ0",
                    "weight_idx": 7
                },
                {
                    "name": "task_label_help wanted",
                    "japanese": "ヘルプ募集ラベル",
                    "description": "協力者募集タスクか",
                    "calculation": "'help wanted' in task.labels ? 1 : 0",
                    "data_source": "GitHub ラベル",
                    "meaning": "協力募集なら1、そうでなければ0",
                    "weight_idx": 8
                }
            ]
        },
        # 開発者特徴量 (6次元)
        {
            "category": "👨‍💻 開発者特徴量",
            "features": [
                {
                    "name": "dev_recent_activity_count",
                    "japanese": "最近の活動数",
                    "description": "開発者の最近のアクション数",
                    "calculation": "len(env.dev_action_history[developer])",
                    "data_source": "環境内アクション履歴",
                    "meaning": "活発な開発者ほど大きい値",
                    "weight_idx": 9
                },
                {
                    "name": "dev_current_workload",
                    "japanese": "現在の作業負荷",
                    "description": "現在担当中のタスク数",
                    "calculation": "len(env.assignments[developer])",
                    "data_source": "環境内タスク割り当て",
                    "meaning": "忙しい開発者ほど大きい値",
                    "weight_idx": 10
                },
                {
                    "name": "dev_total_lines_changed",
                    "japanese": "総変更行数",
                    "description": "過去の総コード変更行数",
                    "calculation": "Σ(merged PRでの変更行数)",
                    "data_source": "GitHub PR履歴",
                    "meaning": "コード変更経験が豊富ほど大きい値",
                    "weight_idx": 11
                },
                {
                    "name": "dev_collaboration_network_size", 
                    "japanese": "協力ネットワークサイズ",
                    "description": "協力したことがある開発者数",
                    "calculation": "len(developer.collaborators)",
                    "data_source": "GitHub co-author 履歴",
                    "meaning": "協力関係が広い開発者ほど大きい値",
                    "weight_idx": 12
                },
                {
                    "name": "dev_comment_interactions",
                    "japanese": "コメント相互作用数",
                    "description": "他開発者のIssue/PRへのコメント数",
                    "calculation": "Σ(他者のIssue/PRへのコメント数)",
                    "data_source": "GitHub コメント履歴",
                    "meaning": "コミュニケーション活発な開発者ほど大きい値",
                    "weight_idx": 13
                },
                {
                    "name": "dev_cross_issue_activity",
                    "japanese": "クロスイシュー活動度",
                    "description": "複数Issueにまたがる活動度",
                    "calculation": "複数Issue参加の複雑度指標",
                    "data_source": "Issue参加履歴",
                    "meaning": "幅広いIssueに関与する開発者ほど大きい値",
                    "weight_idx": 14
                }
            ]
        },
        # マッチング特徴量 (10次元)
        {
            "category": "🤝 マッチング特徴量",
            "features": [
                {
                    "name": "match_collaborated_with_task_author",
                    "japanese": "作成者協力履歴",
                    "description": "タスク作成者との過去の協力",
                    "calculation": "task.author in developer.collaborators ? 1 : 0",
                    "data_source": "協力者リスト × タスク作成者",
                    "meaning": "作成者と協力経験があれば1",
                    "weight_idx": 15
                },
                {
                    "name": "match_collaborator_overlap_count",
                    "japanese": "共通協力者数",
                    "description": "タスク担当者との共通協力者数",
                    "calculation": "len(task_assignees ∩ developer.collaborators)",
                    "data_source": "担当者リスト × 協力者リスト",
                    "meaning": "共通の協力者が多いほど大きい値",
                    "weight_idx": 16
                },
                {
                    "name": "match_has_prior_collaboration",
                    "japanese": "事前協力関係",
                    "description": "タスク関連者との協力履歴有無",
                    "calculation": "len(task_related_devs ∩ developer.collaborators) > 0 ? 1 : 0",
                    "data_source": "タスク関連者 × 協力者リスト",
                    "meaning": "関連者と協力経験があれば1",
                    "weight_idx": 17
                },
                {
                    "name": "match_skill_intersection_count",
                    "japanese": "スキル一致数",
                    "description": "必要スキルと保有スキルの一致数",
                    "calculation": "len(required_skills ∩ developer.skills)",
                    "data_source": "タスクラベル→スキル × 開発者スキル",
                    "meaning": "一致するスキルが多いほど大きい値",
                    "weight_idx": 18
                },
                {
                    "name": "match_file_experience_count",
                    "japanese": "ファイル経験数",
                    "description": "変更ファイルの編集経験数",
                    "calculation": "len(task.changed_files ∩ developer.touched_files)",
                    "data_source": "変更ファイル × 編集履歴",
                    "meaning": "経験のあるファイルが多いほど大きい値",
                    "weight_idx": 19
                },
                {
                    "name": "match_affinity_for_bug",
                    "japanese": "バグ対応親和性",
                    "description": "バグタスクへの開発者親和性",
                    "calculation": "task has bug label ? developer.label_affinity.bug : 0",
                    "data_source": "開発者親和性プロファイル",
                    "meaning": "バグタスクの場合の親和性スコア",
                    "weight_idx": 20
                },
                {
                    "name": "match_affinity_for_enhancement",
                    "japanese": "機能強化親和性",
                    "description": "機能強化タスクへの開発者親和性",
                    "calculation": "task has enhancement label ? developer.label_affinity.enhancement : 0", 
                    "data_source": "開発者親和性プロファイル",
                    "meaning": "機能強化タスクの場合の親和性スコア",
                    "weight_idx": 21
                },
                {
                    "name": "match_affinity_for_documentation",
                    "japanese": "ドキュメント親和性",
                    "description": "ドキュメントタスクへの開発者親和性",
                    "calculation": "task has doc label ? developer.label_affinity.documentation : 0",
                    "data_source": "開発者親和性プロファイル", 
                    "meaning": "ドキュメントタスクの場合の親和性スコア",
                    "weight_idx": 22
                },
                {
                    "name": "match_affinity_for_question",
                    "japanese": "質問対応親和性",
                    "description": "質問タスクへの開発者親和性",
                    "calculation": "task has question label ? developer.label_affinity.question : 0",
                    "data_source": "開発者親和性プロファイル",
                    "meaning": "質問タスクの場合の親和性スコア",
                    "weight_idx": 23
                },
                {
                    "name": "match_affinity_for_help wanted",
                    "japanese": "ヘルプ対応親和性",
                    "description": "ヘルプタスクへの開発者親和性",
                    "calculation": "task has help label ? developer.label_affinity.help_wanted : 0",
                    "data_source": "開発者親和性プロファイル",
                    "meaning": "ヘルプタスクの場合の親和性スコア",
                    "weight_idx": 24
                }
            ]
        },
        # GAT統計特徴量 (5次元)
        {
            "category": "🧠 GAT統計特徴量",
            "features": [
                {
                    "name": "gat_similarity",
                    "japanese": "GAT類似度",
                    "description": "GATによる開発者-タスク類似度",
                    "calculation": "cosine_similarity(dev_embedding, task_embedding)",
                    "data_source": "GAT埋め込みベクトル",
                    "meaning": "GAT空間での類似度。高いほど適合",
                    "weight_idx": 25
                },
                {
                    "name": "gat_dev_expertise",
                    "japanese": "GAT開発者専門性",
                    "description": "GATによる開発者専門性スコア",
                    "calculation": "mean(top_k_similarity(dev, all_tasks))",
                    "data_source": "GAT埋め込み類似度計算",
                    "meaning": "専門性が高い開発者ほど大きい値",
                    "weight_idx": 26
                },
                {
                    "name": "gat_task_popularity",
                    "japanese": "GATタスク人気度",
                    "description": "GATによるタスク人気度スコア",
                    "calculation": "mean(top_k_similarity(task, all_devs))",
                    "data_source": "GAT埋め込み類似度計算",
                    "meaning": "人気の高いタスクほど大きい値",
                    "weight_idx": 27
                },
                {
                    "name": "gat_collaboration_strength",
                    "japanese": "GAT協力関係強度",
                    "description": "協力ネットワークでの開発者の強度",
                    "calculation": "Σ(edge_weights) / max_possible_strength",
                    "data_source": "開発者協力ネットワーク",
                    "meaning": "協力関係が強い開発者ほど大きい値",
                    "weight_idx": 28
                },
                {
                    "name": "gat_network_centrality",
                    "japanese": "GATネットワーク中心性",
                    "description": "協力ネットワークでの中心性",
                    "calculation": "degree_count / max_possible_degree", 
                    "data_source": "協力ネットワーク次数",
                    "meaning": "ネットワーク中心にいる開発者ほど大きい値",
                    "weight_idx": 29
                }
            ]
        }
    ]
    
    # 各カテゴリの特徴量を表示
    for category_info in features_info:
        print(f"\n{category_info['category']} ({len(category_info['features'])}次元)")
        print("─" * 70)
        
        for i, feature in enumerate(category_info['features'], 1):
            weight = weights[feature['weight_idx']] if feature['weight_idx'] < len(weights) else 0.0
            importance = "非常に重要" if abs(weight) > 1.0 else "重要" if abs(weight) > 0.5 else "軽微" if abs(weight) > 0.1 else "無視"
            direction = "好む" if weight > 0 else "避ける" if weight < 0 else "中立"
            
            print(f"\n{i:2d}. {feature['name']}")
            print(f"    🏷️  名称: {feature['japanese']}")
            print(f"    📋 説明: {feature['description']}")
            print(f"    🧮 計算: {feature['calculation']}")
            print(f"    📊 データ: {feature['data_source']}")
            print(f"    💡 意味: {feature['meaning']}")
            print(f"    ⚖️  IRL重み: {weight:8.4f} ({importance}, {direction})")
    
    # GAT埋め込み次元の説明
    print(f"\n🤖 GAT埋め込み特徴量 (32次元)")
    print("─" * 70)
    print("📋 説明: GATニューラルネットワークが学習した開発者の抽象的表現")
    print("🧮 計算: GAT(graph_data, developer_node_features)")
    print("📊 データ: 開発者-タスク関係グラフ + ノード特徴量")
    print("💡 意味: 開発者のスキル、協力パターン、適性などが32次元ベクトルで表現")
    print("⚖️  IRL重み: 各次元ごとに学習された重み（-1.2 ～ +2.2の範囲）")
    
    if len(weights) >= 62:
        gat_weights = weights[30:62]  # GAT埋め込み32次元
        important_dims = [(i, w) for i, w in enumerate(gat_weights) if abs(w) > 0.8]
        important_dims.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n    🔥 重要次元 (|重み| > 0.8):")
        for dim, weight in important_dims[:10]:  # 上位10次元
            print(f"       gat_dev_emb_{dim:2d}: {weight:8.4f}")
    
    print(f"\n" + "="*80)
    print("📊 【IRL学習結果サマリー】")
    print("="*80)
    
    # 重要特徴量のランキング
    feature_names = []
    for category_info in features_info:
        feature_names.extend([f['name'] for f in category_info['features']])
    feature_names.extend([f'gat_dev_emb_{i}' for i in range(32)])
    
    if len(weights) == len(feature_names):
        feature_importance = [(name, weights[i], abs(weights[i])) for i, name in enumerate(feature_names)]
        feature_importance.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n🏆 【最重要特徴量 TOP 10】")
        for i, (name, weight, importance) in enumerate(feature_importance[:10], 1):
            direction = "✅ 好む" if weight > 0 else "❌ 避ける"
            print(f"{i:2d}. {name[:35]:35s} | {weight:8.4f} | {direction}")
        
        print(f"\n⬇️ 【最も避けられる特徴量 TOP 5】")
        negative_features = [(n, w, i) for n, w, i in feature_importance if w < 0]
        for i, (name, weight, importance) in enumerate(negative_features[:5], 1):
            print(f"{i:2d}. {name[:35]:35s} | {weight:8.4f}")
    
    print(f"\n" + "="*80)
    print("🎯 【実用的な解釈】")
    print("="*80)
    
    print(f"""
✅ 【IRLが学習した専門家の判断パターン】

1. 👥 人間関係重視:
   • コミュニケーション能力 > 技術スキル
   • 協力ネットワークの広さ・強さを重視
   • 質問対応や協力募集タスクを好む

2. 📁 経験・コンテキスト重視:
   • ファイル編集経験 > 一般的スキル一致
   • GAT埋め込みが捉える複雑な適性パターン
   • 作業負荷のバランスを考慮

3. ⚡ 避けるパターン:
   • バグ修正タスク（複雑で時間がかかる）
   • スキル過剰一致（オーバースペック回避）
   • 高負荷開発者への追加割り当て

4. 🎯 推薦戦略:
   • GAT特徴量の重み合計: {np.sum(weights[25:]) if len(weights) > 25 else "N/A"}
   • ネットワーク特徴量の重み合計: {np.sum([weights[12], weights[13], weights[28], weights[29]]) if len(weights) > 29 else "N/A"}
   • タスクタイプ特徴量の範囲: {np.ptp(weights[4:9]) if len(weights) > 9 else "N/A"}

💡 【活用方法】
1. 開発者推薦: 特徴量ベクトル × IRL重み = 適性スコア
2. チーム分析: 協力ネットワーク特徴量でチーム相性分析
3. タスク優先度: 質問・ヘルプ系タスクの優先度向上
4. 負荷分散: 現在作業負荷と協力関係強度のバランス調整
""")
    
    print(f"\n" + "="*80)
    print("🔚 解説完了")
    print("="*80)

def main():
    """メイン実行関数"""
    generate_comprehensive_feature_summary()

if __name__ == "__main__":
    main()
