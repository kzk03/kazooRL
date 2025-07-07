#!/usr/bin/env python3
"""
Kazoo プロジェクト - 全特徴量詳細解説書
============================================

このドキュメントでは、IRLで使用される全62次元の特徴量について、
計算方法、意味、データソースを詳しく説明します。
"""


def print_feature_detailed_explanation():
    """全特徴量の詳細解説を出力"""

    print("=" * 80)
    print("📊 Kazoo プロジェクト - 全特徴量詳細解説")
    print("=" * 80)

    print("\n🎯 【概要】")
    print("総特徴量数: 62次元")
    print("- 基本特徴量: 25次元 (タスク9次元 + 開発者6次元 + マッチング10次元)")
    print("- GAT特徴量: 37次元 (統計5次元 + 埋め込み32次元)")

    print("\n" + "=" * 80)
    print("📝 【カテゴリ1: タスク特徴量】(9次元)")
    print("=" * 80)

    task_features = [
        {
            "name": "task_days_since_last_activity",
            "description": "タスクの最終活動からの日数",
            "calculation": "(データ内の最新日時 - task.updated_at) / (3600 * 24)",
            "meaning": "タスクが放置されている期間。値が大きいほど古いタスク",
            "data_source": "task.updated_at (GitHub Issue/PR の最終更新時刻)",
            "irl_weight": "-0.006133 (ほぼ無視)",
            "interpretation": "古さはほとんど重要でない",
        },
        {
            "name": "task_discussion_activity",
            "description": "タスクのディスカッション活動度",
            "calculation": "float(task.comments or 0)",
            "meaning": "タスクに対するコメント数。多いほど議論が活発",
            "data_source": "task.comments (GitHub Issue/PR のコメント数)",
            "irl_weight": "-0.027890 (軽微な負)",
            "interpretation": "議論が多すぎるタスクは避ける傾向",
        },
        {
            "name": "task_text_length",
            "description": "タスクテキストの長さ",
            "calculation": "float(len(task.body or ''))",
            "meaning": "タスクの説明文の文字数。長いほど詳細",
            "data_source": "task.body (GitHub Issue/PR の本文)",
            "irl_weight": "0.000034 (ほぼ無視)",
            "interpretation": "テキスト長はほとんど重要でない",
        },
        {
            "name": "task_code_block_count",
            "description": "タスク内のコードブロック数",
            "calculation": "float(task_body.count('```') // 2)",
            "meaning": "タスクに含まれるコードブロックの数",
            "data_source": "task.body 内の ``` マーカーの個数",
            "irl_weight": "0.213454 (中程度の正)",
            "interpretation": "コード例があるタスクを好む",
        },
        {
            "name": "task_label_bug",
            "description": "バグラベルの有無",
            "calculation": "1.0 if 'bug' in task.labels else 0.0",
            "meaning": "バグ修正タスクかどうか",
            "data_source": "task.labels (GitHub ラベル)",
            "irl_weight": "-0.759014 (強い負)",
            "interpretation": "バグ修正タスクは避ける傾向が強い",
        },
        {
            "name": "task_label_enhancement",
            "description": "機能強化ラベルの有無",
            "calculation": "1.0 if 'enhancement' in task.labels else 0.0",
            "meaning": "新機能追加タスクかどうか",
            "data_source": "task.labels",
            "irl_weight": "-0.417597 (中程度の負)",
            "interpretation": "機能強化タスクもやや避ける傾向",
        },
        {
            "name": "task_label_documentation",
            "description": "ドキュメントラベルの有無",
            "calculation": "1.0 if 'documentation' in task.labels else 0.0",
            "meaning": "ドキュメント作成タスクかどうか",
            "data_source": "task.labels",
            "irl_weight": "-0.585529 (中程度の負)",
            "interpretation": "ドキュメント作業は避ける傾向",
        },
        {
            "name": "task_label_question",
            "description": "質問ラベルの有無",
            "calculation": "1.0 if 'question' in task.labels else 0.0",
            "meaning": "質問・相談タスクかどうか",
            "data_source": "task.labels",
            "irl_weight": "1.732911 (非常に強い正)",
            "interpretation": "質問対応タスクを強く好む",
        },
        {
            "name": "task_label_help wanted",
            "description": "ヘルプ募集ラベルの有無",
            "calculation": "1.0 if 'help wanted' in task.labels else 0.0",
            "meaning": "協力者募集タスクかどうか",
            "data_source": "task.labels",
            "irl_weight": "0.595309 (中程度の正)",
            "interpretation": "協力募集タスクを好む",
        },
    ]

    for i, feature in enumerate(task_features):
        print(f"\n{i+1:2d}. {feature['name']}")
        print(f"    📋 説明: {feature['description']}")
        print(f"    🧮 計算: {feature['calculation']}")
        print(f"    💡 意味: {feature['meaning']}")
        print(f"    📊 データ: {feature['data_source']}")
        print(f"    ⚖️  IRL重み: {feature['irl_weight']}")
        print(f"    🤔 解釈: {feature['interpretation']}")

    print("\n" + "=" * 80)
    print("👨‍💻 【カテゴリ2: 開発者特徴量】(6次元)")
    print("=" * 80)

    dev_features = [
        {
            "name": "dev_recent_activity_count",
            "description": "開発者の最近の活動数",
            "calculation": "float(len(env.dev_action_history.get(developer_name, [])))",
            "meaning": "開発者の最近のアクション履歴数",
            "data_source": "環境内のアクション履歴",
            "irl_weight": "0.882315 (強い正)",
            "interpretation": "活発な開発者を強く好む",
        },
        {
            "name": "dev_current_workload",
            "description": "開発者の現在の作業負荷",
            "calculation": "float(len(env.assignments.get(developer_name, set())))",
            "meaning": "現在担当しているタスク数",
            "data_source": "環境内のタスク割り当て状況",
            "irl_weight": "-0.598743 (中程度の負)",
            "interpretation": "作業負荷が高い開発者は避ける",
        },
        {
            "name": "dev_total_lines_changed",
            "description": "開発者の総変更行数",
            "calculation": "float(developer_profile.get('total_lines_changed', 0))",
            "meaning": "過去にマージされたPRでの総変更行数",
            "data_source": "GitHub PR履歴から事前計算",
            "irl_weight": "0.084734 (軽微な正)",
            "interpretation": "コード変更経験はやや重視",
        },
        {
            "name": "dev_collaboration_network_size",
            "description": "開発者の協力ネットワークサイズ",
            "calculation": "float(developer_profile.get('collaboration_network_size', 0))",
            "meaning": "一緒に作業したことのある開発者数",
            "data_source": "GitHub PR の co-author から事前計算",
            "irl_weight": "0.765557 (強い正)",
            "interpretation": "協力ネットワークが広い開発者を好む",
        },
        {
            "name": "dev_comment_interactions",
            "description": "開発者のコメント相互作用数",
            "calculation": "float(developer_profile.get('comment_interactions', 0))",
            "meaning": "他の開発者のIssue/PRにコメントした回数",
            "data_source": "GitHub コメント履歴から事前計算",
            "irl_weight": "1.401790 (非常に強い正)",
            "interpretation": "コミュニケーション活発な開発者を強く好む",
        },
        {
            "name": "dev_cross_issue_activity",
            "description": "開発者のクロスイシュー活動",
            "calculation": "float(developer_profile.get('cross_issue_activity', 0))",
            "meaning": "複数のIssueにまたがる活動度",
            "data_source": "Issue参加履歴から事前計算",
            "irl_weight": "0.659460 (中程度の正)",
            "interpretation": "幅広いIssueに関与する開発者を好む",
        },
    ]

    for i, feature in enumerate(dev_features):
        print(f"\n{i+1:2d}. {feature['name']}")
        print(f"    📋 説明: {feature['description']}")
        print(f"    🧮 計算: {feature['calculation']}")
        print(f"    💡 意味: {feature['meaning']}")
        print(f"    📊 データ: {feature['data_source']}")
        print(f"    ⚖️  IRL重み: {feature['irl_weight']}")
        print(f"    🤔 解釈: {feature['interpretation']}")

    print("\n" + "=" * 80)
    print("🤝 【カテゴリ3: マッチング特徴量】(10次元)")
    print("=" * 80)

    match_features = [
        {
            "name": "match_collaborated_with_task_author",
            "description": "タスク作成者との協力履歴",
            "calculation": "1.0 if task_author and task_author in dev_collaborators else 0.0",
            "meaning": "開発者がタスクの作成者と過去に協力したことがあるか",
            "data_source": "task.user.login と developer.profile.collaborators",
            "irl_weight": "0.111267 (軽微な正)",
            "interpretation": "作成者との過去の協力は軽微に重視",
        },
        {
            "name": "match_collaborator_overlap_count",
            "description": "共通協力者数",
            "calculation": "float(len(assignee_logins.intersection(dev_collaborators)))",
            "meaning": "タスクの担当者と開発者の共通協力者数",
            "data_source": "task.assignees と developer.profile.collaborators",
            "irl_weight": "0.260951 (軽微な正)",
            "interpretation": "共通の協力者がいることは軽微に重視",
        },
        {
            "name": "match_has_prior_collaboration",
            "description": "過去の協力関係の有無",
            "calculation": "1.0 if len(task_related_devs.intersection(dev_collaborators)) > 0 else 0.0",
            "meaning": "タスク関連開発者との過去の協力関係があるか",
            "data_source": "タスク関連開発者 と developer.profile.collaborators",
            "irl_weight": "-0.643749 (中程度の負)",
            "interpretation": "過去の協力関係がある場合は避ける傾向",
        },
        {
            "name": "match_skill_intersection_count",
            "description": "スキル交差数",
            "calculation": "float(len(required_skills.intersection(developer_skills)))",
            "meaning": "タスクに必要なスキルと開発者のスキルの一致数",
            "data_source": "task.labels → skills と developer.profile.skills",
            "irl_weight": "-1.295156 (非常に強い負)",
            "interpretation": "スキル一致は逆に避ける傾向（意外な結果）",
        },
        {
            "name": "match_file_experience_count",
            "description": "ファイル経験数",
            "calculation": "float(len(pr_changed_files.intersection(dev_touched_files)))",
            "meaning": "タスクで変更されるファイルの開発者の編集経験数",
            "data_source": "task.changed_files と developer.profile.touched_files",
            "irl_weight": "1.417670 (非常に強い正)",
            "interpretation": "ファイル経験は非常に重要",
        },
        {
            "name": "match_affinity_for_bug",
            "description": "バグ対応への親和性",
            "calculation": "dev_affinity_profile.get('bug', 0.0) if task has bug label",
            "meaning": "開発者のバグ対応タスクへの親和性スコア",
            "data_source": "developer.profile.label_affinity.bug",
            "irl_weight": "-0.265101 (軽微な負)",
            "interpretation": "バグ親和性は軽微に避ける",
        },
        {
            "name": "match_affinity_for_enhancement",
            "description": "機能強化への親和性",
            "calculation": "dev_affinity_profile.get('enhancement', 0.0) if task has enhancement label",
            "meaning": "開発者の機能強化タスクへの親和性スコア",
            "data_source": "developer.profile.label_affinity.enhancement",
            "irl_weight": "-1.153890 (非常に強い負)",
            "interpretation": "機能強化親和性は強く避ける",
        },
        {
            "name": "match_affinity_for_documentation",
            "description": "ドキュメント作業への親和性",
            "calculation": "dev_affinity_profile.get('documentation', 0.0) if task has doc label",
            "meaning": "開発者のドキュメント作業への親和性スコア",
            "data_source": "developer.profile.label_affinity.documentation",
            "irl_weight": "0.989552 (強い正)",
            "interpretation": "ドキュメント親和性は重視",
        },
        {
            "name": "match_affinity_for_question",
            "description": "質問対応への親和性",
            "calculation": "dev_affinity_profile.get('question', 0.0) if task has question label",
            "meaning": "開発者の質問対応への親和性スコア",
            "data_source": "developer.profile.label_affinity.question",
            "irl_weight": "0.606813 (中程度の正)",
            "interpretation": "質問対応親和性を好む",
        },
        {
            "name": "match_affinity_for_help wanted",
            "description": "ヘルプ対応への親和性",
            "calculation": "dev_affinity_profile.get('help wanted', 0.0) if task has help label",
            "meaning": "開発者のヘルプ対応への親和性スコア",
            "data_source": "developer.profile.label_affinity.help_wanted",
            "irl_weight": "0.501730 (中程度の正)",
            "interpretation": "ヘルプ対応親和性を好む",
        },
    ]

    for i, feature in enumerate(match_features):
        print(f"\n{i+1:2d}. {feature['name']}")
        print(f"    📋 説明: {feature['description']}")
        print(f"    🧮 計算: {feature['calculation']}")
        print(f"    💡 意味: {feature['meaning']}")
        print(f"    📊 データ: {feature['data_source']}")
        print(f"    ⚖️  IRL重み: {feature['irl_weight']}")
        print(f"    🤔 解釈: {feature['interpretation']}")

    print("\n" + "=" * 80)
    print("🧠 【カテゴリ4: GAT統計特徴量】(5次元)")
    print("=" * 80)

    gat_stats_features = [
        {
            "name": "gat_similarity",
            "description": "GAT類似度",
            "calculation": "F.cosine_similarity(dev_embedding, task_embedding)",
            "meaning": "GATモデルによる開発者-タスク間の類似度",
            "data_source": "GAT埋め込みベクトルのコサイン類似度",
            "irl_weight": "-1.134863 (非常に強い負)",
            "interpretation": "GAT類似度が高い場合は避ける（意外な結果）",
        },
        {
            "name": "gat_dev_expertise",
            "description": "GAT開発者専門性",
            "calculation": "torch.mean(torch.topk(dev_vs_all_tasks_similarity, k=10).values)",
            "meaning": "開発者の全タスクに対する平均類似度（専門性指標）",
            "data_source": "GAT埋め込みによる類似度計算",
            "irl_weight": "0.541512 (中程度の正)",
            "interpretation": "GAT専門性スコアを好む",
        },
        {
            "name": "gat_task_popularity",
            "description": "GATタスク人気度",
            "calculation": "torch.mean(torch.topk(task_vs_all_devs_similarity, k=10).values)",
            "meaning": "タスクの全開発者に対する平均類似度（人気指標）",
            "data_source": "GAT埋め込みによる類似度計算",
            "irl_weight": "-0.137711 (軽微な負)",
            "interpretation": "GAT人気度はやや避ける",
        },
        {
            "name": "gat_collaboration_strength",
            "description": "GAT協力関係強度",
            "calculation": "sum(edge_weights for edges involving developer) / max_strength",
            "meaning": "開発者の協力ネットワーク内での強度（正規化済み）",
            "data_source": "開発者協力ネットワークのエッジ重み",
            "irl_weight": "1.838603 (非常に強い正)",
            "interpretation": "協力関係強度を非常に重視",
        },
        {
            "name": "gat_network_centrality",
            "description": "GATネットワーク中心性",
            "calculation": "degree_count / max_possible_degree",
            "meaning": "開発者のネットワーク内での中心性（正規化済み）",
            "data_source": "協力ネットワークの次数中心性",
            "irl_weight": "1.235597 (非常に強い正)",
            "interpretation": "ネットワーク中心性を非常に重視",
        },
    ]

    for i, feature in enumerate(gat_stats_features):
        print(f"\n{i+1:2d}. {feature['name']}")
        print(f"    📋 説明: {feature['description']}")
        print(f"    🧮 計算: {feature['calculation']}")
        print(f"    💡 意味: {feature['meaning']}")
        print(f"    📊 データ: {feature['data_source']}")
        print(f"    ⚖️  IRL重み: {feature['irl_weight']}")
        print(f"    🤔 解釈: {feature['interpretation']}")

    print("\n" + "=" * 80)
    print("🤖 【カテゴリ5: GAT埋め込み】(32次元)")
    print("=" * 80)

    print("\n概要:")
    print("- gat_dev_emb_0 ～ gat_dev_emb_31 の32次元")
    print("- GATニューラルネットワークが学習した開発者の抽象的表現")
    print("- 各次元の直接的な意味は解釈困難（分散表現）")
    print("- 開発者のスキル、協力パターン、プロジェクト適性などが複合的に表現")

    # 重要な埋め込み次元のみ表示
    important_embeddings = [
        ("gat_dev_emb_26", 2.175824, "最重要次元"),
        ("gat_dev_emb_22", 1.892911, "第2重要次元"),
        ("gat_dev_emb_11", 1.606356, "第5重要次元"),
        ("gat_dev_emb_17", 1.256952, "第9重要次元"),
        ("gat_dev_emb_1", 1.244942, "第10重要次元"),
        ("gat_dev_emb_19", -1.186278, "強い負の重み"),
        ("gat_dev_emb_27", -1.016314, "負の重み"),
        ("gat_dev_emb_16", -0.991054, "負の重み"),
    ]

    print("\n重要な埋め込み次元:")
    for name, weight, description in important_embeddings:
        print(f"  {name:<15}: {weight:>9.6f} ({description})")

    print("\n💡 GAT埋め込みの特徴:")
    print("- 平均重み: 0.445030")
    print("- 重み範囲: [-1.186278, 2.175824]")
    print("- 絶対値平均: 0.788985")
    print("- 正の重み: 25個 (78.1%)")
    print("- 負の重み: 7個 (21.9%)")

    print("\n🤔 解釈:")
    print("- 第26次元が最も重要（重み: 2.176）")
    print("- 第22次元、第11次元も非常に重要")
    print("- 大部分の次元が正の重みで、GAT特徴量全体が重視されている")
    print("- 個別次元の意味は不明だが、全体として開発者の適性を表現")

    print("\n" + "=" * 80)
    print("📈 【データ生成プロセス】")
    print("=" * 80)

    print("\n1. 🏗️  事前処理 (data_processing/)")
    print("   generate_profiles.py:")
    print("   - GitHub Archive データから開発者プロファイルを生成")
    print("   - label_affinity, touched_files, collaboration_network などを計算")
    print("   - 結果: configs/dev_profiles.yaml")

    print("\n   generate_graph.py:")
    print("   - 開発者-タスク関係グラフを生成")
    print("   - ノード特徴量（開発者8次元、タスク9次元）を設定")
    print("   - 結果: data/graph.pt")

    print("\n2. 🧠 GAT学習 (training/gat/)")
    print("   train_collaborative_gat.py:")
    print("   - グラフニューラルネットワークを訓練")
    print("   - 開発者間協力関係を学習")
    print("   - 結果: data/gnn_model_collaborative.pt")

    print("\n3. 🔄 IRL学習 (training/irl/)")
    print("   train_irl.py:")
    print("   - 専門家の行動データから報酬関数を学習")
    print("   - 全62次元の特徴量重みを最適化")
    print("   - 結果: data/learned_weights_training.npy")

    print("\n4. 💪 強化学習 (training/rl/)")
    print("   train_rl.py:")
    print("   - 学習した報酬関数でPPOエージェントを訓練")
    print("   - 開発者選択ポリシーを学習")

    print("\n" + "=" * 80)
    print("🎯 【IRL学習結果の解釈】")
    print("=" * 80)

    print("\n✅ 重視される特徴 (正の重み):")
    print("1. 協力ネットワーク関連:")
    print("   - gat_collaboration_strength (1.839)")
    print("   - gat_network_centrality (1.236)")
    print("   - dev_collaboration_network_size (0.766)")

    print("\n2. コミュニケーション能力:")
    print("   - dev_comment_interactions (1.402)")
    print("   - task_label_question (1.733)")

    print("\n3. 経験とファイル知識:")
    print("   - match_file_experience_count (1.418)")
    print("   - dev_recent_activity_count (0.882)")

    print("\n4. GAT埋め込み次元:")
    print("   - gat_dev_emb_26 (2.176) - 最重要")
    print("   - gat_dev_emb_22 (1.893)")
    print("   - gat_dev_emb_11 (1.606)")

    print("\n❌ 避けられる特徴 (負の重み):")
    print("1. スキル一致 (意外な結果):")
    print("   - match_skill_intersection_count (-1.295)")
    print("   - match_affinity_for_enhancement (-1.154)")

    print("\n2. 特定のタスクタイプ:")
    print("   - task_label_bug (-0.759)")
    print("   - task_label_documentation (-0.585)")

    print("\n3. 作業負荷:")
    print("   - dev_current_workload (-0.599)")

    print("\n🤔 【専門家の判断パターン】")
    print("=" * 80)

    print("\n1. 「人」重視:")
    print("   - コミュニケーション能力 > 技術スキル")
    print("   - 協力ネットワーク > 個人能力")
    print("   - 活動性 > 経験年数")

    print("\n2. 「文脈」重視:")
    print("   - ファイル経験 > 一般的スキル")
    print("   - 質問対応 > バグ修正")
    print("   - 協力募集 > 個人作業")

    print("\n3. 「バランス」重視:")
    print("   - 適度な作業負荷")
    print("   - 幅広いIssue参加")
    print("   - ネットワーク中心性")

    print("\n💡 【活用方法】")
    print("=" * 80)

    print("\n1. 🎯 タスク推薦:")
    print("   特徴量ベクトル × IRL重み = 適性スコア")

    print("\n2. 👥 チーム編成:")
    print("   協力ネットワーク特徴量を活用")

    print("\n3. 📊 開発者評価:")
    print("   コミュニケーション・協力度の定量化")

    print("\n4. 🔮 成果予測:")
    print("   ファイル経験・タスク親和性の組み合わせ")

    print("\n" + "=" * 80)
    print("🔚 解説完了")
    print("=" * 80)


if __name__ == "__main__":
    print_feature_detailed_explanation()
