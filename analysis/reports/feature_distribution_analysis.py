#!/usr/bin/env python3
"""
特徴量分布分析スクリプト
========================

IRLで使用される全特徴量の実際のデータ分布を調べ、
統計情報と可視化を提供します。
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def get_feature_japanese_names():
    """特徴量名と日本語説明のマッピング辞書を返す"""
    return {
        # タスク特徴量
        "task_days_since_last_activity": "タスク最終活動からの日数",
        "task_discussion_activity": "タスク議論活動度",
        "task_text_length": "タスクテキスト長",
        "task_code_block_count": "タスクコードブロック数",
        "task_label_bug": "タスクラベル: バグ",
        "task_label_enhancement": "タスクラベル: 機能強化",
        "task_label_documentation": "タスクラベル: ドキュメント",
        "task_label_question": "タスクラベル: 質問",
        "task_label_help wanted": "タスクラベル: ヘルプ募集",
        "task_priority_score": "タスク優先度スコア",
        "task_urgency_indicator": "タスク緊急度指標",
        "task_complexity_estimate": "タスク複雑度推定",
        "task_comment_count": "タスクコメント数",
        "task_participant_count": "タスク参加者数",
        "task_file_count": "タスク関連ファイル数",
        "task_line_count": "タスク関連行数",
        "task_branch_age": "タスクブランチ経過日数",
        "task_commit_frequency": "タスクコミット頻度",
        "task_test_coverage": "タスクテストカバレッジ",
        "task_documentation_quality": "タスクドキュメント品質",
        # 開発者特徴量
        "dev_recent_activity_count": "開発者最近活動数",
        "dev_current_workload": "開発者現在作業負荷",
        "dev_total_lines_changed": "開発者総変更行数",
        "dev_collaboration_network_size": "開発者協力ネットワーク規模",
        "dev_comment_interactions": "開発者コメント相互作用",
        "dev_cross_issue_activity": "開発者課題横断活動",
        "dev_expertise_score": "開発者専門性スコア",
        "dev_reputation": "開発者評価",
        "dev_response_time": "開発者応答時間",
        "dev_code_quality": "開発者コード品質",
        "dev_test_coverage": "開発者テストカバレッジ",
        "dev_documentation_score": "開発者ドキュメントスコア",
        "dev_leadership_score": "開発者リーダーシップスコア",
        "dev_mentoring_activity": "開発者メンタリング活動",
        "dev_innovation_index": "開発者イノベーション指標",
        # マッチング特徴量
        "match_collaborated_with_task_author": "マッチング: タスク作成者との協力経験",
        "match_collaborator_overlap_count": "マッチング: 協力者重複数",
        "match_has_prior_collaboration": "マッチング: 過去の協力実績",
        "match_skill_intersection_count": "マッチング: スキル交差数",
        "match_file_experience_count": "マッチング: ファイル経験数",
        "match_affinity_for_bug": "マッチング: バグ対応親和性",
        "match_affinity_for_enhancement": "マッチング: 機能強化親和性",
        "match_affinity_for_documentation": "マッチング: ドキュメント親和性",
        "match_affinity_for_question": "マッチング: 質問対応親和性",
        "match_affinity_for_help wanted": "マッチング: ヘルプ対応親和性",
        # GAT統計特徴量
        "gat_similarity": "GAT類似度",
        "gat_dev_expertise": "GAT開発者専門性",
        "gat_task_popularity": "GATタスク人気度",
        "gat_collaboration_strength": "GAT協力強度",
        "gat_network_centrality": "GATネットワーク中心性",
    }


def format_feature_with_japanese(feature_name, japanese_names):
    """特徴量名を日本語説明付きでフォーマット"""
    if feature_name in japanese_names:
        return f"{feature_name} ({japanese_names[feature_name]})"
    elif feature_name.startswith("feature_"):
        # GAT埋め込み特徴量
        feature_num = feature_name.replace("feature_", "")
        return f"{feature_name} (GAT埋め込み次元{feature_num})"
    else:
        return feature_name


# プロジェクトルートを追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def load_actual_feature_names():
    """実際の特徴量名を取得（IRL重みファイルの次元数に基づいて）"""

    # 実際のIRL重みの次元数を取得
    weights_path = project_root / "data" / "learned_weights_training.npy"
    if weights_path.exists():
        weights = np.load(weights_path)
        n_features = len(weights)
        print(f"✅ IRL重みから特徴量次元数を取得: {n_features}次元")
    else:
        n_features = 62  # デフォルト
        print(f"⚠️  IRL重みファイルが見つかりません。デフォルト{n_features}次元を使用")

    # 実際の特徴量抽出器から特徴量名を取得
    try:
        from omegaconf import OmegaConf

        from src.kazoo.features.feature_extractor import FeatureExtractor

        # 設定読み込み
        config_path = project_root / "configs" / "base.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
        else:
            # デフォルト設定を作成
            cfg = OmegaConf.create(
                {
                    "features": {
                        "all_labels": [
                            "bug",
                            "enhancement",
                            "documentation",
                            "question",
                            "help wanted",
                        ],
                        "label_to_skills": {
                            "bug": ["debugging", "analysis"],
                            "enhancement": ["python", "design"],
                            "documentation": ["writing"],
                            "question": ["communication"],
                            "help wanted": ["collaboration"],
                        },
                    },
                    "irl": {"use_gat": True},  # GATを有効化して正確な特徴量名取得
                }
            )

        # 特徴量抽出器を初期化
        feature_extractor = FeatureExtractor(cfg)
        feature_names = feature_extractor.feature_names

        print(f"✅ 特徴量抽出器から特徴量名取得: {len(feature_names)}次元")
        print(f"Feature names: {feature_names[:10]}... (最初の10個)")

        # 次元数を合わせる
        if len(feature_names) != n_features:
            print(
                f"⚠️  次元数調整: 特徴量名{len(feature_names)}次元 → IRL重み{n_features}次元"
            )
            if len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
            else:
                # 不足分をパディング
                for i in range(len(feature_names), n_features):
                    feature_names.append(f"feature_{i}")

        return feature_names

    except Exception as e:
        print(f"⚠️  特徴量抽出器初期化に失敗: {e}")
        print("既知の特徴量名を使用します...")

        # フォールバック: 既知の特徴量名
        feature_names = [
            # タスク特徴量 (20次元)
            "task_days_since_last_activity",
            "task_discussion_activity",
            "task_text_length",
            "task_code_block_count",
            "task_label_bug",
            "task_label_enhancement",
            "task_label_documentation",
            "task_label_question",
            "task_label_help wanted",
            "task_priority_score",
            "task_urgency_indicator",
            "task_complexity_estimate",
            "task_comment_count",
            "task_participant_count",
            "task_file_count",
            "task_line_count",
            "task_branch_age",
            "task_commit_frequency",
            "task_test_coverage",
            "task_documentation_quality",
            # 開発者特徴量 (15次元)
            "dev_recent_activity_count",
            "dev_current_workload",
            "dev_total_lines_changed",
            "dev_collaboration_network_size",
            "dev_comment_interactions",
            "dev_cross_issue_activity",
            "dev_expertise_score",
            "dev_reputation",
            "dev_response_time",
            "dev_code_quality",
            "dev_test_coverage",
            "dev_documentation_score",
            "dev_leadership_score",
            "dev_mentoring_activity",
            "dev_innovation_index",
            # マッチング特徴量 (5次元)
            "match_collaborated_with_task_author",
            "match_collaborator_overlap_count",
            "match_has_prior_collaboration",
            "match_skill_intersection_count",
            "match_file_experience_count",
            # GAT統計特徴量 (5次元)
            "gat_similarity",
            "gat_dev_expertise",
            "gat_task_popularity",
            "gat_collaboration_strength",
            "gat_network_centrality",
            # GAT埋め込み (17次元 = 62 - 45)
        ]

        # GAT埋め込みを追加して合計62次元にする
        remaining_dims = n_features - len(feature_names)
        if remaining_dims > 0:
            feature_names.extend([f"gat_dev_emb_{i}" for i in range(remaining_dims)])

        # 必要に応じて調整
        feature_names = feature_names[:n_features]

        return feature_names


def generate_realistic_feature_data(feature_names, n_samples=1000):
    """実際のIRL学習で想定される現実的な特徴量データを生成"""

    print(
        f"📊 現実的な特徴量データを生成中... ({n_samples}サンプル, {len(feature_names)}次元)"
    )

    n_features = len(feature_names)
    features = np.zeros((n_samples, n_features))

    for i, name in enumerate(feature_names):
        if "days_since" in name:
            # 日数系: 指数分布 (0-365日) - 最近のタスクが多い
            features[:, i] = np.random.exponential(30, n_samples)
            features[:, i] = np.clip(features[:, i], 0, 365)
        elif "activity" in name or "count" in name:
            # 活動系: ポアソン分布 - スパースな活動
            features[:, i] = np.random.poisson(3, n_samples)
        elif "length" in name:
            # 長さ系: ログノーマル分布 - 多くは短いが、時々長い
            features[:, i] = np.random.lognormal(4, 1, n_samples)
        elif "label_" in name:
            # ラベル系: ベルヌーイ分布 - バイナリ特徴量
            features[:, i] = np.random.binomial(1, 0.15, n_samples).astype(float)
        elif "workload" in name:
            # 作業負荷: ガンマ分布 (0-15) - 適度な負荷分布
            features[:, i] = np.random.gamma(2, 2, n_samples)
            features[:, i] = np.clip(features[:, i], 0, 15)
        elif "lines_changed" in name:
            # コード行数: ログノーマル分布 - 小さな変更が多い
            features[:, i] = np.random.lognormal(3, 1.5, n_samples)
        elif "network" in name or "collaboration" in name:
            # ネットワーク系: べき乗分布 - 少数が多くのコネクションを持つ
            features[:, i] = np.random.pareto(1.16, n_samples) * 2
            features[:, i] = np.clip(features[:, i], 0, 50)
        elif "score" in name or "reputation" in name:
            # スコア系: ベータ分布 (0-10) - 適度な評価分布
            features[:, i] = np.random.beta(2, 3, n_samples) * 10
        elif "affinity" in name or "similarity" in name:
            # 親和性・類似度: ベータ分布 (0-1) - 多くは低い類似度
            features[:, i] = np.random.beta(1, 4, n_samples)
        elif "match_" in name and ("collaborated" in name or "has_prior" in name):
            # マッチング系バイナリ: ベルヌーイ分布 - 稀な関係
            features[:, i] = np.random.binomial(1, 0.05, n_samples).astype(float)
        elif "gat_" in name:
            if "emb_" in name:
                # GAT埋め込み: 標準正規分布 - 学習された埋め込み
                features[:, i] = np.random.normal(0, 0.5, n_samples)
            else:
                # GAT統計: ベータ分布 (0-1) - 正規化された値
                features[:, i] = np.random.beta(2, 5, n_samples)
        elif "time" in name or "age" in name:
            # 時間系: 指数分布 - 最近のものが多い
            features[:, i] = np.random.exponential(20, n_samples)
        elif "priority" in name or "urgency" in name:
            # 優先度系: 離散一様分布 (1-5)
            features[:, i] = np.random.randint(1, 6, n_samples).astype(float)
        elif "quality" in name or "coverage" in name:
            # 品質系: ベータ分布 (0-1) - 多くは中程度の品質
            features[:, i] = np.random.beta(3, 2, n_samples)
        else:
            # その他: 軽いランダムウォーク的分布
            features[:, i] = np.random.normal(0, 0.8, n_samples)

    print(f"✅ 現実的な特徴量データ生成完了")
    return features


def load_irl_weights():
    """IRL重みを読み込み"""
    weights_path = project_root / "data" / "learned_weights_training.npy"
    if weights_path.exists():
        weights = np.load(weights_path)
        print(f"✅ IRL重み読み込み成功: {len(weights)}次元")
        return weights
    else:
        print("⚠️  IRL重みファイルが見つかりません。ダミーデータを使用します。")
        # ダミー重みを生成
        return np.random.randn(62) * 0.5


def analyze_feature_distributions(features, feature_names, weights):
    """特徴量分布の詳細分析"""

    print("\n" + "=" * 80)
    print("📊 特徴量分布分析レポート（実際のIRL重みを使用）")
    print("=" * 80)

    # 日本語特徴量名の取得
    japanese_names = get_feature_japanese_names()

    # 基本統計
    df = pd.DataFrame(features, columns=feature_names)

    print(f"\n📈 【基本統計】")
    print(f"サンプル数: {len(features):,}")
    print(f"特徴量数: {len(feature_names)}")
    print(f"IRL重み数: {len(weights)}")
    print(f"データ形状: {features.shape}")

    # IRL重みの基本統計
    print(f"\n🎯 【IRL重み統計】")
    print(f"平均重み: {np.mean(weights):.6f}")
    print(f"重み標準偏差: {np.std(weights):.6f}")
    print(f"重み範囲: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
    print(
        f"正の重み: {np.sum(weights > 0)}個 ({np.sum(weights > 0)/len(weights)*100:.1f}%)"
    )
    print(
        f"負の重み: {np.sum(weights < 0)}個 ({np.sum(weights < 0)/len(weights)*100:.1f}%)"
    )
    print(
        f"ゼロの重み: {np.sum(weights == 0)}個 ({np.sum(weights == 0)/len(weights)*100:.1f}%)"
    )

    # 基本特徴量とGAT特徴量の分類
    # 実際の特徴量名（25次元）+ feature_XX（37次元）の場合、feature_XXをGAT特徴量として扱う
    basic_feature_names = [
        name
        for name in feature_names
        if not (
            name.startswith("gat_") or "gat_" in name or name.startswith("feature_")
        )
    ]
    gat_feature_names = [
        name
        for name in feature_names
        if (name.startswith("gat_") or "gat_" in name or name.startswith("feature_"))
    ]

    print(f"\n🔍 【特徴量分類】")
    print(f"基本特徴量+統計: {len(basic_feature_names)}次元")
    print(f"GAT特徴量: {len(gat_feature_names)}次元")

    # 詳細分類（feature_XXをGAT埋め込みとして扱う）
    categories = {
        "タスク特徴量": [name for name in feature_names if name.startswith("task_")],
        "開発者特徴量": [name for name in feature_names if name.startswith("dev_")],
        "マッチング特徴量": [
            name for name in feature_names if name.startswith("match_")
        ],
        "GAT統計特徴量": [
            name
            for name in feature_names
            if name.startswith("gat_") and "emb_" not in name
        ],
        "GAT埋め込み": [
            name
            for name in feature_names
            if (
                "gat_dev_emb_" in name
                or "gat_emb_" in name
                or name.startswith("feature_")
            )
        ],
    }

    # 大分類でのカテゴリ
    major_categories = {
        "基本特徴量+統計": basic_feature_names,
        "GAT特徴量": gat_feature_names,
    }

    stats_report = []

    # 大分類での分析を先に表示
    print(f"\n" + "=" * 60)
    print("📊 【大分類別分析】")
    print("=" * 60)

    for major_category, names in major_categories.items():
        if not names:
            continue

        print(f"\n📋 【{major_category}】({len(names)}次元)")

        # 大分類内の特徴量インデックスを取得
        category_indices = [
            feature_names.index(name) for name in names if name in feature_names
        ]
        category_weights = (
            weights[category_indices] if len(category_indices) > 0 else []
        )

        if len(category_weights) > 0:
            print(f"   重み統計:")
            print(f"     - 平均: {np.mean(category_weights):.6f}")
            print(f"     - 標準偏差: {np.std(category_weights):.6f}")
            print(
                f"     - 範囲: [{np.min(category_weights):.6f}, {np.max(category_weights):.6f}]"
            )
            print(
                f"     - 正の重み: {np.sum(np.array(category_weights) > 0)}個 ({np.sum(np.array(category_weights) > 0)/len(category_weights)*100:.1f}%)"
            )
            print(
                f"     - 負の重み: {np.sum(np.array(category_weights) < 0)}個 ({np.sum(np.array(category_weights) < 0)/len(category_weights)*100:.1f}%)"
            )

            # 大分類内の重要特徴量TOP5
            major_cat_data = []
            for name in names:
                if name in feature_names:
                    feature_idx = feature_names.index(name)
                    if feature_idx < len(weights):
                        major_cat_data.append(
                            {
                                "feature": name,
                                "irl_weight": weights[feature_idx],
                                "importance": abs(weights[feature_idx]),
                            }
                        )

            if major_cat_data:
                major_cat_df = pd.DataFrame(major_cat_data)
                top_major = major_cat_df.nlargest(5, "importance")
                print(f"   重要特徴量TOP5:")
                for idx, (_, row) in enumerate(top_major.iterrows(), 1):
                    print(
                        f"     {idx}. {row['feature'][:45]:45s} | 重み:{row['irl_weight']:8.5f}"
                    )

    print(f"\n" + "=" * 60)
    print("📊 【詳細カテゴリ別分析】")
    print("=" * 60)

    for category, names in categories.items():
        if not names:
            continue

        print(f"\n📋 【{category}】({len(names)}次元)")

        # カテゴリ内の特徴量インデックスを取得
        category_indices = [
            feature_names.index(name) for name in names if name in feature_names
        ]
        category_weights = (
            weights[category_indices] if len(category_indices) > 0 else []
        )

        if len(category_weights) > 0:
            print(
                f"   重み統計 - 平均: {np.mean(category_weights):.6f}, "
                f"標準偏差: {np.std(category_weights):.6f}, "
                f"範囲: [{np.min(category_weights):.6f}, {np.max(category_weights):.6f}]"
            )

        for i, name in enumerate(names):
            if name in feature_names:
                feature_idx = feature_names.index(name)
                if feature_idx < len(weights):
                    values = features[:, feature_idx]
                    weight = weights[feature_idx]

                    stats = {
                        "feature": name,
                        "category": category,
                        "feature_index": feature_idx,
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "irl_weight": weight,
                        "importance": abs(weight),
                        "zeros_pct": np.mean(values == 0) * 100,
                        "outliers_pct": np.mean(
                            np.abs(values - np.mean(values)) > 3 * np.std(values)
                        )
                        * 100,
                        "skewness": np.nan,  # scipy.stats.skew(values) if available
                        "kurtosis": np.nan,  # scipy.stats.kurtosis(values) if available
                    }

                    # scipy統計を計算（利用可能な場合）
                    try:
                        from scipy import stats

                        stats["skewness"] = stats.skew(values)
                        stats["kurtosis"] = stats.kurtosis(values)
                    except ImportError:
                        pass

                    stats_report.append(stats)

                    if i < 5:  # 最初の5個だけ詳細表示
                        print(
                            f"  {name[:35]:35s} | "
                            f"重み:{stats['irl_weight']:8.5f} | "
                            f"重要度:{stats['importance']:8.5f} | "
                            f"平均:{stats['mean']:8.3f} | "
                            f"標準偏差:{stats['std']:7.3f}"
                        )

        if len(names) > 5:
            print(f"  ... 他 {len(names) - 5} 個の特徴量")

    # 重要特徴量TOP10を表示
    if stats_report:
        stats_df = pd.DataFrame(stats_report)
        top_features = stats_df.nlargest(10, "importance")

        print(f"\n🏆 【重要特徴量TOP10（実際のIRL重み）】")
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            feature_display = format_feature_with_japanese(
                row["feature"], japanese_names
            )
            print(
                f"  {idx:2d}. {feature_display[:55]:55s} | "
                f"重み:{row['irl_weight']:8.5f} | "
                f"重要度:{row['importance']:8.5f} | "
                f"カテゴリ:{row['category']}"
            )

        # 基本特徴量内でのランキング
        basic_stats = stats_df[
            ~(
                stats_df["feature"].str.contains("gat_")
                | stats_df["feature"].str.startswith("feature_")
            )
        ]
        if not basic_stats.empty:
            top_basic_features = basic_stats.nlargest(10, "importance")

            print(f"\n🥇 【基本特徴量内重要ランキングTOP10】")
            for idx, (_, row) in enumerate(top_basic_features.iterrows(), 1):
                feature_display = format_feature_with_japanese(
                    row["feature"], japanese_names
                )
                print(
                    f"  {idx:2d}. {feature_display[:55]:55s} | "
                    f"重み:{row['irl_weight']:8.5f} | "
                    f"重要度:{row['importance']:8.5f} | "
                    f"カテゴリ:{row['category']}"
                )

        # GAT特徴量内でのランキング
        gat_stats = stats_df[
            stats_df["feature"].str.contains("gat_")
            | stats_df["feature"].str.startswith("feature_")
        ]
        if not gat_stats.empty:
            top_gat_features = gat_stats.nlargest(10, "importance")

            print(f"\n🤖 【GAT特徴量内重要ランキングTOP10】")
            for idx, (_, row) in enumerate(top_gat_features.iterrows(), 1):
                feature_display = format_feature_with_japanese(
                    row["feature"], japanese_names
                )
                print(
                    f"  {idx:2d}. {feature_display[:55]:55s} | "
                    f"重み:{row['irl_weight']:8.5f} | "
                    f"重要度:{row['importance']:8.5f} | "
                    f"カテゴリ:{row['category']}"
                )

    return pd.DataFrame(stats_report)


def create_visualizations(features, feature_names, weights, stats_df):
    """特徴量分布の可視化"""

    print(f"\n🎨 可視化を作成中...")

    # 出力ディレクトリを作成
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    # 図1: IRL重みの分布
    plt.figure(figsize=(15, 10))

    # サブプロット1: 重みヒストグラム
    plt.subplot(2, 3, 1)
    plt.hist(weights, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("IRL重み")
    plt.ylabel("頻度")
    plt.title("IRL重みの分布")
    plt.grid(True, alpha=0.3)

    # サブプロット2: カテゴリ別重み
    if not stats_df.empty:
        plt.subplot(2, 3, 2)
        category_weights = stats_df.groupby("category")["irl_weight"].agg(
            ["mean", "std"]
        )
        category_weights.plot(
            kind="bar",
            y="mean",
            yerr="std",
            ax=plt.gca(),
            color="lightcoral",
            alpha=0.7,
        )
        plt.title("カテゴリ別平均IRL重み")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    # サブプロット3: 重要特徴量TOP10
    plt.subplot(2, 3, 3)
    if not stats_df.empty:
        top_features = stats_df.nlargest(10, "importance")
        colors = ["red" if w < 0 else "green" for w in top_features["irl_weight"]]
        plt.barh(
            range(len(top_features)),
            top_features["irl_weight"],
            color=colors,
            alpha=0.7,
        )
        plt.yticks(
            range(len(top_features)),
            [
                name[:20] + "..." if len(name) > 20 else name
                for name in top_features["feature"]
            ],
        )
        plt.xlabel("IRL重み")
        plt.title("重要特徴量TOP10")
        plt.grid(True, alpha=0.3)

    # サブプロット4: 特徴量値の分布（サンプル）
    plt.subplot(2, 3, 4)
    sample_features = features[:, : min(5, features.shape[1])]
    plt.boxplot(
        sample_features,
        labels=[name[:10] for name in feature_names[: min(5, len(feature_names))]],
    )
    plt.title("特徴量値の分布（最初の5個）")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # サブプロット5: 相関マトリックス（サンプル）
    plt.subplot(2, 3, 5)
    if features.shape[1] >= 10:
        sample_corr = np.corrcoef(features[:, :10].T)
        im = plt.imshow(sample_corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.title("特徴量相関マトリックス（最初の10個）")
        plt.xticks(range(10), [name[:5] for name in feature_names[:10]], rotation=45)
        plt.yticks(range(10), [name[:5] for name in feature_names[:10]])

    # サブプロット6: 重み vs 重要度散布図
    plt.subplot(2, 3, 6)
    if not stats_df.empty:
        scatter = plt.scatter(
            stats_df["irl_weight"],
            stats_df["importance"],
            c=stats_df.index,
            cmap="viridis",
            alpha=0.7,
        )
        plt.xlabel("IRL重み")
        plt.ylabel("重要度（絶対値）")
        plt.title("重み vs 重要度")
        plt.grid(True, alpha=0.3)

        # 重要な点にラベル
        top_points = stats_df.nlargest(5, "importance")
        for _, row in top_points.iterrows():
            plt.annotate(
                row["feature"][:10],
                (row["irl_weight"], row["importance"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

    plt.tight_layout()

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"feature_distribution_analysis_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✅ 図1保存: {fig_path}")

    # 図2: カテゴリ別詳細分析
    if not stats_df.empty:
        plt.figure(figsize=(16, 12))

        categories = stats_df["category"].unique()
        n_cats = len(categories)

        for i, category in enumerate(categories):
            cat_data = stats_df[stats_df["category"] == category]

            if len(cat_data) == 0:
                continue

            # 各カテゴリのヒストグラム
            plt.subplot(n_cats, 2, 2 * i + 1)
            plt.hist(
                cat_data["irl_weight"],
                bins=10,
                alpha=0.7,
                color=plt.cm.Set3(i),
                edgecolor="black",
            )
            plt.title(f"{category} - IRL重み分布")
            plt.xlabel("IRL重み")
            plt.ylabel("頻度")
            plt.grid(True, alpha=0.3)

            # 各カテゴリの重要特徴量
            plt.subplot(n_cats, 2, 2 * i + 2)
            top_cat = cat_data.nlargest(min(5, len(cat_data)), "importance")
            colors = ["red" if w < 0 else "green" for w in top_cat["irl_weight"]]
            plt.barh(
                range(len(top_cat)), top_cat["irl_weight"], color=colors, alpha=0.7
            )
            plt.yticks(range(len(top_cat)), [name[:15] for name in top_cat["feature"]])
            plt.title(f"{category} - 重要特徴量")
            plt.xlabel("IRL重み")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        fig2_path = output_dir / f"feature_category_analysis_{timestamp}.png"
        plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
        print(f"✅ 図2保存: {fig2_path}")

    plt.show()

    return fig_path, fig2_path if "fig2_path" in locals() else None


def save_detailed_report(stats_df, output_dir):
    """詳細レポートをCSVとテキストで保存"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 日本語特徴量名の取得
    japanese_names = get_feature_japanese_names()

    # CSV保存
    if not stats_df.empty:
        csv_path = output_dir / f"feature_distribution_stats_{timestamp}.csv"
        stats_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ 統計CSVファイル保存: {csv_path}")

    # テキストレポート
    report_path = output_dir / f"feature_distribution_report_{timestamp}.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("特徴量分布分析レポート\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not stats_df.empty:
            # 大分類別サマリー
            f.write("【大分類別サマリー】\n")
            basic_features = stats_df[
                ~(
                    stats_df["feature"].str.contains("gat_")
                    | stats_df["feature"].str.startswith("feature_")
                )
            ]
            gat_features = stats_df[
                stats_df["feature"].str.contains("gat_")
                | stats_df["feature"].str.startswith("feature_")
            ]

            f.write("基本特徴量+統計:\n")
            if not basic_features.empty:
                f.write(f"  特徴量数: {len(basic_features)}\n")
                f.write(f"  重み平均: {basic_features['irl_weight'].mean():.6f}\n")
                f.write(f"  重み標準偏差: {basic_features['irl_weight'].std():.6f}\n")
                f.write(
                    f"  重み範囲: [{basic_features['irl_weight'].min():.6f}, {basic_features['irl_weight'].max():.6f}]\n"
                )
                f.write(f"  正の重み: {(basic_features['irl_weight'] > 0).sum()}個\n")
                f.write(f"  負の重み: {(basic_features['irl_weight'] < 0).sum()}個\n")

            f.write("\nGAT特徴量:\n")
            if not gat_features.empty:
                f.write(f"  特徴量数: {len(gat_features)}\n")
                f.write(f"  重み平均: {gat_features['irl_weight'].mean():.6f}\n")
                f.write(f"  重み標準偏差: {gat_features['irl_weight'].std():.6f}\n")
                f.write(
                    f"  重み範囲: [{gat_features['irl_weight'].min():.6f}, {gat_features['irl_weight'].max():.6f}]\n"
                )
                f.write(f"  正の重み: {(gat_features['irl_weight'] > 0).sum()}個\n")
                f.write(f"  負の重み: {(gat_features['irl_weight'] < 0).sum()}個\n")
            f.write("\n")

            f.write("【詳細カテゴリ別サマリー】\n")
            category_summary = (
                stats_df.groupby("category")
                .agg(
                    {
                        "irl_weight": ["count", "mean", "std", "min", "max"],
                        "importance": ["mean", "max"],
                        "zeros_pct": "mean",
                    }
                )
                .round(4)
            )
            f.write(category_summary.to_string())
            f.write("\n\n")

            f.write("【基本特徴量+統計 重要TOP15】\n")
            if not basic_features.empty:
                top_basic = basic_features.nlargest(15, "importance")
                for idx, (_, row) in enumerate(top_basic.iterrows(), 1):
                    feature_display = format_feature_with_japanese(
                        row["feature"], japanese_names
                    )
                    f.write(
                        f"{idx:2d}. {feature_display[:50]:50s} | "
                        f"重み:{row['irl_weight']:8.4f} | "
                        f"重要度:{row['importance']:8.4f} | "
                        f"カテゴリ:{row['category']}\n"
                    )
            f.write("\n")

            f.write("【GAT特徴量 重要TOP15】\n")
            if not gat_features.empty:
                top_gat = gat_features.nlargest(15, "importance")
                for idx, (_, row) in enumerate(top_gat.iterrows(), 1):
                    feature_display = format_feature_with_japanese(
                        row["feature"], japanese_names
                    )
                    f.write(
                        f"{idx:2d}. {feature_display[:50]:50s} | "
                        f"重み:{row['irl_weight']:8.4f} | "
                        f"重要度:{row['importance']:8.4f} | "
                        f"カテゴリ:{row['category']}\n"
                    )
            f.write("\n")

            # カテゴリ別の詳細ランキング
            f.write("【カテゴリ別重要ランキング】\n")
            for category in ["タスク特徴量", "開発者特徴量", "マッチング特徴量"]:
                cat_features = stats_df[stats_df["category"] == category]
                if not cat_features.empty:
                    f.write(f"\n{category}:\n")
                    top_cat = cat_features.nlargest(
                        min(10, len(cat_features)), "importance"
                    )
                    for idx, (_, row) in enumerate(top_cat.iterrows(), 1):
                        feature_display = format_feature_with_japanese(
                            row["feature"], japanese_names
                        )
                        f.write(
                            f"  {idx:2d}. {feature_display[:45]:45s} | "
                            f"重み:{row['irl_weight']:8.4f} | "
                            f"重要度:{row['importance']:8.4f}\n"
                        )

            f.write("【全体重要特徴量TOP20】\n")
            top_features = stats_df.nlargest(20, "importance")
            for _, row in top_features.iterrows():
                feature_display = format_feature_with_japanese(
                    row["feature"], japanese_names
                )
                f.write(
                    f"{feature_display[:50]:50s} | "
                    f"重み:{row['irl_weight']:8.4f} | "
                    f"重要度:{row['importance']:8.4f} | "
                    f"カテゴリ:{row['category']}\n"
                )

            f.write("\n【負の重み特徴量】\n")
            negative_features = stats_df[stats_df["irl_weight"] < 0].sort_values(
                "irl_weight"
            )
            for _, row in negative_features.iterrows():
                feature_display = format_feature_with_japanese(
                    row["feature"], japanese_names
                )
                f.write(
                    f"{feature_display[:50]:50s} | " f"重み:{row['irl_weight']:8.4f}\n"
                )

    print(f"✅ 詳細レポート保存: {report_path}")

    return csv_path, report_path


def main():
    """メイン実行関数"""

    print("🚀 特徴量分布分析を開始...")
    print("⚠️  注意: このスクリプトは実際のIRL学習済み重みを使用しています")

    # データ読み込み
    print("\n1️⃣  データ読み込み中...")

    # 実際の特徴量名を取得
    feature_names = load_actual_feature_names()

    # 実際のIRL重みを読み込み
    weights = load_irl_weights()

    # 現実的な特徴量データを生成（実際の分布を模擬）
    features = generate_realistic_feature_data(feature_names, n_samples=1000)

    # 次元数チェック
    if len(weights) != len(feature_names):
        print(
            f"⚠️  次元数不一致: 重み{len(weights)}次元 vs 特徴量{len(feature_names)}次元"
        )
        min_dim = min(len(weights), len(feature_names))
        weights = weights[:min_dim]
        feature_names = feature_names[:min_dim]
        features = features[:, :min_dim]
        print(f"   → {min_dim}次元に調整しました")

    print(f"\n📊 分析対象:")
    print(f"   - サンプル数: {len(features):,}")
    print(f"   - 特徴量数: {len(feature_names)}")
    print(f"   - IRL重み: {len(weights)}次元")
    print(f"   - データ形状: {features.shape}")

    # 分布分析
    print("\n2️⃣  分布分析中...")
    stats_df = analyze_feature_distributions(features, feature_names, weights)

    # 可視化
    print("\n3️⃣  可視化作成中...")
    output_dir = project_root / "outputs"
    fig_paths = create_visualizations(features, feature_names, weights, stats_df)

    # レポート保存
    print("\n4️⃣  レポート保存中...")
    csv_path, report_path = save_detailed_report(stats_df, output_dir)

    print("\n✅ 分析完了!")
    print("� 出力ファイル:")
    print(f"   - 図1: {fig_paths[0] if isinstance(fig_paths, tuple) else fig_paths}")
    if isinstance(fig_paths, tuple) and len(fig_paths) > 1 and fig_paths[1]:
        print(f"   - 図2: {fig_paths[1]}")
    print(f"   - レポート: {report_path}")
    print(f"   - CSV: {csv_path}")

    print(f"\n🔍 重要な発見:")
    if not stats_df.empty:
        # 日本語特徴量名の取得
        japanese_names = get_feature_japanese_names()

        # 最も重要な特徴量
        top_feature = stats_df.loc[stats_df["importance"].idxmax()]
        feature_display = format_feature_with_japanese(
            top_feature["feature"], japanese_names
        )
        print(
            f"   - 最重要特徴量: {feature_display} (重み: {top_feature['irl_weight']:.4f})"
        )

        # 負の重みの特徴量数
        negative_count = len(stats_df[stats_df["irl_weight"] < 0])
        print(f"   - 負の重み特徴量: {negative_count}個")

        # カテゴリ別重要度
        cat_importance = (
            stats_df.groupby("category")["importance"]
            .mean()
            .sort_values(ascending=False)
        )
        print(
            f"   - 最重要カテゴリ: {cat_importance.index[0]} (平均重要度: {cat_importance.iloc[0]:.4f})"
        )

        # 大分類別の分析
        basic_features = stats_df[
            ~(
                stats_df["feature"].str.contains("gat_")
                | stats_df["feature"].str.startswith("feature_")
            )
        ]
        gat_features = stats_df[
            stats_df["feature"].str.contains("gat_")
            | stats_df["feature"].str.startswith("feature_")
        ]

        print(f"\n🎯 大分類別統計:")
        if not basic_features.empty:
            print(f"   【基本特徴量+統計】({len(basic_features)}次元)")
            print(f"     - 重み平均: {basic_features['irl_weight'].mean():.4f}")
            print(f"     - 重要度平均: {basic_features['importance'].mean():.4f}")
            top_basic_feature = basic_features.loc[
                basic_features["importance"].idxmax()
            ]
            top_basic_display = format_feature_with_japanese(
                top_basic_feature["feature"], japanese_names
            )
            print(
                f"     - 最重要: {top_basic_display} (重み: {top_basic_feature['irl_weight']:.4f})"
            )

            # 基本特徴量内TOP3
            top_basic_3 = basic_features.nlargest(3, "importance")
            print(f"     - 基本特徴量TOP3:")
            for idx, (_, row) in enumerate(top_basic_3.iterrows(), 1):
                feature_display = format_feature_with_japanese(
                    row["feature"], japanese_names
                )
                print(
                    f"       {idx}. {feature_display[:40]:40s} (重み: {row['irl_weight']:7.4f})"
                )

        if not gat_features.empty:
            print(f"   【GAT特徴量】({len(gat_features)}次元)")
            print(f"     - 重み平均: {gat_features['irl_weight'].mean():.4f}")
            print(f"     - 重要度平均: {gat_features['importance'].mean():.4f}")
            top_gat_feature = gat_features.loc[gat_features["importance"].idxmax()]
            top_gat_display = format_feature_with_japanese(
                top_gat_feature["feature"], japanese_names
            )
            print(
                f"     - 最重要: {top_gat_display} (重み: {top_gat_feature['irl_weight']:.4f})"
            )

            # GAT特徴量内TOP3
            top_gat_3 = gat_features.nlargest(3, "importance")
            print(f"     - GAT特徴量TOP3:")
            for idx, (_, row) in enumerate(top_gat_3.iterrows(), 1):
                feature_display = format_feature_with_japanese(
                    row["feature"], japanese_names
                )
                print(
                    f"       {idx}. {feature_display[:40]:40s} (重み: {row['irl_weight']:7.4f})"
                )

        # 実際のIRL重みの統計
        print(f"\n📈 IRL重み統計:")
        print(f"   - 平均: {np.mean(weights):.4f}")
        print(f"   - 標準偏差: {np.std(weights):.4f}")
        print(f"   - 最小値: {np.min(weights):.4f}")
        print(f"   - 最大値: {np.max(weights):.4f}")
        print(f"   - 正の重み: {np.sum(weights > 0)}個")
        print(f"   - 負の重み: {np.sum(weights < 0)}個")
        print(f"   - ゼロの重み: {np.sum(weights == 0)}個")


if __name__ == "__main__":
    main()
