#!/usr/bin/env python3
"""
IRL重みの詳細分析 - 各特徴量が何を表しているかを分析
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("src")


def get_feature_names_and_descriptions():
    """特徴量名と説明の対応を取得"""

    # 基本特徴量（25次元）
    base_features = [
        ("login_length", "ログイン名の長さ"),
        ("name_exists", "名前の有無"),
        ("name_length", "名前の長さ"),
        ("company_exists", "会社情報の有無"),
        ("company_length", "会社名の長さ"),
        ("location_exists", "場所情報の有無"),
        ("location_length", "場所情報の長さ"),
        ("bio_exists", "プロフィール文の有無"),
        ("bio_length", "プロフィール文の長さ"),
        ("public_repos", "公開リポジトリ数"),
        ("public_repos_log", "公開リポジトリ数（対数）"),
        ("followers", "フォロワー数"),
        ("followers_log", "フォロワー数（対数）"),
        ("following", "フォロー数"),
        ("following_log", "フォロー数（対数）"),
        ("account_age_days", "アカウント年数（日）"),
        ("account_age_years", "アカウント年数（年）"),
        ("followers_following_ratio", "フォロワー/フォロー比"),
        ("repos_per_year", "年間リポジトリ作成数"),
        ("popularity_score", "人気度スコア"),
        ("activity_score", "活動度スコア"),
        ("influence_score", "影響力スコア"),
        ("experience_score", "経験値スコア"),
        ("social_score", "社交性スコア"),
        ("profile_completeness", "プロフィール完成度"),
    ]

    # GAT特徴量（37次元）
    gat_features = []
    for i in range(37):
        gat_features.append((f"gat_feature_{i}", f"GAT特徴量{i}（協力関係埋め込み）"))

    return base_features + gat_features


def analyze_irl_weights_detailed():
    """IRL重みの詳細分析"""
    print("🎯 IRL重み詳細分析")
    print("=" * 60)

    # IRL重みの読み込み
    weights_path = Path("data/learned_weights_training.npy")
    if not weights_path.exists():
        print(f"❌ IRL重みファイルが見つかりません: {weights_path}")
        return

    weights = np.load(weights_path)
    print(f"✅ IRL重み読み込み成功: {weights.shape}")

    # 特徴量名とその説明を取得
    feature_info = get_feature_names_and_descriptions()

    if len(weights) != len(feature_info):
        print(
            f"⚠️ 重み数({len(weights)})と特徴量定義数({len(feature_info)})が一致しません"
        )
        # 重みの数に合わせて調整
        if len(weights) < len(feature_info):
            feature_info = feature_info[: len(weights)]
        else:
            # 足りない分は汎用的な名前で補完
            for i in range(len(feature_info), len(weights)):
                feature_info.append((f"unknown_feature_{i}", f"未定義特徴量{i}"))

    print(f"📊 分析対象特徴量数: {len(weights)}")

    # 重みの統計情報
    print(f"\n📈 重み統計:")
    print(f"  - 平均: {weights.mean():.6f}")
    print(f"  - 標準偏差: {weights.std():.6f}")
    print(f"  - 最小値: {weights.min():.6f}")
    print(f"  - 最大値: {weights.max():.6f}")

    # 重要な特徴量の分析（絶対値順）
    print(f"\n🔝 重要度ランキング（絶対値順）:")
    importance_indices = np.argsort(np.abs(weights))[::-1]  # 降順

    print("順位 | 重み値    | 特徴量名                | 説明")
    print("-" * 70)
    for rank, idx in enumerate(importance_indices[:15], 1):
        weight_val = weights[idx]
        feature_name, description = feature_info[idx]
        print(f"{rank:2d}位 | {weight_val:8.4f} | {feature_name:20s} | {description}")

    # 正の重みと負の重みの分析
    positive_weights = weights[weights > 0]
    negative_weights = weights[weights < 0]
    zero_weights = weights[weights == 0]

    print(f"\n📊 重みの符号分析:")
    print(
        f"  - 正の重み: {len(positive_weights)}個 (平均: {positive_weights.mean():.4f})"
    )
    print(
        f"  - 負の重み: {len(negative_weights)}個 (平均: {negative_weights.mean():.4f})"
    )
    print(f"  - ゼロの重み: {len(zero_weights)}個")

    # 正の重みトップ10
    print(f"\n✅ 正の影響が大きい特徴量 Top 10:")
    positive_indices = np.where(weights > 0)[0]
    positive_sorted = positive_indices[np.argsort(weights[positive_indices])[::-1]]

    for rank, idx in enumerate(positive_sorted[:10], 1):
        weight_val = weights[idx]
        feature_name, description = feature_info[idx]
        print(f"{rank:2d}. {weight_val:6.4f} | {feature_name:20s} | {description}")

    # 負の重みトップ10
    print(f"\n❌ 負の影響が大きい特徴量 Top 10:")
    negative_indices = np.where(weights < 0)[0]
    negative_sorted = negative_indices[
        np.argsort(weights[negative_indices])
    ]  # 昇順（最も負の値）

    for rank, idx in enumerate(negative_sorted[:10], 1):
        weight_val = weights[idx]
        feature_name, description = feature_info[idx]
        print(f"{rank:2d}. {weight_val:6.4f} | {feature_name:20s} | {description}")

    # 特徴量カテゴリ別分析
    analyze_by_category(weights, feature_info)

    # 可視化
    create_detailed_visualization(weights, feature_info)

    return weights, feature_info


def analyze_by_category(weights, feature_info):
    """カテゴリ別の重み分析"""
    print(f"\n🏷️ カテゴリ別重み分析:")
    print("-" * 50)

    # 基本特徴量 vs GAT特徴量
    base_weights = weights[:25]  # 最初の25個
    gat_weights = weights[25:] if len(weights) > 25 else np.array([])

    print(f"📋 基本特徴量（25次元）:")
    print(f"  - 平均重み: {base_weights.mean():.4f}")
    print(f"  - 標準偏差: {base_weights.std():.4f}")
    print(f"  - 最大値: {base_weights.max():.4f}")
    print(f"  - 最小値: {base_weights.min():.4f}")

    if len(gat_weights) > 0:
        print(f"🧠 GAT特徴量（{len(gat_weights)}次元）:")
        print(f"  - 平均重み: {gat_weights.mean():.4f}")
        print(f"  - 標準偏差: {gat_weights.std():.4f}")
        print(f"  - 最大値: {gat_weights.max():.4f}")
        print(f"  - 最小値: {gat_weights.min():.4f}")

    # サブカテゴリ分析（基本特徴量内で）
    categories = {
        "プロフィール情報": list(range(0, 9)),  # login～bio関連
        "活動指標": list(range(9, 17)),  # repos, followers, following関連
        "計算済み指標": list(range(17, 25)),  # ratio, score関連
    }

    for cat_name, indices in categories.items():
        if max(indices) < len(weights):
            cat_weights = weights[indices]
            print(f"\n📊 {cat_name}:")
            print(f"  - 平均重み: {cat_weights.mean():.4f}")
            print(f"  - 重要度上位:")
            sorted_indices = np.argsort(np.abs(cat_weights))[::-1]
            for i, idx in enumerate(sorted_indices[:3]):
                global_idx = indices[idx]
                feature_name, description = feature_info[global_idx]
                print(f"    {i+1}. {weights[global_idx]:6.4f} | {feature_name}")


def create_detailed_visualization(weights, feature_info):
    """詳細な可視化"""
    print(f"\n📊 可視化グラフ生成中...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 重み値の棒グラフ（上位20）
    top_20_indices = np.argsort(np.abs(weights))[-20:]
    top_20_weights = weights[top_20_indices]
    top_20_names = [feature_info[i][0] for i in top_20_indices]

    colors = ["red" if w < 0 else "blue" for w in top_20_weights]
    bars = ax1.barh(range(len(top_20_weights)), top_20_weights, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_20_weights)))
    ax1.set_yticklabels(
        [name[:15] + "..." if len(name) > 15 else name for name in top_20_names],
        fontsize=8,
    )
    ax1.set_xlabel("Weight Value")
    ax1.set_title("Top 20 Features by Absolute Weight")
    ax1.grid(True, alpha=0.3)

    # 2. 基本特徴量 vs GAT特徴量の比較
    base_weights = weights[:25]
    gat_weights = weights[25:] if len(weights) > 25 else np.array([])

    categories = ["Base Features"]
    means = [base_weights.mean()]
    stds = [base_weights.std()]

    if len(gat_weights) > 0:
        categories.append("GAT Features")
        means.append(gat_weights.mean())
        stds.append(gat_weights.std())

    x_pos = np.arange(len(categories))
    ax2.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.7,
        color=["skyblue", "lightcoral"][: len(categories)],
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Mean Weight")
    ax2.set_title("Feature Category Comparison")
    ax2.grid(True, alpha=0.3)

    # 3. 重みの分布ヒストグラム
    ax3.hist(weights, bins=30, alpha=0.7, edgecolor="black")
    ax3.axvline(
        weights.mean(), color="red", linestyle="--", label=f"Mean: {weights.mean():.3f}"
    )
    ax3.axvline(0, color="black", linestyle="-", alpha=0.5, label="Zero")
    ax3.set_xlabel("Weight Value")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Weight Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 重みの累積分布
    sorted_abs_weights = np.sort(np.abs(weights))[::-1]
    cumsum_weights = np.cumsum(sorted_abs_weights)
    cumsum_normalized = cumsum_weights / cumsum_weights[-1] * 100

    ax4.plot(range(1, len(weights) + 1), cumsum_normalized, "b-", linewidth=2)
    ax4.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% threshold")
    ax4.axhline(95, color="orange", linestyle="--", alpha=0.7, label="95% threshold")
    ax4.set_xlabel("Number of Features")
    ax4.set_ylabel("Cumulative Importance (%)")
    ax4.set_title("Cumulative Feature Importance")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 重要度80%を占める特徴量数を計算
    threshold_80_idx = np.where(cumsum_normalized >= 80)[0][0] + 1
    threshold_95_idx = np.where(cumsum_normalized >= 95)[0][0] + 1
    ax4.axvline(threshold_80_idx, color="red", linestyle=":", alpha=0.7)
    ax4.axvline(threshold_95_idx, color="orange", linestyle=":", alpha=0.7)

    plt.tight_layout()

    # 保存
    output_path = (
        Path("outputs")
        / f"irl_detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 詳細分析グラフ保存: {output_path}")
    plt.close()

    print(f"\n💡 重要度分析結果:")
    print(f"  - 重要度80%を占める特徴量数: {threshold_80_idx}/{len(weights)}")
    print(f"  - 重要度95%を占める特徴量数: {threshold_95_idx}/{len(weights)}")


def interpret_results(weights, feature_info):
    """結果の解釈"""
    print(f"\n🧠 IRL重み結果の解釈:")
    print("=" * 60)

    # 最も影響力の大きい特徴量
    top_positive_idx = np.argmax(weights)
    top_negative_idx = np.argmin(weights)

    print(f"🔝 最も正の影響が大きい特徴量:")
    feature_name, description = feature_info[top_positive_idx]
    print(f"   {feature_name} ({weights[top_positive_idx]:.4f})")
    print(f"   → {description}")
    print(f"   → 開発者選択において強く優先される要素")

    print(f"\n🔻 最も負の影響が大きい特徴量:")
    feature_name, description = feature_info[top_negative_idx]
    print(f"   {feature_name} ({weights[top_negative_idx]:.4f})")
    print(f"   → {description}")
    print(f"   → 開発者選択において避けられる要素")

    # GAT特徴量の影響
    if len(weights) > 25:
        gat_weights = weights[25:]
        gat_importance = np.mean(np.abs(gat_weights))
        base_importance = np.mean(np.abs(weights[:25]))

        print(f"\n🤝 協力関係（GAT）の影響度:")
        print(f"   基本特徴量の平均重要度: {base_importance:.4f}")
        print(f"   GAT特徴量の平均重要度: {gat_importance:.4f}")

        if gat_importance > base_importance:
            print(f"   → 協力関係情報が開発者選択により強く影響している")
        else:
            print(f"   → 基本的な開発者属性がより重要")

    # 実用的な示唆
    print(f"\n💼 実用的な示唆:")

    # 正の重みが大きい特徴量から示唆を導出
    positive_indices = np.where(weights > 0)[0]
    top_positive = positive_indices[np.argsort(weights[positive_indices])[-5:]]

    print(f"   優先される開発者特性:")
    for idx in reversed(top_positive):
        feature_name, description = feature_info[idx]
        print(f"   ✅ {description} (重み: {weights[idx]:.3f})")

    # 負の重みが大きい特徴量から示唆を導出
    negative_indices = np.where(weights < 0)[0]
    if len(negative_indices) > 0:
        top_negative = negative_indices[np.argsort(weights[negative_indices])[:5]]

        print(f"\n   避けられる傾向がある特性:")
        for idx in top_negative:
            feature_name, description = feature_info[idx]
            print(f"   ❌ {description} (重み: {weights[idx]:.3f})")


def main():
    """メイン実行"""
    print("🔍 IRL重み詳細分析")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)

    try:
        weights, feature_info = analyze_irl_weights_detailed()
        interpret_results(weights, feature_info)

        print(f"\n🎉 分析完了!")
        print(f"📊 分析結果:")
        print(f"   - 総特徴量数: {len(weights)}")
        print(f"   - 有意な重み数: {np.sum(np.abs(weights) > 0.01)}")
        print(f"   - 正の重み数: {np.sum(weights > 0)}")
        print(f"   - 負の重み数: {np.sum(weights < 0)}")

    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
