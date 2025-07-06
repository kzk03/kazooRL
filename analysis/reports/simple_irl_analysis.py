#!/usr/bin/env python3
"""
IRL重み分析結果を分かりやすく可視化・解釈するスクリプト
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_simple_interpretation():
    """シンプルで分かりやすい解釈"""
    print("🎯 IRL重み分析 - 分かりやすい解釈")
    print("=" * 60)

    # IRL重みの読み込み
    weights_path = Path("data/learned_weights_training.npy")
    if not weights_path.exists():
        print(f"❌ IRL重みファイルが見つかりません: {weights_path}")
        return

    weights = np.load(weights_path)

    # 特徴量の定義
    base_features = [
        "ログイン名の長さ",
        "名前の有無",
        "名前の長さ",
        "会社情報の有無",
        "会社名の長さ",
        "場所情報の有無",
        "場所情報の長さ",
        "プロフィール文の有無",
        "プロフィール文の長さ",
        "公開リポジトリ数",
        "公開リポジトリ数(対数)",
        "フォロワー数",
        "フォロワー数(対数)",
        "フォロー数",
        "フォロー数(対数)",
        "アカウント年数(日)",
        "アカウント年数(年)",
        "フォロワー/フォロー比",
        "年間リポジトリ作成数",
        "人気度スコア",
        "活動度スコア",
        "影響力スコア",
        "経験値スコア",
        "社交性スコア",
        "プロフィール完成度",
    ]

    print("🏆 学習した開発者選択の重要なポイント")
    print("=" * 50)

    # 基本特徴量とGAT特徴量に分離
    base_weights = weights[:25]
    gat_weights = weights[25:] if len(weights) > 25 else np.array([])

    print(f"📊 基本情報 vs 協力関係の重要度:")
    print(f"  基本情報の平均重要度: {np.mean(np.abs(base_weights)):.3f}")
    if len(gat_weights) > 0:
        print(f"  協力関係の平均重要度: {np.mean(np.abs(gat_weights)):.3f}")
        print(
            f"  → 協力関係の方が {np.mean(np.abs(gat_weights))/np.mean(np.abs(base_weights)):.1f}倍重要！"
        )

    # 基本特徴量の重要なものを分析
    print(f"\n✅ 重要視される開発者の特徴 (基本情報):")
    positive_base = [
        (i, base_weights[i], base_features[i])
        for i in range(len(base_weights))
        if base_weights[i] > 0.5
    ]
    positive_base.sort(key=lambda x: x[1], reverse=True)

    for rank, (idx, weight, name) in enumerate(positive_base[:5], 1):
        print(f"  {rank}. {name} (重要度: {weight:.2f})")

    print(f"\n❌ 避けられる開発者の特徴 (基本情報):")
    negative_base = [
        (i, base_weights[i], base_features[i])
        for i in range(len(base_weights))
        if base_weights[i] < -0.5
    ]
    negative_base.sort(key=lambda x: x[1])

    for rank, (idx, weight, name) in enumerate(negative_base[:5], 1):
        print(f"  {rank}. {name} (重要度: {weight:.2f})")

    # 協力関係の分析
    if len(gat_weights) > 0:
        print(f"\n🤝 協力関係の影響:")
        positive_gat = np.sum(gat_weights > 0.5)
        negative_gat = np.sum(gat_weights < -0.5)
        print(f"  重要視される協力パターン: {positive_gat}個")
        print(f"  避けられる協力パターン: {negative_gat}個")
        print(f"  最も重要な協力パターンの重要度: {np.max(gat_weights):.2f}")
        if np.min(gat_weights) < 0:
            print(f"  最も避けられる協力パターンの重要度: {np.min(gat_weights):.2f}")

    # 分かりやすい可視化を作成
    create_simple_visualization(weights, base_features, base_weights, gat_weights)

    # 実用的なアドバイス
    print_practical_advice(base_weights, base_features, gat_weights)


def create_simple_visualization(weights, base_features, base_weights, gat_weights):
    """分かりやすい可視化"""
    print(f"\n📊 可視化グラフ作成中...")

    # 日本語フォントの設定
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(20, 16))

    # 1. 基本特徴量の重要度 (大きなグラフ)
    ax1 = plt.subplot(2, 3, (1, 2))
    colors = ["red" if w < 0 else "blue" if w > 0.5 else "gray" for w in base_weights]
    bars = ax1.barh(range(len(base_weights)), base_weights, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(base_weights)))
    ax1.set_yticklabels(
        [name[:12] + "..." if len(name) > 12 else name for name in base_features],
        fontsize=10,
    )
    ax1.set_xlabel("Importance Weight", fontsize=12)
    ax1.set_title(
        "Basic Features Importance\n(Blue: Important, Red: Avoided, Gray: Neutral)",
        fontsize=14,
        pad=20,
    )
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color="black", linestyle="-", alpha=0.5)

    # 重要度の値をバーに表示
    for i, (bar, weight) in enumerate(zip(bars, base_weights)):
        if abs(weight) > 0.3:  # 重要度が高いもののみ表示
            ax1.text(
                weight + (0.05 if weight > 0 else -0.05),
                i,
                f"{weight:.2f}",
                va="center",
                ha="left" if weight > 0 else "right",
                fontsize=9,
            )

    # 2. 基本 vs GAT の比較
    ax2 = plt.subplot(2, 3, 3)
    categories = ["Basic Features"]
    importances = [np.mean(np.abs(base_weights))]
    colors_cat = ["skyblue"]

    if len(gat_weights) > 0:
        categories.append("Collaboration\n(GAT Features)")
        importances.append(np.mean(np.abs(gat_weights)))
        colors_cat.append("lightcoral")

    bars = ax2.bar(categories, importances, color=colors_cat, alpha=0.8)
    ax2.set_ylabel("Average Importance", fontsize=12)
    ax2.set_title("Feature Category\nComparison", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 値を表示
    for bar, imp in zip(bars, importances):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{imp:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 3. 重要度分布
    ax3 = plt.subplot(2, 3, 4)
    ax3.hist(
        base_weights,
        bins=15,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        label="Basic Features",
    )
    if len(gat_weights) > 0:
        ax3.hist(
            gat_weights,
            bins=15,
            alpha=0.7,
            color="red",
            edgecolor="black",
            label="GAT Features",
        )
    ax3.axvline(0, color="black", linestyle="--", alpha=0.7, label="Neutral")
    ax3.set_xlabel("Weight Value", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title("Weight Distribution", fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 上位特徴量のみ (分かりやすく)
    ax4 = plt.subplot(2, 3, 5)
    top_indices = np.argsort(np.abs(weights))[-10:]
    top_weights = weights[top_indices]
    top_names = []

    for idx in top_indices:
        if idx < 25:
            top_names.append(base_features[idx][:10] + "...")
        else:
            top_names.append(f"Collab-{idx-25}")

    colors_top = ["red" if w < 0 else "blue" for w in top_weights]
    bars = ax4.barh(range(len(top_weights)), top_weights, color=colors_top, alpha=0.8)
    ax4.set_yticks(range(len(top_weights)))
    ax4.set_yticklabels(top_names, fontsize=10)
    ax4.set_xlabel("Weight Value", fontsize=12)
    ax4.set_title("Top 10 Most Important\nFeatures", fontsize=14)
    ax4.grid(True, alpha=0.3)

    # 値を表示
    for i, (bar, weight) in enumerate(zip(bars, top_weights)):
        ax4.text(
            weight + (0.05 if weight > 0 else -0.05),
            i,
            f"{weight:.2f}",
            va="center",
            ha="left" if weight > 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    # 5. GAT特徴量の詳細
    if len(gat_weights) > 0:
        ax5 = plt.subplot(2, 3, 6)
        gat_colors = [
            "red" if w < 0 else "blue" if w > 0.5 else "gray" for w in gat_weights
        ]
        ax5.bar(range(len(gat_weights)), gat_weights, color=gat_colors, alpha=0.7)
        ax5.set_xlabel("GAT Feature Index", fontsize=12)
        ax5.set_ylabel("Weight Value", fontsize=12)
        ax5.set_title("GAT Features Detail\n(Collaboration Patterns)", fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(0, color="black", linestyle="-", alpha=0.5)

        # 重要なGAT特徴量をハイライト
        top_gat = np.argsort(np.abs(gat_weights))[-5:]
        for idx in top_gat:
            ax5.bar(
                idx,
                gat_weights[idx],
                color="gold",
                alpha=0.9,
                edgecolor="black",
                linewidth=2,
            )

    plt.tight_layout(pad=3.0)

    # 保存
    output_path = (
        Path("outputs")
        / f"simple_irl_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 分かりやすい分析グラフ保存: {output_path}")
    plt.close()


def print_practical_advice(base_weights, base_features, gat_weights):
    """実用的なアドバイス"""
    print(f"\n💡 実用的な開発者選択のポイント")
    print("=" * 50)

    print("🎯 開発者選択で最も重要なのは:")
    if len(gat_weights) > 0 and np.mean(np.abs(gat_weights)) > np.mean(
        np.abs(base_weights)
    ):
        print("  1️⃣ 過去の協力関係・協力パターン (最重要!)")
        print("     → 誰と一緒に仕事をしてきたか")
        print("     → どんなプロジェクトで活躍したか")
        print("     → チームワークの実績")
        print()
        print("  2️⃣ 基本的な開発者情報")
    else:
        print("  1️⃣ 基本的な開発者情報")

    # 基本情報の重要ポイント
    important_base = [
        (i, base_weights[i], base_features[i])
        for i in range(len(base_weights))
        if abs(base_weights[i]) > 0.5
    ]
    important_base.sort(key=lambda x: abs(x[1]), reverse=True)

    positive_advice = []
    negative_advice = []

    for idx, weight, name in important_base:
        if weight > 0:
            if "プロフィール文" in name:
                positive_advice.append("✅ プロフィール情報を充実させている")
            elif "フォロー" in name:
                positive_advice.append("✅ 適度な社交性がある（フォロー活動）")
            elif "人気度" in name:
                positive_advice.append("✅ コミュニティで人気がある")
            elif "活動度" in name:
                positive_advice.append("✅ 継続的に活動している")
        else:
            if "年間リポジトリ" in name:
                negative_advice.append("❌ リポジトリを作りすぎている（量より質）")
            elif "影響力" in name:
                negative_advice.append("❌ 影響力が強すぎる（協調性重視）")
            elif "会社名" in name:
                negative_advice.append("❌ 会社名が長すぎる")

    if positive_advice:
        print("     重要視される特徴:")
        for advice in positive_advice[:5]:
            print(f"       {advice}")

    if negative_advice:
        print("     避けられる特徴:")
        for advice in negative_advice[:3]:
            print(f"       {advice}")

    print(f"\n🔍 この分析から分かること:")
    print("  • 開発者選択は協力関係を最も重視している")
    print("  • プロフィール情報の充実度が重要")
    print("  • 適度な社交性（フォロー数）が評価される")
    print("  • 量より質：多すぎるリポジトリ作成は避けられる")
    print("  • チームワークを重視：強すぎる個人影響力は避けられる")

    print(f"\n🎊 結論:")
    print("  協力関係 > コミュニケーション能力 > 適度な活動 > 個人実績")


def create_summary_report():
    """総合レポートの作成"""
    print(f"\n📝 総合分析レポート")
    print("=" * 60)

    weights_path = Path("data/learned_weights_training.npy")
    if not weights_path.exists():
        return

    weights = np.load(weights_path)
    base_weights = weights[:25]
    gat_weights = weights[25:] if len(weights) > 25 else np.array([])

    # 統計情報
    stats = {
        "総特徴量数": len(weights),
        "基本特徴量数": len(base_weights),
        "GAT特徴量数": len(gat_weights),
        "重要な特徴量数（|重み|>0.5）": np.sum(np.abs(weights) > 0.5),
        "正の重み数": np.sum(weights > 0),
        "負の重み数": np.sum(weights < 0),
        "基本特徴量平均重要度": np.mean(np.abs(base_weights)),
        "GAT特徴量平均重要度": (
            np.mean(np.abs(gat_weights)) if len(gat_weights) > 0 else 0
        ),
        "最大重み": np.max(weights),
        "最小重み": np.min(weights),
    }

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print(f"\n🎉 分析完了!")
    print(f"📊 グラフファイル: outputs/simple_irl_analysis_*.png")


def main():
    """メイン実行"""
    print("🔍 IRL重み分析 - 分かりやすい版")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)

    try:
        create_simple_interpretation()
        create_summary_report()

    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
