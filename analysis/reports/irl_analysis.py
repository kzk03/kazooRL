#!/usr/bin/env python3
"""
統合IRL分析スクリプト
複数のIRL分析機能を統合し、分かりやすいレポートを生成
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../src")


class IRLAnalyzer:
    """IRL結果の統合分析クラス"""

    def __init__(self, weights_path="data/learned_weights_training.npy"):
        self.weights_path = Path(weights_path)
        self.weights = None
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        """特徴量名の定義"""
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

        gat_features = [f"GAT特徴量{i}" for i in range(37)]
        return base_features + gat_features

    def load_weights(self):
        """重みを読み込み"""
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"重みファイルが見つかりません: {self.weights_path}"
            )

        self.weights = np.load(self.weights_path)
        print(f"✅ IRL重み読み込み成功: {self.weights.shape}")
        return self.weights

    def analyze_weights(self):
        """重みの基本分析"""
        if self.weights is None:
            self.load_weights()

        analysis = {
            "総特徴量数": len(self.weights),
            "基本特徴量数": 25,
            "GAT特徴量数": len(self.weights) - 25,
            "重要特徴量数": np.sum(np.abs(self.weights) > 0.5),
            "正の重み数": np.sum(self.weights > 0),
            "負の重み数": np.sum(self.weights < 0),
            "平均重み": self.weights.mean(),
            "標準偏差": self.weights.std(),
            "最大重み": self.weights.max(),
            "最小重み": self.weights.min(),
        }

        return analysis

    def get_important_features(self, top_n=10):
        """重要な特徴量を取得"""
        if self.weights is None:
            self.load_weights()

        # 絶対値で重要度をソート
        importance_indices = np.argsort(np.abs(self.weights))[::-1]

        important_features = []
        for i in range(min(top_n, len(importance_indices))):
            idx = importance_indices[i]
            weight = self.weights[idx]
            name = (
                self.feature_names[idx]
                if idx < len(self.feature_names)
                else f"特徴量{idx}"
            )
            important_features.append((name, weight, idx))

        return important_features

    def generate_simple_report(self):
        """分かりやすいレポートを生成"""
        if self.weights is None:
            self.load_weights()

        print("🎯 IRL学習結果 - 分かりやすい解釈")
        print("=" * 60)

        # 基本統計
        analysis = self.analyze_weights()

        print("📊 学習結果サマリー:")
        print(f"  分析した特徴量数: {analysis['総特徴量数']}")
        print(f"  重要な特徴量数: {analysis['重要特徴量数']}")
        print(f"  正の影響: {analysis['正の重み数']}個")
        print(f"  負の影響: {analysis['負の重み数']}個")

        # 協力関係 vs 基本情報
        base_weights = self.weights[:25]
        gat_weights = self.weights[25:] if len(self.weights) > 25 else np.array([])

        base_importance = np.mean(np.abs(base_weights))
        gat_importance = np.mean(np.abs(gat_weights)) if len(gat_weights) > 0 else 0

        print(f"\n🤝 協力関係 vs 基本情報:")
        print(f"  基本情報の重要度: {base_importance:.3f}")
        print(f"  協力関係の重要度: {gat_importance:.3f}")

        if gat_importance > base_importance:
            ratio = gat_importance / base_importance
            print(f"  → 協力関係が {ratio:.1f}倍重要！")

        # 重要な特徴量
        important_features = self.get_important_features(10)

        print(f"\n✅ 最重要特徴量 Top 10:")
        for rank, (name, weight, idx) in enumerate(important_features, 1):
            status = "優先" if weight > 0 else "回避"
            print(f"  {rank:2d}. {name[:20]:20s} ({status}: {weight:6.3f})")

        return analysis, important_features

    def create_visualization(self):
        """可視化グラフを作成"""
        if self.weights is None:
            self.load_weights()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 重要度ランキング
        important_features = self.get_important_features(15)
        names = [
            f[0][:15] + "..." if len(f[0]) > 15 else f[0] for f in important_features
        ]
        weights = [f[1] for f in important_features]
        colors = ["blue" if w > 0 else "red" for w in weights]

        ax1.barh(range(len(weights)), weights, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(weights)))
        ax1.set_yticklabels(names, fontsize=10)
        ax1.set_xlabel("Weight Value")
        ax1.set_title("Top 15 Most Important Features")
        ax1.grid(True, alpha=0.3)

        # 2. 基本 vs GAT比較
        base_weights = self.weights[:25]
        gat_weights = self.weights[25:] if len(self.weights) > 25 else np.array([])

        categories = ["Basic Features"]
        importances = [np.mean(np.abs(base_weights))]

        if len(gat_weights) > 0:
            categories.append("GAT Features")
            importances.append(np.mean(np.abs(gat_weights)))

        ax2.bar(
            categories,
            importances,
            color=["skyblue", "lightcoral"][: len(categories)],
            alpha=0.8,
        )
        ax2.set_ylabel("Average Importance")
        ax2.set_title("Feature Category Comparison")
        ax2.grid(True, alpha=0.3)

        # 3. 重み分布
        ax3.hist(self.weights, bins=30, alpha=0.7, edgecolor="black")
        ax3.axvline(0, color="red", linestyle="--", label="Zero")
        ax3.set_xlabel("Weight Value")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Weight Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 累積重要度
        sorted_abs_weights = np.sort(np.abs(self.weights))[::-1]
        cumsum_weights = np.cumsum(sorted_abs_weights)
        cumsum_normalized = cumsum_weights / cumsum_weights[-1] * 100

        ax4.plot(range(1, len(self.weights) + 1), cumsum_normalized, "b-", linewidth=2)
        ax4.axhline(80, color="red", linestyle="--", alpha=0.7, label="80%")
        ax4.axhline(95, color="orange", linestyle="--", alpha=0.7, label="95%")
        ax4.set_xlabel("Number of Features")
        ax4.set_ylabel("Cumulative Importance (%)")
        ax4.set_title("Cumulative Feature Importance")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        output_path = (
            Path("outputs")
            / f"irl_unified_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ 分析グラフ保存: {output_path}")
        plt.close()

        return output_path


def main():
    """メイン実行"""
    print("🔍 統合IRL分析")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)

    try:
        analyzer = IRLAnalyzer()
        analysis, important_features = analyzer.generate_simple_report()
        output_path = analyzer.create_visualization()

        print(f"\n🎉 分析完了!")
        print(f"📊 可視化: {output_path}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
