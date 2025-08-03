"""
特徴量相関分析器
==============

特徴量間のピアソン相関係数を計算し、相関行列を生成する機能を提供します。
高相関特徴量ペアの特定と冗長特徴量候補の抽出機能を含みます。
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns  # Optional dependency
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureCorrelationAnalyzer:
    """特徴量相関分析器

    特徴量間のピアソン相関係数を計算し、相関行列を生成する機能を実装。
    高相関特徴量ペア（|r| > 0.8）の特定と冗長特徴量候補の抽出機能を提供。
    """

    def __init__(self, feature_data: np.ndarray, feature_names: List[str]):
        """
        Args:
            feature_data: 特徴量データ (n_samples, n_features)
            feature_names: 特徴量名のリスト
        """
        self.feature_data = feature_data
        self.feature_names = feature_names
        self.correlation_matrix = None
        self.feature_categories = None

        # データの基本チェック
        self._validate_data()

        # 特徴量カテゴリを定義
        self._define_feature_categories()

        logger.info(f"FeatureCorrelationAnalyzer初期化完了: {self.feature_data.shape}")

    def _validate_data(self) -> None:
        """データの妥当性をチェック"""
        if self.feature_data.shape[1] != len(self.feature_names):
            raise ValueError(
                f"特徴量データの次元数({self.feature_data.shape[1]})と"
                f"特徴量名の数({len(self.feature_names)})が一致しません"
            )

        # NaNや無限値のチェック
        if np.any(np.isnan(self.feature_data)):
            logger.warning("特徴量データにNaNが含まれています")

        if np.any(np.isinf(self.feature_data)):
            logger.warning("特徴量データに無限値が含まれています")

        # 分散がゼロの特徴量をチェック
        zero_var_features = []
        for i, name in enumerate(self.feature_names):
            if np.var(self.feature_data[:, i]) == 0:
                zero_var_features.append(name)

        if zero_var_features:
            logger.warning(f"分散がゼロの特徴量: {zero_var_features}")

    def _define_feature_categories(self) -> None:
        """特徴量カテゴリを定義"""
        self.feature_categories = {}

        for i, name in enumerate(self.feature_names):
            if name.startswith("task_"):
                category = "タスク特徴量"
            elif name.startswith("dev_"):
                category = "開発者特徴量"
            elif name.startswith("match_"):
                category = "マッチング特徴量"
            elif name.startswith("gat_") and "emb_" not in name:
                category = "GAT統計特徴量"
            elif "gat_" in name or name.startswith("feature_"):
                category = "GAT埋め込み特徴量"
            else:
                category = "その他特徴量"

            if category not in self.feature_categories:
                self.feature_categories[category] = []
            self.feature_categories[category].append(i)

    def analyze_correlations(
        self, correlation_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """相関分析を実行

        Args:
            correlation_threshold: 高相関とみなす閾値

        Returns:
            相関分析結果を含む辞書
        """
        results = {
            "correlation_matrix": self._compute_correlation_matrix(),
            "high_correlation_pairs": self._find_high_correlations(
                correlation_threshold
            ),
            "redundant_features": self._identify_redundant_features(
                correlation_threshold
            ),
            "category_correlations": self._analyze_category_correlations(),
            "correlation_statistics": self._compute_correlation_statistics(),
        }

        logger.info("特徴量相関分析完了")
        return results

    def _compute_correlation_matrix(self) -> np.ndarray:
        """相関行列計算

        Returns:
            ピアソン相関係数行列
        """
        # NaNや無限値を処理
        clean_data = self.feature_data.copy()

        # NaNを中央値で置換
        for i in range(clean_data.shape[1]):
            col_data = clean_data[:, i]
            if np.any(np.isnan(col_data)):
                median_val = np.nanmedian(col_data)
                clean_data[np.isnan(col_data), i] = median_val

        # 無限値をクリップ
        clean_data = np.clip(clean_data, -1e10, 1e10)

        # 相関行列を計算
        self.correlation_matrix = np.corrcoef(clean_data.T)

        # 対角成分を1に設定（数値誤差対策）
        np.fill_diagonal(self.correlation_matrix, 1.0)

        # NaNを0で置換（分散がゼロの特徴量対策）
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix, nan=0.0)

        logger.info(f"相関行列計算完了: {self.correlation_matrix.shape}")
        return self.correlation_matrix

    def _find_high_correlations(
        self, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """高相関ペア特定

        Args:
            threshold: 高相関とみなす閾値

        Returns:
            (特徴量1, 特徴量2, 相関係数)のタプルリスト
        """
        if self.correlation_matrix is None:
            self._compute_correlation_matrix()

        high_corr_pairs = []
        n_features = len(self.feature_names)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_val = self.correlation_matrix[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append(
                        (self.feature_names[i], self.feature_names[j], float(corr_val))
                    )

        # 相関係数の絶対値で降順ソート
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        logger.info(f"高相関ペア({threshold}以上)を{len(high_corr_pairs)}個発見")
        return high_corr_pairs

    def _identify_redundant_features(
        self, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """冗長特徴量特定

        Args:
            threshold: 冗長とみなす相関閾値

        Returns:
            冗長特徴量の情報リスト
        """
        high_corr_pairs = self._find_high_correlations(threshold)

        # 各特徴量の高相関ペア数をカウント
        feature_corr_count = {}
        for feat1, feat2, corr in high_corr_pairs:
            feature_corr_count[feat1] = feature_corr_count.get(feat1, 0) + 1
            feature_corr_count[feat2] = feature_corr_count.get(feat2, 0) + 1

        # 冗長特徴量候補を特定
        redundant_candidates = []
        processed_pairs = set()

        for feat1, feat2, corr in high_corr_pairs:
            pair_key = tuple(sorted([feat1, feat2]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # より多くの高相関を持つ特徴量を冗長候補とする
            feat1_count = feature_corr_count.get(feat1, 0)
            feat2_count = feature_corr_count.get(feat2, 0)

            if feat1_count > feat2_count:
                redundant_feature = feat1
                keep_feature = feat2
            elif feat2_count > feat1_count:
                redundant_feature = feat2
                keep_feature = feat1
            else:
                # 同じ場合は名前順で決定
                redundant_feature = max(feat1, feat2)
                keep_feature = min(feat1, feat2)

            redundant_candidates.append(
                {
                    "redundant_feature": redundant_feature,
                    "keep_feature": keep_feature,
                    "correlation": corr,
                    "redundant_corr_count": feature_corr_count.get(
                        redundant_feature, 0
                    ),
                    "keep_corr_count": feature_corr_count.get(keep_feature, 0),
                }
            )

        # 冗長度でソート
        redundant_candidates.sort(key=lambda x: x["redundant_corr_count"], reverse=True)

        logger.info(f"冗長特徴量候補を{len(redundant_candidates)}個特定")
        return redundant_candidates

    def _analyze_category_correlations(self) -> Dict[str, Dict[str, float]]:
        """カテゴリ間相関分析

        Returns:
            カテゴリ間の平均相関係数
        """
        if self.correlation_matrix is None:
            self._compute_correlation_matrix()

        category_correlations = {}

        # カテゴリ間の組み合わせを生成
        categories = list(self.feature_categories.keys())

        for cat1, cat2 in combinations(categories, 2):
            indices1 = self.feature_categories[cat1]
            indices2 = self.feature_categories[cat2]

            # カテゴリ間の相関係数を抽出
            inter_corrs = []
            for i in indices1:
                for j in indices2:
                    corr_val = self.correlation_matrix[i, j]
                    if not np.isnan(corr_val):
                        inter_corrs.append(abs(corr_val))

            if inter_corrs:
                category_key = f"{cat1} - {cat2}"
                category_correlations[category_key] = {
                    "mean_correlation": float(np.mean(inter_corrs)),
                    "max_correlation": float(np.max(inter_corrs)),
                    "std_correlation": float(np.std(inter_corrs)),
                    "pair_count": len(inter_corrs),
                }

        # カテゴリ内相関も計算
        for category, indices in self.feature_categories.items():
            if len(indices) > 1:
                intra_corrs = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        corr_val = self.correlation_matrix[idx1, idx2]
                        if not np.isnan(corr_val):
                            intra_corrs.append(abs(corr_val))

                if intra_corrs:
                    category_key = f"{category} (内部)"
                    category_correlations[category_key] = {
                        "mean_correlation": float(np.mean(intra_corrs)),
                        "max_correlation": float(np.max(intra_corrs)),
                        "std_correlation": float(np.std(intra_corrs)),
                        "pair_count": len(intra_corrs),
                    }

        return category_correlations

    def _compute_correlation_statistics(self) -> Dict[str, float]:
        """相関統計を計算

        Returns:
            相関行列の統計情報
        """
        if self.correlation_matrix is None:
            self._compute_correlation_matrix()

        # 上三角行列から相関係数を抽出（対角成分を除く）
        n = self.correlation_matrix.shape[0]
        upper_triangle = []

        for i in range(n):
            for j in range(i + 1, n):
                corr_val = self.correlation_matrix[i, j]
                if not np.isnan(corr_val):
                    upper_triangle.append(corr_val)

        upper_triangle = np.array(upper_triangle)
        abs_correlations = np.abs(upper_triangle)

        statistics = {
            "total_pairs": len(upper_triangle),
            "mean_correlation": float(np.mean(upper_triangle)),
            "mean_abs_correlation": float(np.mean(abs_correlations)),
            "std_correlation": float(np.std(upper_triangle)),
            "max_correlation": float(np.max(upper_triangle)),
            "min_correlation": float(np.min(upper_triangle)),
            "max_abs_correlation": float(np.max(abs_correlations)),
            "high_corr_count_08": int(np.sum(abs_correlations >= 0.8)),
            "high_corr_count_06": int(np.sum(abs_correlations >= 0.6)),
            "high_corr_count_04": int(np.sum(abs_correlations >= 0.4)),
            "low_corr_count_02": int(np.sum(abs_correlations <= 0.2)),
        }

        return statistics

    def generate_correlation_report(
        self, output_path: Optional[str] = None, correlation_threshold: float = 0.8
    ) -> str:
        """相関分析レポートを生成

        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            correlation_threshold: 高相関の閾値

        Returns:
            生成されたレポートファイルのパス
        """
        analysis_results = self.analyze_correlations(correlation_threshold)

        if output_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"feature_correlation_report_{timestamp}.txt"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("特徴量相関分析レポート\n")
            f.write("=" * 60 + "\n\n")

            # 基本統計
            f.write("【基本統計】\n")
            f.write(f"特徴量数: {len(self.feature_names)}\n")
            f.write(f"データサンプル数: {self.feature_data.shape[0]}\n")
            f.write(
                f"相関ペア総数: {len(self.feature_names) * (len(self.feature_names) - 1) // 2}\n\n"
            )

            # 相関統計
            corr_stats = analysis_results["correlation_statistics"]
            f.write("【相関統計】\n")
            f.write(f"平均相関係数: {corr_stats['mean_correlation']:.6f}\n")
            f.write(f"平均絶対相関係数: {corr_stats['mean_abs_correlation']:.6f}\n")
            f.write(f"相関係数標準偏差: {corr_stats['std_correlation']:.6f}\n")
            f.write(f"最大相関係数: {corr_stats['max_correlation']:.6f}\n")
            f.write(f"最小相関係数: {corr_stats['min_correlation']:.6f}\n")
            f.write(f"高相関ペア(|r|≥0.8): {corr_stats['high_corr_count_08']}個\n")
            f.write(f"中相関ペア(|r|≥0.6): {corr_stats['high_corr_count_06']}個\n")
            f.write(f"低相関ペア(|r|≤0.2): {corr_stats['low_corr_count_02']}個\n\n")

            # 高相関ペア
            high_corr_pairs = analysis_results["high_correlation_pairs"]
            f.write(f"【高相関ペア (|r|≥{correlation_threshold})】\n")
            if high_corr_pairs:
                for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:20], 1):
                    f.write(f"{i:2d}. {feat1:30s} - {feat2:30s} | r={corr:7.4f}\n")
                if len(high_corr_pairs) > 20:
                    f.write(f"... 他 {len(high_corr_pairs) - 20} ペア\n")
            else:
                f.write("高相関ペアは見つかりませんでした。\n")
            f.write("\n")

            # 冗長特徴量
            redundant_features = analysis_results["redundant_features"]
            f.write("【冗長特徴量候補】\n")
            if redundant_features:
                for i, item in enumerate(redundant_features[:15], 1):
                    f.write(
                        f"{i:2d}. 削除候補: {item['redundant_feature']:30s} | "
                        f"保持: {item['keep_feature']:30s} | "
                        f"相関: {item['correlation']:7.4f}\n"
                    )
                if len(redundant_features) > 15:
                    f.write(f"... 他 {len(redundant_features) - 15} 個\n")
            else:
                f.write("冗長特徴量候補は見つかりませんでした。\n")
            f.write("\n")

            # カテゴリ間相関
            f.write("【カテゴリ間相関分析】\n")
            category_corrs = analysis_results["category_correlations"]
            for category_pair, stats in category_corrs.items():
                f.write(f"{category_pair}:\n")
                f.write(f"  平均相関: {stats['mean_correlation']:.4f}\n")
                f.write(f"  最大相関: {stats['max_correlation']:.4f}\n")
                f.write(f"  ペア数: {stats['pair_count']}\n\n")

        logger.info(f"相関分析レポート生成完了: {output_path}")
        return str(output_path)

    def visualize_correlations(
        self,
        output_dir: Optional[str] = None,
        show_plot: bool = True,
        max_features: int = 50,
    ) -> str:
        """相関分析の可視化

        Args:
            output_dir: 出力ディレクトリ（Noneの場合は自動生成）
            show_plot: プロットを表示するかどうか
            max_features: 相関行列で表示する最大特徴量数

        Returns:
            生成された図のファイルパス
        """
        if self.correlation_matrix is None:
            self._compute_correlation_matrix()

        analysis_results = self.analyze_correlations()

        if output_dir is None:
            output_dir = Path("outputs")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 図を作成
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle("特徴量相関分析", fontsize=16, fontweight="bold")

        # 1. 相関行列ヒートマップ（最初のmax_features個）
        n_show = min(max_features, len(self.feature_names))
        corr_subset = self.correlation_matrix[:n_show, :n_show]
        feature_names_subset = [name[:15] for name in self.feature_names[:n_show]]

        im = axes[0, 0].imshow(corr_subset, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0, 0].set_xticks(range(n_show))
        axes[0, 0].set_yticks(range(n_show))
        axes[0, 0].set_xticklabels(feature_names_subset, rotation=90, fontsize=6)
        axes[0, 0].set_yticklabels(feature_names_subset, fontsize=6)
        axes[0, 0].set_title(f"相関行列ヒートマップ (最初の{n_show}特徴量)")
        plt.colorbar(im, ax=axes[0, 0])

        # 2. 相関係数分布ヒストグラム
        upper_triangle = []
        n = self.correlation_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                corr_val = self.correlation_matrix[i, j]
                if not np.isnan(corr_val):
                    upper_triangle.append(corr_val)

        axes[0, 1].hist(
            upper_triangle, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 1].set_xlabel("相関係数")
        axes[0, 1].set_ylabel("頻度")
        axes[0, 1].set_title("相関係数分布")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 高相関ペア可視化
        high_corr_pairs = analysis_results["high_correlation_pairs"][:15]
        if high_corr_pairs:
            pair_labels = [f"{pair[0][:10]}-{pair[1][:10]}" for pair in high_corr_pairs]
            correlations = [pair[2] for pair in high_corr_pairs]
            colors = ["red" if c < 0 else "green" for c in correlations]

            y_pos = np.arange(len(pair_labels))
            axes[0, 2].barh(y_pos, correlations, color=colors, alpha=0.7)
            axes[0, 2].set_yticks(y_pos)
            axes[0, 2].set_yticklabels(pair_labels, fontsize=8)
            axes[0, 2].set_xlabel("相関係数")
            axes[0, 2].set_title("高相関ペア TOP15")
            axes[0, 2].grid(True, alpha=0.3)

        # 4. 絶対相関係数分布
        abs_correlations = [abs(corr) for corr in upper_triangle]
        axes[1, 0].hist(
            abs_correlations, bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[1, 0].set_xlabel("絶対相関係数")
        axes[1, 0].set_ylabel("頻度")
        axes[1, 0].set_title("絶対相関係数分布")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. カテゴリ間平均相関
        category_corrs = analysis_results["category_correlations"]
        if category_corrs:
            categories = list(category_corrs.keys())[:10]  # 最初の10個
            mean_corrs = [category_corrs[cat]["mean_correlation"] for cat in categories]

            axes[1, 1].bar(
                range(len(categories)), mean_corrs, alpha=0.7, color="lightgreen"
            )
            axes[1, 1].set_xticks(range(len(categories)))
            axes[1, 1].set_xticklabels(
                [cat[:15] for cat in categories], rotation=45, ha="right", fontsize=8
            )
            axes[1, 1].set_ylabel("平均相関係数")
            axes[1, 1].set_title("カテゴリ間平均相関")
            axes[1, 1].grid(True, alpha=0.3)

        # 6. 相関統計サマリー
        corr_stats = analysis_results["correlation_statistics"]
        stats_labels = ["高相関\n(≥0.8)", "中相関\n(≥0.6)", "低相関\n(≤0.2)"]
        stats_values = [
            corr_stats["high_corr_count_08"],
            corr_stats["high_corr_count_06"],
            corr_stats["low_corr_count_02"],
        ]

        axes[1, 2].bar(
            stats_labels, stats_values, alpha=0.7, color=["red", "orange", "blue"]
        )
        axes[1, 2].set_ylabel("ペア数")
        axes[1, 2].set_title("相関レベル別ペア数")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_dir / f"feature_correlation_analysis_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        logger.info(f"相関分析可視化完了: {fig_path}")
        return str(fig_path)

    def get_correlation_matrix(self) -> np.ndarray:
        """相関行列を取得

        Returns:
            相関行列
        """
        if self.correlation_matrix is None:
            self._compute_correlation_matrix()
        return self.correlation_matrix

    def get_feature_correlations(
        self, feature_name: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """指定特徴量との相関が高い特徴量を取得

        Args:
            feature_name: 対象特徴量名
            top_k: 上位何個を取得するか

        Returns:
            (特徴量名, 相関係数)のタプルリスト
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"特徴量 '{feature_name}' が見つかりません")

        if self.correlation_matrix is None:
            self._compute_correlation_matrix()

        feature_idx = self.feature_names.index(feature_name)
        correlations = self.correlation_matrix[feature_idx, :]

        # 自分自身を除いて相関の絶対値でソート
        corr_pairs = []
        for i, corr in enumerate(correlations):
            if i != feature_idx and not np.isnan(corr):
                corr_pairs.append((self.feature_names[i], float(corr)))

        corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        return corr_pairs[:top_k]
