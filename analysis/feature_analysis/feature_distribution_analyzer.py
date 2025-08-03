"""
特徴量分布分析器
==============

各特徴量の分布統計（平均、分散、歪度、尖度）を計算する機能を提供します。
正規性検定、外れ値検出、スケール不均衡問題の特定機能を含みます。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns  # Optional dependency
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureDistributionAnalyzer:
    """特徴量分布分析器

    各特徴量の分布統計（平均、分散、歪度、尖度）を計算する機能を実装。
    正規性検定、外れ値検出、スケール不均衡問題の特定機能を提供。
    """

    def __init__(self, feature_data: np.ndarray, feature_names: List[str]):
        """
        Args:
            feature_data: 特徴量データ (n_samples, n_features)
            feature_names: 特徴量名のリスト
        """
        self.feature_data = feature_data
        self.feature_names = feature_names
        self.feature_categories = None

        # データの基本チェック
        self._validate_data()

        # 特徴量カテゴリを定義
        self._define_feature_categories()

        logger.info(f"FeatureDistributionAnalyzer初期化完了: {self.feature_data.shape}")

    def _validate_data(self) -> None:
        """データの妥当性をチェック"""
        if self.feature_data.shape[1] != len(self.feature_names):
            raise ValueError(
                f"特徴量データの次元数({self.feature_data.shape[1]})と"
                f"特徴量名の数({len(self.feature_names)})が一致しません"
            )

        # NaNや無限値のチェック
        nan_count = np.sum(np.isnan(self.feature_data))
        inf_count = np.sum(np.isinf(self.feature_data))

        if nan_count > 0:
            logger.warning(f"特徴量データにNaNが{nan_count}個含まれています")

        if inf_count > 0:
            logger.warning(f"特徴量データに無限値が{inf_count}個含まれています")

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

    def analyze_distributions(self) -> Dict[str, Any]:
        """分布分析を実行

        Returns:
            分布分析結果を含む辞書
        """
        results = {
            "distribution_stats": self._compute_distribution_stats(),
            "normality_tests": self._test_normality(),
            "outlier_detection": self._detect_outliers(),
            "scale_imbalance": self._detect_scale_imbalance(),
            "category_distributions": self._analyze_category_distributions(),
            "data_quality_issues": self._identify_data_quality_issues(),
        }

        logger.info("特徴量分布分析完了")
        return results

    def _compute_distribution_stats(self) -> Dict[str, Dict[str, float]]:
        """分布統計計算

        Returns:
            各特徴量の分布統計
        """
        distribution_stats = {}

        for i, name in enumerate(self.feature_names):
            values = self.feature_data[:, i]

            # NaNを除外
            clean_values = values[~np.isnan(values)]

            if len(clean_values) == 0:
                stats_dict = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "var": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "skewness": np.nan,
                    "kurtosis": np.nan,
                    "range": np.nan,
                    "iqr": np.nan,
                    "cv": np.nan,  # 変動係数
                    "zero_ratio": np.nan,
                    "nan_ratio": 1.0,
                }
            else:
                q25, q75 = np.percentile(clean_values, [25, 75])
                mean_val = np.mean(clean_values)
                std_val = np.std(clean_values)

                stats_dict = {
                    "count": len(clean_values),
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "var": float(np.var(clean_values)),
                    "min": float(np.min(clean_values)),
                    "max": float(np.max(clean_values)),
                    "median": float(np.median(clean_values)),
                    "q25": float(q25),
                    "q75": float(q75),
                    "skewness": float(stats.skew(clean_values)),
                    "kurtosis": float(stats.kurtosis(clean_values)),
                    "range": float(np.max(clean_values) - np.min(clean_values)),
                    "iqr": float(q75 - q25),
                    "cv": float(std_val / abs(mean_val)) if mean_val != 0 else np.inf,
                    "zero_ratio": float(np.sum(clean_values == 0) / len(clean_values)),
                    "nan_ratio": float(np.sum(np.isnan(values)) / len(values)),
                }

            distribution_stats[name] = stats_dict

        return distribution_stats

    def _test_normality(self) -> Dict[str, Dict[str, float]]:
        """正規性検定

        Returns:
            各特徴量の正規性検定結果
        """
        normality_results = {}

        for i, name in enumerate(self.feature_names):
            values = self.feature_data[:, i]
            clean_values = values[~np.isnan(values)]

            if len(clean_values) < 3:
                # サンプル数が少なすぎる場合
                normality_results[name] = {
                    "shapiro_statistic": np.nan,
                    "shapiro_p_value": np.nan,
                    "is_normal_shapiro": False,
                    "ks_statistic": np.nan,
                    "ks_p_value": np.nan,
                    "is_normal_ks": False,
                    "sample_size": len(clean_values),
                }
                continue

            try:
                # Shapiro-Wilk検定（サンプル数5000以下）
                if len(clean_values) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(clean_values)
                else:
                    # サンプル数が多い場合はサブサンプリング
                    sample_indices = np.random.choice(
                        len(clean_values), 5000, replace=False
                    )
                    shapiro_stat, shapiro_p = stats.shapiro(
                        clean_values[sample_indices]
                    )

                # Kolmogorov-Smirnov検定
                ks_stat, ks_p = stats.kstest(
                    clean_values,
                    "norm",
                    args=(np.mean(clean_values), np.std(clean_values)),
                )

                normality_results[name] = {
                    "shapiro_statistic": float(shapiro_stat),
                    "shapiro_p_value": float(shapiro_p),
                    "is_normal_shapiro": shapiro_p > 0.05,
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_p),
                    "is_normal_ks": ks_p > 0.05,
                    "sample_size": len(clean_values),
                }

            except Exception as e:
                logger.warning(f"正規性検定でエラー ({name}): {e}")
                normality_results[name] = {
                    "shapiro_statistic": np.nan,
                    "shapiro_p_value": np.nan,
                    "is_normal_shapiro": False,
                    "ks_statistic": np.nan,
                    "ks_p_value": np.nan,
                    "is_normal_ks": False,
                    "sample_size": len(clean_values),
                    "error": str(e),
                }

        return normality_results

    def _detect_outliers(self) -> Dict[str, Dict[str, Any]]:
        """外れ値検出

        Returns:
            各特徴量の外れ値情報
        """
        outlier_results = {}

        for i, name in enumerate(self.feature_names):
            values = self.feature_data[:, i]
            clean_values = values[~np.isnan(values)]

            if len(clean_values) == 0:
                outlier_results[name] = {
                    "iqr_outliers": [],
                    "zscore_outliers": [],
                    "iqr_outlier_count": 0,
                    "zscore_outlier_count": 0,
                    "iqr_outlier_ratio": 0.0,
                    "zscore_outlier_ratio": 0.0,
                }
                continue

            # IQR法による外れ値検出
            q25, q75 = np.percentile(clean_values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            iqr_outlier_mask = (clean_values < lower_bound) | (
                clean_values > upper_bound
            )
            iqr_outliers = np.where(iqr_outlier_mask)[0].tolist()

            # Z-score法による外れ値検出（|z| > 3）
            if np.std(clean_values) > 0:
                z_scores = np.abs(stats.zscore(clean_values))
                zscore_outlier_mask = z_scores > 3
                zscore_outliers = np.where(zscore_outlier_mask)[0].tolist()
            else:
                zscore_outliers = []

            outlier_results[name] = {
                "iqr_outliers": iqr_outliers,
                "zscore_outliers": zscore_outliers,
                "iqr_outlier_count": len(iqr_outliers),
                "zscore_outlier_count": len(zscore_outliers),
                "iqr_outlier_ratio": len(iqr_outliers) / len(clean_values),
                "zscore_outlier_ratio": len(zscore_outliers) / len(clean_values),
                "iqr_bounds": [float(lower_bound), float(upper_bound)],
                "zscore_threshold": 3.0,
            }

        return outlier_results

    def _detect_scale_imbalance(self) -> Dict[str, float]:
        """スケール不均衡検出

        Returns:
            特徴量間のスケール不均衡情報
        """
        scale_stats = {}

        # 各特徴量の標準偏差を計算
        feature_stds = []
        feature_ranges = []
        feature_means = []

        for i, name in enumerate(self.feature_names):
            values = self.feature_data[:, i]
            clean_values = values[~np.isnan(values)]

            if len(clean_values) > 0:
                std_val = np.std(clean_values)
                range_val = np.max(clean_values) - np.min(clean_values)
                mean_val = np.mean(clean_values)

                feature_stds.append(std_val)
                feature_ranges.append(range_val)
                feature_means.append(abs(mean_val))
            else:
                feature_stds.append(0)
                feature_ranges.append(0)
                feature_means.append(0)

        feature_stds = np.array(feature_stds)
        feature_ranges = np.array(feature_ranges)
        feature_means = np.array(feature_means)

        # スケール不均衡の指標を計算
        scale_stats = {
            "std_ratio_max_min": (
                float(np.max(feature_stds) / np.min(feature_stds[feature_stds > 0]))
                if np.any(feature_stds > 0)
                else 1.0
            ),
            "range_ratio_max_min": (
                float(
                    np.max(feature_ranges) / np.min(feature_ranges[feature_ranges > 0])
                )
                if np.any(feature_ranges > 0)
                else 1.0
            ),
            "mean_ratio_max_min": (
                float(np.max(feature_means) / np.min(feature_means[feature_means > 0]))
                if np.any(feature_means > 0)
                else 1.0
            ),
            "std_cv": (
                float(np.std(feature_stds) / np.mean(feature_stds))
                if np.mean(feature_stds) > 0
                else 0.0
            ),
            "range_cv": (
                float(np.std(feature_ranges) / np.mean(feature_ranges))
                if np.mean(feature_ranges) > 0
                else 0.0
            ),
            "features_with_zero_std": int(np.sum(feature_stds == 0)),
            "features_with_large_std": int(
                np.sum(feature_stds > np.mean(feature_stds) + 3 * np.std(feature_stds))
            ),
            "scale_imbalance_severe": (
                float(np.max(feature_stds) / np.min(feature_stds[feature_stds > 0]))
                > 1000
                if np.any(feature_stds > 0)
                else False
            ),
        }

        return scale_stats

    def _analyze_category_distributions(self) -> Dict[str, Dict[str, float]]:
        """カテゴリ別分布分析

        Returns:
            カテゴリ別の分布統計
        """
        category_stats = {}

        for category, indices in self.feature_categories.items():
            if not indices:
                continue

            category_data = self.feature_data[:, indices]

            # カテゴリ内の統計を計算
            means = []
            stds = []
            skewnesses = []
            kurtoses = []
            zero_ratios = []

            for i in range(category_data.shape[1]):
                values = category_data[:, i]
                clean_values = values[~np.isnan(values)]

                if len(clean_values) > 0:
                    means.append(np.mean(clean_values))
                    stds.append(np.std(clean_values))
                    skewnesses.append(stats.skew(clean_values))
                    kurtoses.append(stats.kurtosis(clean_values))
                    zero_ratios.append(np.sum(clean_values == 0) / len(clean_values))

            if means:
                category_stats[category] = {
                    "feature_count": len(indices),
                    "mean_of_means": float(np.mean(means)),
                    "mean_of_stds": float(np.mean(stds)),
                    "mean_skewness": float(np.mean(skewnesses)),
                    "mean_kurtosis": float(np.mean(kurtoses)),
                    "mean_zero_ratio": float(np.mean(zero_ratios)),
                    "std_of_means": float(np.std(means)),
                    "std_of_stds": float(np.std(stds)),
                    "max_std": float(np.max(stds)),
                    "min_std": float(np.min(stds)),
                    "scale_ratio": (
                        float(np.max(stds) / np.min(stds))
                        if np.min(stds) > 0
                        else np.inf
                    ),
                }

        return category_stats

    def _identify_data_quality_issues(self) -> Dict[str, List[str]]:
        """データ品質問題の特定

        Returns:
            データ品質問題のある特徴量リスト
        """
        issues = {
            "high_nan_ratio": [],  # NaN比率が高い（>10%）
            "zero_variance": [],  # 分散がゼロ
            "extreme_skewness": [],  # 極端な歪み（|skew| > 3）
            "extreme_kurtosis": [],  # 極端な尖度（|kurt| > 10）
            "high_outlier_ratio": [],  # 外れ値比率が高い（>5%）
            "constant_values": [],  # 定数値
            "infinite_values": [],  # 無限値を含む
            "single_value_dominant": [],  # 単一値が支配的（>90%）
        }

        distribution_stats = self._compute_distribution_stats()
        outlier_results = self._detect_outliers()

        for name, stats_dict in distribution_stats.items():
            # NaN比率チェック
            if stats_dict["nan_ratio"] > 0.1:
                issues["high_nan_ratio"].append(name)

            # 分散ゼロチェック
            if stats_dict["var"] == 0:
                issues["zero_variance"].append(name)

            # 極端な歪みチェック
            if abs(stats_dict["skewness"]) > 3:
                issues["extreme_skewness"].append(name)

            # 極端な尖度チェック
            if abs(stats_dict["kurtosis"]) > 10:
                issues["extreme_kurtosis"].append(name)

            # 外れ値比率チェック
            if name in outlier_results:
                if outlier_results[name]["iqr_outlier_ratio"] > 0.05:
                    issues["high_outlier_ratio"].append(name)

            # 定数値チェック
            if stats_dict["min"] == stats_dict["max"] and stats_dict["count"] > 1:
                issues["constant_values"].append(name)

            # 単一値支配チェック
            if stats_dict["zero_ratio"] > 0.9:
                issues["single_value_dominant"].append(name)

        # 無限値チェック
        for i, name in enumerate(self.feature_names):
            values = self.feature_data[:, i]
            if np.any(np.isinf(values)):
                issues["infinite_values"].append(name)

        return issues

    def generate_distribution_report(self, output_path: Optional[str] = None) -> str:
        """分布分析レポートを生成

        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）

        Returns:
            生成されたレポートファイルのパス
        """
        analysis_results = self.analyze_distributions()

        if output_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"feature_distribution_report_{timestamp}.txt"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("特徴量分布分析レポート\n")
            f.write("=" * 60 + "\n\n")

            # 基本統計
            f.write("【基本統計】\n")
            f.write(f"特徴量数: {len(self.feature_names)}\n")
            f.write(f"データサンプル数: {self.feature_data.shape[0]}\n\n")

            # スケール不均衡
            scale_imbalance = analysis_results["scale_imbalance"]
            f.write("【スケール不均衡分析】\n")
            f.write(
                f"標準偏差比（最大/最小）: {scale_imbalance['std_ratio_max_min']:.2f}\n"
            )
            f.write(
                f"レンジ比（最大/最小）: {scale_imbalance['range_ratio_max_min']:.2f}\n"
            )
            f.write(
                f"平均値比（最大/最小）: {scale_imbalance['mean_ratio_max_min']:.2f}\n"
            )
            f.write(
                f"分散ゼロの特徴量数: {scale_imbalance['features_with_zero_std']}\n"
            )
            f.write(
                f"深刻なスケール不均衡: {'はい' if scale_imbalance['scale_imbalance_severe'] else 'いいえ'}\n\n"
            )

            # データ品質問題
            quality_issues = analysis_results["data_quality_issues"]
            f.write("【データ品質問題】\n")
            for issue_type, features in quality_issues.items():
                if features:
                    f.write(f"{issue_type}: {len(features)}個\n")
                    for feat in features[:5]:  # 最初の5個だけ表示
                        f.write(f"  - {feat}\n")
                    if len(features) > 5:
                        f.write(f"  ... 他 {len(features) - 5} 個\n")
            f.write("\n")

            # カテゴリ別分布統計
            f.write("【カテゴリ別分布統計】\n")
            category_distributions = analysis_results["category_distributions"]
            for category, stats in category_distributions.items():
                f.write(f"\n{category}:\n")
                f.write(f"  特徴量数: {stats['feature_count']}\n")
                f.write(f"  平均の平均: {stats['mean_of_means']:.4f}\n")
                f.write(f"  標準偏差の平均: {stats['mean_of_stds']:.4f}\n")
                f.write(f"  平均歪度: {stats['mean_skewness']:.4f}\n")
                f.write(f"  平均尖度: {stats['mean_kurtosis']:.4f}\n")
                f.write(f"  スケール比: {stats['scale_ratio']:.2f}\n")

            # 正規性検定サマリー
            f.write("\n【正規性検定サマリー】\n")
            normality_tests = analysis_results["normality_tests"]
            normal_count_shapiro = sum(
                1
                for test in normality_tests.values()
                if test.get("is_normal_shapiro", False)
            )
            normal_count_ks = sum(
                1
                for test in normality_tests.values()
                if test.get("is_normal_ks", False)
            )

            f.write(
                f"Shapiro-Wilk検定で正規分布: {normal_count_shapiro}/{len(normality_tests)}\n"
            )
            f.write(
                f"Kolmogorov-Smirnov検定で正規分布: {normal_count_ks}/{len(normality_tests)}\n"
            )

            # 外れ値サマリー
            f.write("\n【外れ値検出サマリー】\n")
            outlier_detection = analysis_results["outlier_detection"]
            high_outlier_features = [
                name
                for name, result in outlier_detection.items()
                if result["iqr_outlier_ratio"] > 0.05
            ]

            f.write(f"外れ値比率が高い特徴量（>5%）: {len(high_outlier_features)}個\n")
            for feat in high_outlier_features[:10]:
                ratio = outlier_detection[feat]["iqr_outlier_ratio"]
                f.write(f"  - {feat}: {ratio:.2%}\n")

            # 詳細統計（上位20特徴量）
            f.write("\n【詳細統計（特徴量別）】\n")
            distribution_stats = analysis_results["distribution_stats"]

            # 歪度の絶対値でソート
            sorted_features = sorted(
                distribution_stats.items(),
                key=lambda x: (
                    abs(x[1]["skewness"]) if not np.isnan(x[1]["skewness"]) else 0
                ),
                reverse=True,
            )

            f.write("歪度が大きい特徴量 TOP10:\n")
            for i, (name, stats) in enumerate(sorted_features[:10], 1):
                f.write(
                    f"{i:2d}. {name:30s} | 歪度:{stats['skewness']:7.3f} | 尖度:{stats['kurtosis']:7.3f} | CV:{stats['cv']:7.3f}\n"
                )

        logger.info(f"分布分析レポート生成完了: {output_path}")
        return str(output_path)

    def visualize_distributions(
        self,
        output_dir: Optional[str] = None,
        show_plot: bool = True,
        max_features: int = 20,
    ) -> str:
        """分布分析の可視化

        Args:
            output_dir: 出力ディレクトリ（Noneの場合は自動生成）
            show_plot: プロットを表示するかどうか
            max_features: 個別分布で表示する最大特徴量数

        Returns:
            生成された図のファイルパス
        """
        analysis_results = self.analyze_distributions()

        if output_dir is None:
            output_dir = Path("outputs")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 図1: 分布統計サマリー
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("特徴量分布分析サマリー", fontsize=16, fontweight="bold")

        distribution_stats = analysis_results["distribution_stats"]

        # 1. 歪度分布
        skewnesses = [
            stats["skewness"]
            for stats in distribution_stats.values()
            if not np.isnan(stats["skewness"])
        ]
        axes[0, 0].hist(
            skewnesses, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_xlabel("歪度")
        axes[0, 0].set_ylabel("特徴量数")
        axes[0, 0].set_title("歪度分布")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 尖度分布
        kurtoses = [
            stats["kurtosis"]
            for stats in distribution_stats.values()
            if not np.isnan(stats["kurtosis"])
        ]
        axes[0, 1].hist(
            kurtoses, bins=30, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].set_xlabel("尖度")
        axes[0, 1].set_ylabel("特徴量数")
        axes[0, 1].set_title("尖度分布")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 変動係数分布
        cvs = [
            stats["cv"]
            for stats in distribution_stats.values()
            if not np.isnan(stats["cv"]) and not np.isinf(stats["cv"])
        ]
        if cvs:
            # 極端な値をクリップ
            cvs_clipped = np.clip(cvs, 0, np.percentile(cvs, 95))
            axes[0, 2].hist(
                cvs_clipped, bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
            )
        axes[0, 2].set_xlabel("変動係数")
        axes[0, 2].set_ylabel("特徴量数")
        axes[0, 2].set_title("変動係数分布")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. ゼロ比率分布
        zero_ratios = [
            stats["zero_ratio"]
            for stats in distribution_stats.values()
            if not np.isnan(stats["zero_ratio"])
        ]
        axes[1, 0].hist(
            zero_ratios, bins=30, alpha=0.7, color="orange", edgecolor="black"
        )
        axes[1, 0].set_xlabel("ゼロ値比率")
        axes[1, 0].set_ylabel("特徴量数")
        axes[1, 0].set_title("ゼロ値比率分布")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. カテゴリ別平均歪度
        category_distributions = analysis_results["category_distributions"]
        if category_distributions:
            categories = list(category_distributions.keys())
            mean_skewnesses = [
                category_distributions[cat]["mean_skewness"] for cat in categories
            ]

            axes[1, 1].bar(
                range(len(categories)), mean_skewnesses, alpha=0.7, color="purple"
            )
            axes[1, 1].set_xticks(range(len(categories)))
            axes[1, 1].set_xticklabels(
                [cat[:10] for cat in categories], rotation=45, ha="right", fontsize=8
            )
            axes[1, 1].set_ylabel("平均歪度")
            axes[1, 1].set_title("カテゴリ別平均歪度")
            axes[1, 1].grid(True, alpha=0.3)

        # 6. データ品質問題サマリー
        quality_issues = analysis_results["data_quality_issues"]
        issue_counts = [len(features) for features in quality_issues.values()]
        issue_labels = [label.replace("_", "\n") for label in quality_issues.keys()]

        axes[1, 2].bar(range(len(issue_labels)), issue_counts, alpha=0.7, color="red")
        axes[1, 2].set_xticks(range(len(issue_labels)))
        axes[1, 2].set_xticklabels(issue_labels, rotation=45, ha="right", fontsize=8)
        axes[1, 2].set_ylabel("問題のある特徴量数")
        axes[1, 2].set_title("データ品質問題")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_dir / f"feature_distribution_summary_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

        # 図2: 個別特徴量分布（上位max_features個）
        if max_features > 0:
            # 歪度の絶対値でソート
            sorted_features = sorted(
                distribution_stats.items(),
                key=lambda x: (
                    abs(x[1]["skewness"]) if not np.isnan(x[1]["skewness"]) else 0
                ),
                reverse=True,
            )

            n_show = min(max_features, len(sorted_features))
            n_cols = 4
            n_rows = (n_show + n_cols - 1) // n_cols

            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            fig2.suptitle(
                f"個別特徴量分布 (歪度上位{n_show}個)", fontsize=16, fontweight="bold"
            )

            if n_rows == 1:
                axes2 = axes2.reshape(1, -1)

            for i, (name, stats) in enumerate(sorted_features[:n_show]):
                row, col = i // n_cols, i % n_cols

                feature_idx = self.feature_names.index(name)
                values = self.feature_data[:, feature_idx]
                clean_values = values[~np.isnan(values)]

                if len(clean_values) > 0:
                    axes2[row, col].hist(
                        clean_values, bins=30, alpha=0.7, edgecolor="black"
                    )
                    axes2[row, col].set_title(
                        f'{name[:20]}\n歪度:{stats["skewness"]:.2f}', fontsize=10
                    )
                    axes2[row, col].grid(True, alpha=0.3)
                else:
                    axes2[row, col].text(
                        0.5,
                        0.5,
                        "No Data",
                        ha="center",
                        va="center",
                        transform=axes2[row, col].transAxes,
                    )
                    axes2[row, col].set_title(name[:20], fontsize=10)

            # 余った軸を非表示
            for i in range(n_show, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes2[row, col].set_visible(False)

            plt.tight_layout()

            fig2_path = output_dir / f"feature_individual_distributions_{timestamp}.png"
            plt.savefig(fig2_path, dpi=300, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

        logger.info(f"分布分析可視化完了: {fig_path}")
        return str(fig_path)
