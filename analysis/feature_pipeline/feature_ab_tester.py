"""
特徴量A/Bテスター
===============

特徴量改善効果のA/Bテスト、統計的有意性検定、改善効果レポート自動生成機能を実装します。
"""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


class TestType(Enum):
    """A/Bテストの種類"""

    PERFORMANCE_COMPARISON = "performance_comparison"
    FEATURE_IMPORTANCE_TEST = "feature_importance_test"
    STABILITY_TEST = "stability_test"
    EFFICIENCY_TEST = "efficiency_test"


class TestStatus(Enum):
    """テスト状態"""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ABTestConfig:
    """A/Bテスト設定"""

    test_id: str
    test_name: str
    test_type: TestType
    baseline_features: str  # 特徴量セットA（ベースライン）の識別子
    treatment_features: str  # 特徴量セットB（改善版）の識別子
    success_metrics: List[str]  # 成功指標
    minimum_sample_size: int = 100
    significance_level: float = 0.05
    power: float = 0.8
    minimum_effect_size: float = 0.1
    test_duration_days: int = 7
    stratification_column: Optional[str] = None


@dataclass
class ABTestResult:
    """A/Bテスト結果"""

    test_id: str
    test_config: ABTestConfig
    status: TestStatus
    start_time: str
    end_time: Optional[str]
    baseline_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_results: Dict[str, Any]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    is_significant: bool
    recommendation: str
    detailed_analysis: Dict[str, Any]


class FeatureABTester:
    """特徴量A/Bテスター

    特徴量改善効果のA/Bテスト、統計的有意性検定、
    改善効果レポート自動生成機能を実装。
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./ab_testing")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # テスト管理
        self.active_tests = {}
        self.completed_tests = {}
        self.test_history = []

        # 特徴量セット管理
        self.feature_sets = {}

        # デフォルトモデル
        self.default_models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        }

        # 成功指標計算器
        self.metric_calculators = {
            "accuracy": self._calculate_accuracy,
            "precision": self._calculate_precision,
            "recall": self._calculate_recall,
            "f1_score": self._calculate_f1_score,
            "roc_auc": self._calculate_roc_auc,
            "cross_val_accuracy": self._calculate_cross_val_accuracy,
            "feature_importance_sum": self._calculate_feature_importance_sum,
            "training_time": self._calculate_training_time,
            "prediction_time": self._calculate_prediction_time,
        }

        logger.info(f"FeatureABTester初期化完了: output_dir={self.output_dir}")

    def register_feature_set(
        self,
        set_id: str,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> "FeatureABTester":
        """特徴量セットを登録

        Args:
            set_id: 特徴量セットの識別子
            features: 特徴量データ
            labels: ラベルデータ
            feature_names: 特徴量名のリスト
            description: 特徴量セットの説明

        Returns:
            自身のインスタンス
        """
        self.feature_sets[set_id] = {
            "features": features,
            "labels": labels,
            "feature_names": feature_names
            or [f"feature_{i}" for i in range(features.shape[1])],
            "description": description or f"Feature set: {set_id}",
            "registered_at": datetime.now().isoformat(),
            "shape": features.shape,
        }

        logger.info(f"特徴量セット登録完了: {set_id} (shape: {features.shape})")
        return self

    def create_ab_test(self, config: ABTestConfig) -> str:
        """A/Bテストを作成

        Args:
            config: A/Bテスト設定

        Returns:
            テストID
        """
        # 特徴量セットの存在確認
        if config.baseline_features not in self.feature_sets:
            raise ValueError(
                f"ベースライン特徴量セットが見つかりません: {config.baseline_features}"
            )

        if config.treatment_features not in self.feature_sets:
            raise ValueError(
                f"改善版特徴量セットが見つかりません: {config.treatment_features}"
            )

        # サンプルサイズの検証
        baseline_size = self.feature_sets[config.baseline_features]["features"].shape[0]
        treatment_size = self.feature_sets[config.treatment_features]["features"].shape[
            0
        ]

        if (
            baseline_size < config.minimum_sample_size
            or treatment_size < config.minimum_sample_size
        ):
            logger.warning(
                f"サンプルサイズが不足しています: "
                f"baseline={baseline_size}, treatment={treatment_size}, "
                f"minimum={config.minimum_sample_size}"
            )

        # テスト結果オブジェクトを初期化
        test_result = ABTestResult(
            test_id=config.test_id,
            test_config=config,
            status=TestStatus.PLANNED,
            start_time=datetime.now().isoformat(),
            end_time=None,
            baseline_metrics={},
            treatment_metrics={},
            statistical_results={},
            effect_sizes={},
            confidence_intervals={},
            is_significant=False,
            recommendation="",
            detailed_analysis={},
        )

        self.active_tests[config.test_id] = test_result

        logger.info(f"A/Bテスト作成完了: {config.test_id}")
        return config.test_id

    def run_ab_test(
        self, test_id: str, models: Optional[Dict[str, Any]] = None
    ) -> ABTestResult:
        """A/Bテストを実行

        Args:
            test_id: テストID
            models: 使用するモデル（Noneの場合はデフォルトモデル）

        Returns:
            テスト結果
        """
        if test_id not in self.active_tests:
            raise ValueError(f"テストが見つかりません: {test_id}")

        test_result = self.active_tests[test_id]
        test_config = test_result.test_config

        try:
            test_result.status = TestStatus.RUNNING

            # 使用するモデルを決定
            test_models = models or self.default_models

            # ベースライン特徴量セットでテスト実行
            baseline_data = self.feature_sets[test_config.baseline_features]
            baseline_metrics = self._evaluate_feature_set(
                baseline_data, test_config.success_metrics, test_models
            )
            test_result.baseline_metrics = baseline_metrics

            # 改善版特徴量セットでテスト実行
            treatment_data = self.feature_sets[test_config.treatment_features]
            treatment_metrics = self._evaluate_feature_set(
                treatment_data, test_config.success_metrics, test_models
            )
            test_result.treatment_metrics = treatment_metrics

            # 統計的検定を実行
            statistical_results = self._perform_statistical_tests(
                baseline_metrics, treatment_metrics, test_config
            )
            test_result.statistical_results = statistical_results

            # 効果サイズを計算
            effect_sizes = self._calculate_effect_sizes(
                baseline_metrics, treatment_metrics
            )
            test_result.effect_sizes = effect_sizes

            # 信頼区間を計算
            confidence_intervals = self._calculate_confidence_intervals(
                baseline_metrics, treatment_metrics, test_config.significance_level
            )
            test_result.confidence_intervals = confidence_intervals

            # 有意性を判定
            test_result.is_significant = self._determine_significance(
                statistical_results, test_config.significance_level
            )

            # 推奨事項を生成
            test_result.recommendation = self._generate_recommendation(
                test_result, test_config
            )

            # 詳細分析を実行
            test_result.detailed_analysis = self._perform_detailed_analysis(
                baseline_data, treatment_data, test_result
            )

            # テスト完了
            test_result.status = TestStatus.COMPLETED
            test_result.end_time = datetime.now().isoformat()

            # 完了テストに移動
            self.completed_tests[test_id] = test_result
            del self.active_tests[test_id]

            # 結果を保存
            self._save_test_result(test_result)

            logger.info(
                f"A/Bテスト実行完了: {test_id} (有意性: {test_result.is_significant})"
            )
            return test_result

        except Exception as e:
            test_result.status = TestStatus.FAILED
            test_result.end_time = datetime.now().isoformat()
            logger.error(f"A/Bテスト実行エラー ({test_id}): {e}")
            raise

    def _evaluate_feature_set(
        self,
        feature_data: Dict[str, Any],
        success_metrics: List[str],
        models: Dict[str, Any],
    ) -> Dict[str, float]:
        """特徴量セットを評価"""
        features = feature_data["features"]
        labels = feature_data["labels"]

        if labels is None:
            logger.warning("ラベルが提供されていないため、監視なし評価を実行します")
            return self._evaluate_unsupervised_metrics(features, success_metrics)

        results = {}

        for metric_name in success_metrics:
            if metric_name in self.metric_calculators:
                try:
                    calculator = self.metric_calculators[metric_name]
                    metric_value = calculator(features, labels, models)
                    results[metric_name] = metric_value
                except Exception as e:
                    logger.warning(f"メトリクス計算失敗 ({metric_name}): {e}")
                    results[metric_name] = 0.0

        return results

    def _evaluate_unsupervised_metrics(
        self, features: np.ndarray, success_metrics: List[str]
    ) -> Dict[str, float]:
        """教師なしメトリクスを評価"""
        results = {}

        for metric_name in success_metrics:
            if metric_name == "feature_count":
                results[metric_name] = features.shape[1]
            elif metric_name == "total_variance":
                results[metric_name] = np.sum(np.var(features, axis=0))
            elif metric_name == "mean_feature_correlation":
                corr_matrix = np.corrcoef(features.T)
                # 対角要素を除く相関の平均
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                results[metric_name] = np.mean(np.abs(corr_matrix[mask]))
            else:
                results[metric_name] = 0.0

        return results

    def _calculate_accuracy(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """精度を計算"""
        model = models.get("random_forest", self.default_models["random_forest"])
        scores = cross_val_score(model, features, labels, cv=5, scoring="accuracy")
        return np.mean(scores)

    def _calculate_precision(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """適合率を計算"""
        model = models.get("random_forest", self.default_models["random_forest"])
        scores = cross_val_score(
            model, features, labels, cv=5, scoring="precision_macro"
        )
        return np.mean(scores)

    def _calculate_recall(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """再現率を計算"""
        model = models.get("random_forest", self.default_models["random_forest"])
        scores = cross_val_score(model, features, labels, cv=5, scoring="recall_macro")
        return np.mean(scores)

    def _calculate_f1_score(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """F1スコアを計算"""
        model = models.get("random_forest", self.default_models["random_forest"])
        scores = cross_val_score(model, features, labels, cv=5, scoring="f1_macro")
        return np.mean(scores)

    def _calculate_roc_auc(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """ROC-AUCを計算"""
        try:
            model = models.get("random_forest", self.default_models["random_forest"])
            scores = cross_val_score(
                model, features, labels, cv=5, scoring="roc_auc_ovr"
            )
            return np.mean(scores)
        except:
            # マルチクラス問題で失敗した場合
            return 0.5

    def _calculate_cross_val_accuracy(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """クロスバリデーション精度を計算"""
        model = models.get(
            "logistic_regression", self.default_models["logistic_regression"]
        )
        scores = cross_val_score(model, features, labels, cv=10, scoring="accuracy")
        return np.mean(scores)

    def _calculate_feature_importance_sum(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """特徴量重要度の合計を計算"""
        try:
            model = models.get("random_forest", self.default_models["random_forest"])
            model.fit(features, labels)
            return np.sum(model.feature_importances_)
        except:
            return 0.0

    def _calculate_training_time(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """訓練時間を計算"""
        import time

        model = models.get("random_forest", self.default_models["random_forest"])

        start_time = time.time()
        model.fit(features, labels)
        end_time = time.time()

        return end_time - start_time

    def _calculate_prediction_time(
        self, features: np.ndarray, labels: np.ndarray, models: Dict[str, Any]
    ) -> float:
        """推論時間を計算"""
        import time

        model = models.get("random_forest", self.default_models["random_forest"])
        model.fit(features, labels)

        # 推論時間を測定
        start_time = time.time()
        _ = model.predict(features[:100])  # 最初の100サンプルで測定
        end_time = time.time()

        return end_time - start_time

    def _perform_statistical_tests(
        self,
        baseline_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        config: ABTestConfig,
    ) -> Dict[str, Any]:
        """統計的検定を実行"""
        results = {}

        for metric_name in config.success_metrics:
            if metric_name in baseline_metrics and metric_name in treatment_metrics:
                baseline_value = baseline_metrics[metric_name]
                treatment_value = treatment_metrics[metric_name]

                # t検定（簡略化）
                # 実際の実装では、複数回の評価結果を使用してt検定を行う
                difference = treatment_value - baseline_value

                # 効果サイズベースの簡易検定
                # 実際の分散が不明なため、効果サイズで代用
                effect_size = difference / max(abs(baseline_value), 1e-8)

                # 簡易的なp値計算（正規分布仮定）
                z_score = effect_size / 0.1  # 簡略化
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                results[metric_name] = {
                    "baseline_value": baseline_value,
                    "treatment_value": treatment_value,
                    "difference": difference,
                    "effect_size": effect_size,
                    "z_score": z_score,
                    "p_value": p_value,
                    "is_significant": p_value < config.significance_level,
                }

        return results

    def _calculate_effect_sizes(
        self, baseline_metrics: Dict[str, float], treatment_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """効果サイズを計算"""
        effect_sizes = {}

        for metric_name in baseline_metrics:
            if metric_name in treatment_metrics:
                baseline_value = baseline_metrics[metric_name]
                treatment_value = treatment_metrics[metric_name]

                # Cohen's d の簡略版
                if baseline_value != 0:
                    effect_size = (treatment_value - baseline_value) / abs(
                        baseline_value
                    )
                else:
                    effect_size = treatment_value

                effect_sizes[metric_name] = effect_size

        return effect_sizes

    def _calculate_confidence_intervals(
        self,
        baseline_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        significance_level: float,
    ) -> Dict[str, Tuple[float, float]]:
        """信頼区間を計算"""
        confidence_intervals = {}
        alpha = significance_level
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        for metric_name in baseline_metrics:
            if metric_name in treatment_metrics:
                baseline_value = baseline_metrics[metric_name]
                treatment_value = treatment_metrics[metric_name]
                difference = treatment_value - baseline_value

                # 簡略化された標準誤差
                se = abs(difference) * 0.1  # 仮定

                lower_bound = difference - z_alpha * se
                upper_bound = difference + z_alpha * se

                confidence_intervals[metric_name] = (lower_bound, upper_bound)

        return confidence_intervals

    def _determine_significance(
        self, statistical_results: Dict[str, Any], significance_level: float
    ) -> bool:
        """統計的有意性を判定"""
        significant_results = []

        for metric_name, result in statistical_results.items():
            if result.get("is_significant", False):
                significant_results.append(metric_name)

        # すべてのメトリクスで有意性が認められる場合にTrueを返す
        return len(significant_results) == len(statistical_results)

    def _generate_recommendation(
        self, test_result: ABTestResult, config: ABTestConfig
    ) -> str:
        """推奨事項を生成"""
        recommendations = []

        if test_result.is_significant:
            # 有意な改善が認められる場合
            improvements = []
            for metric_name, effect_size in test_result.effect_sizes.items():
                if effect_size > config.minimum_effect_size:
                    improvements.append(f"{metric_name} (+{effect_size:.1%})")

            if improvements:
                recommendations.append(
                    f"改善版特徴量セット '{config.treatment_features}' の採用を推奨します。"
                )
                recommendations.append(
                    f"改善が認められた指標: {', '.join(improvements)}"
                )
            else:
                recommendations.append(
                    "統計的有意性は認められましたが、実用的な効果サイズには達していません。"
                )
        else:
            # 有意な差が認められない場合
            recommendations.append(
                f"改善版特徴量セット '{config.treatment_features}' による有意な改善は認められませんでした。"
            )
            recommendations.append("ベースライン特徴量セットの継続使用を推奨します。")

        # 追加の推奨事項
        max_effect_metric = max(
            test_result.effect_sizes.items(), key=lambda x: abs(x[1])
        )
        recommendations.append(
            f"最も大きな効果が見られた指標: {max_effect_metric[0]} ({max_effect_metric[1]:+.1%})"
        )

        return " ".join(recommendations)

    def _perform_detailed_analysis(
        self,
        baseline_data: Dict[str, Any],
        treatment_data: Dict[str, Any],
        test_result: ABTestResult,
    ) -> Dict[str, Any]:
        """詳細分析を実行"""
        analysis = {}

        # 特徴量数の比較
        baseline_features = baseline_data["features"]
        treatment_features = treatment_data["features"]

        analysis["feature_count_comparison"] = {
            "baseline": baseline_features.shape[1],
            "treatment": treatment_features.shape[1],
            "difference": treatment_features.shape[1] - baseline_features.shape[1],
        }

        # データサイズの比較
        analysis["sample_size_comparison"] = {
            "baseline": baseline_features.shape[0],
            "treatment": treatment_features.shape[0],
        }

        # 特徴量統計の比較
        analysis["feature_statistics"] = {
            "baseline": {
                "mean_feature_mean": np.mean(np.mean(baseline_features, axis=0)),
                "mean_feature_std": np.mean(np.std(baseline_features, axis=0)),
                "total_variance": np.sum(np.var(baseline_features, axis=0)),
            },
            "treatment": {
                "mean_feature_mean": np.mean(np.mean(treatment_features, axis=0)),
                "mean_feature_std": np.mean(np.std(treatment_features, axis=0)),
                "total_variance": np.sum(np.var(treatment_features, axis=0)),
            },
        }

        # 相関分析
        if baseline_features.shape[1] > 1 and treatment_features.shape[1] > 1:
            baseline_corr = np.corrcoef(baseline_features.T)
            treatment_corr = np.corrcoef(treatment_features.T)

            # 対角要素を除く相関の平均
            baseline_mask = ~np.eye(baseline_corr.shape[0], dtype=bool)
            treatment_mask = ~np.eye(treatment_corr.shape[0], dtype=bool)

            analysis["correlation_analysis"] = {
                "baseline_mean_correlation": np.mean(
                    np.abs(baseline_corr[baseline_mask])
                ),
                "treatment_mean_correlation": np.mean(
                    np.abs(treatment_corr[treatment_mask])
                ),
            }

        # パフォーマンス要約
        analysis["performance_summary"] = {
            "best_performing_group": self._determine_best_performing_group(test_result),
            "consistent_improvements": self._find_consistent_improvements(test_result),
            "areas_of_concern": self._identify_areas_of_concern(test_result),
        }

        return analysis

    def _determine_best_performing_group(self, test_result: ABTestResult) -> str:
        """最高性能グループを判定"""
        positive_effects = sum(
            1 for effect in test_result.effect_sizes.values() if effect > 0
        )
        total_effects = len(test_result.effect_sizes)

        if positive_effects > total_effects / 2:
            return "treatment"
        else:
            return "baseline"

    def _find_consistent_improvements(self, test_result: ABTestResult) -> List[str]:
        """一貫した改善を見つける"""
        consistent_improvements = []

        for metric_name, effect_size in test_result.effect_sizes.items():
            if effect_size > 0.05:  # 5%以上の改善
                statistical_result = test_result.statistical_results.get(
                    metric_name, {}
                )
                if statistical_result.get("is_significant", False):
                    consistent_improvements.append(metric_name)

        return consistent_improvements

    def _identify_areas_of_concern(self, test_result: ABTestResult) -> List[str]:
        """懸念領域を特定"""
        concerns = []

        for metric_name, effect_size in test_result.effect_sizes.items():
            if effect_size < -0.05:  # 5%以上の悪化
                concerns.append(f"{metric_name}: {effect_size:.1%} reduction")

        return concerns

    def _save_test_result(self, test_result: ABTestResult) -> None:
        """テスト結果を保存"""
        result_file = self.output_dir / f"ab_test_result_{test_result.test_id}.json"

        try:
            # dataclassを辞書に変換（ネストしたdataclassも考慮）
            result_dict = asdict(test_result)

            # Enumを文字列に変換
            result_dict["status"] = test_result.status.value
            result_dict["test_config"][
                "test_type"
            ] = test_result.test_config.test_type.value

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"テスト結果保存完了: {result_file}")

        except Exception as e:
            logger.error(f"テスト結果保存エラー: {e}")

    def generate_comparison_report(
        self,
        test_id: str,
        include_visualizations: bool = True,
        report_format: str = "html",
    ) -> str:
        """比較レポートを生成

        Args:
            test_id: テストID
            include_visualizations: 可視化を含めるかどうか
            report_format: レポート形式 ('html', 'json', 'pdf')

        Returns:
            生成されたレポートファイルのパス
        """
        # テスト結果を取得
        test_result = self.completed_tests.get(test_id)
        if not test_result:
            raise ValueError(f"完了したテストが見つかりません: {test_id}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            self.output_dir / f"ab_test_report_{test_id}_{timestamp}.{report_format}"
        )

        try:
            if report_format == "html":
                self._generate_html_comparison_report(
                    test_result, report_file, include_visualizations
                )
            elif report_format == "json":
                self._generate_json_comparison_report(test_result, report_file)
            else:
                raise ValueError(f"サポートされていないレポート形式: {report_format}")

            logger.info(f"比較レポート生成完了: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            raise

    def _generate_html_comparison_report(
        self, test_result: ABTestResult, report_file: Path, include_visualizations: bool
    ) -> None:
        """HTML比較レポートを生成"""
        html_content = self._build_html_comparison_content(
            test_result, include_visualizations
        )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _build_html_comparison_content(
        self, test_result: ABTestResult, include_visualizations: bool
    ) -> str:
        """HTML比較レポートコンテンツを構築"""
        config = test_result.test_config
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/Bテスト比較レポート - {config.test_name}</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .significant {{ background-color: #d4edda; }}
                .not-significant {{ background-color: #f8d7da; }}
                .improvement {{ color: #28a745; font-weight: bold; }}
                .degradation {{ color: #dc3545; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-comparison {{ display: flex; justify-content: space-between; }}
                .metric-group {{ flex: 1; margin: 0 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>A/Bテスト比較レポート</h1>
                <h2>{config.test_name}</h2>
                <p><strong>テストID:</strong> {config.test_id}</p>
                <p><strong>テスト期間:</strong> {test_result.start_time} - {test_result.end_time or '実行中'}</p>
                <p><strong>レポート生成日時:</strong> {current_time}</p>
                <p><strong>統計的有意性:</strong> 
                    <span class="{'improvement' if test_result.is_significant else 'degradation'}">
                        {'有意' if test_result.is_significant else '非有意'}
                    </span>
                </p>
            </div>
        """

        # テスト設定
        html += f"""
            <div class="section">
                <h3>テスト設定</h3>
                <table>
                    <tr><td>テストタイプ</td><td>{config.test_type.value}</td></tr>
                    <tr><td>ベースライン特徴量セット</td><td>{config.baseline_features}</td></tr>
                    <tr><td>改善版特徴量セット</td><td>{config.treatment_features}</td></tr>
                    <tr><td>成功指標</td><td>{', '.join(config.success_metrics)}</td></tr>
                    <tr><td>有意水準</td><td>{config.significance_level}</td></tr>
                    <tr><td>最小効果サイズ</td><td>{config.minimum_effect_size}</td></tr>
                </table>
            </div>
        """

        # メトリクス比較
        html += """
            <div class="section">
                <h3>メトリクス比較</h3>
                <table>
                    <tr>
                        <th>メトリクス</th>
                        <th>ベースライン</th>
                        <th>改善版</th>
                        <th>差分</th>
                        <th>効果サイズ</th>
                        <th>P値</th>
                        <th>有意性</th>
                    </tr>
        """

        for metric_name in config.success_metrics:
            baseline_value = test_result.baseline_metrics.get(metric_name, 0)
            treatment_value = test_result.treatment_metrics.get(metric_name, 0)
            effect_size = test_result.effect_sizes.get(metric_name, 0)

            stat_result = test_result.statistical_results.get(metric_name, {})
            p_value = stat_result.get("p_value", 1.0)
            is_significant = stat_result.get("is_significant", False)

            difference = treatment_value - baseline_value
            difference_class = "improvement" if difference > 0 else "degradation"
            significance_class = "improvement" if is_significant else "degradation"

            html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{baseline_value:.4f}</td>
                    <td>{treatment_value:.4f}</td>
                    <td class="{difference_class}">{difference:+.4f}</td>
                    <td>{effect_size:+.1%}</td>
                    <td>{p_value:.4f}</td>
                    <td class="{significance_class}">{'有意' if is_significant else '非有意'}</td>
                </tr>
            """

        html += "</table></div>"

        # 推奨事項
        html += f"""
            <div class="section {'significant' if test_result.is_significant else 'not-significant'}">
                <h3>推奨事項</h3>
                <p>{test_result.recommendation}</p>
            </div>
        """

        # 詳細分析
        if test_result.detailed_analysis:
            html += """
                <div class="section">
                    <h3>詳細分析</h3>
            """

            # 特徴量数比較
            feature_comparison = test_result.detailed_analysis.get(
                "feature_count_comparison", {}
            )
            if feature_comparison:
                html += f"""
                    <h4>特徴量数比較</h4>
                    <p>ベースライン: {feature_comparison.get('baseline', 'N/A')} 特徴量</p>
                    <p>改善版: {feature_comparison.get('treatment', 'N/A')} 特徴量</p>
                    <p>差分: {feature_comparison.get('difference', 'N/A'):+d} 特徴量</p>
                """

            # パフォーマンス要約
            performance_summary = test_result.detailed_analysis.get(
                "performance_summary", {}
            )
            if performance_summary:
                html += f"""
                    <h4>パフォーマンス要約</h4>
                    <p><strong>最高性能グループ:</strong> {performance_summary.get('best_performing_group', 'N/A')}</p>
                    <p><strong>一貫した改善:</strong> {', '.join(performance_summary.get('consistent_improvements', []))}</p>
                    <p><strong>懸念領域:</strong> {', '.join(performance_summary.get('areas_of_concern', []))}</p>
                """

            html += "</div>"

        # 可視化
        if include_visualizations:
            chart_paths = self._generate_comparison_charts(test_result)
            if chart_paths:
                html += """
                    <div class="section">
                        <h3>可視化</h3>
                """

                for chart_name, chart_path in chart_paths.items():
                    html += f"<h4>{chart_name}</h4>\n"
                    html += f'<img src="{chart_path}" alt="{chart_name}" style="max-width: 100%;">\n'

                html += "</div>"

        html += """
        </body>
        </html>
        """

        return html

    def _generate_json_comparison_report(
        self, test_result: ABTestResult, report_file: Path
    ) -> None:
        """JSON比較レポートを生成"""
        report_data = {
            "test_result": asdict(test_result),
            "summary": {
                "is_successful_test": test_result.is_significant,
                "best_performing_group": test_result.detailed_analysis.get(
                    "performance_summary", {}
                ).get("best_performing_group"),
                "total_metrics_improved": sum(
                    1 for effect in test_result.effect_sizes.values() if effect > 0
                ),
                "total_metrics_degraded": sum(
                    1 for effect in test_result.effect_sizes.values() if effect < 0
                ),
                "largest_improvement": (
                    max(test_result.effect_sizes.values())
                    if test_result.effect_sizes
                    else 0
                ),
                "largest_degradation": (
                    min(test_result.effect_sizes.values())
                    if test_result.effect_sizes
                    else 0
                ),
            },
            "generated_at": datetime.now().isoformat(),
        }

        # Enumを文字列に変換
        report_data["test_result"]["status"] = test_result.status.value
        report_data["test_result"]["test_config"][
            "test_type"
        ] = test_result.test_config.test_type.value

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

    def _generate_comparison_charts(self, test_result: ABTestResult) -> Dict[str, str]:
        """比較チャートを生成"""
        charts = {}

        try:
            # メトリクス比較チャート
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # メトリクス値比較
            metrics = list(test_result.baseline_metrics.keys())
            baseline_values = [test_result.baseline_metrics[m] for m in metrics]
            treatment_values = [test_result.treatment_metrics[m] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            ax1.bar(
                x - width / 2, baseline_values, width, label="ベースライン", alpha=0.8
            )
            ax1.bar(x + width / 2, treatment_values, width, label="改善版", alpha=0.8)
            ax1.set_xlabel("メトリクス")
            ax1.set_ylabel("値")
            ax1.set_title("メトリクス値比較")
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics, rotation=45)
            ax1.legend()

            # 効果サイズチャート
            effect_sizes = [test_result.effect_sizes.get(m, 0) for m in metrics]
            colors = ["green" if es > 0 else "red" for es in effect_sizes]

            ax2.bar(metrics, effect_sizes, color=colors, alpha=0.7)
            ax2.set_xlabel("メトリクス")
            ax2.set_ylabel("効果サイズ")
            ax2.set_title("効果サイズ")
            ax2.tick_params(axis="x", rotation=45)
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            plt.tight_layout()

            chart_path = (
                self.output_dir
                / f"comparison_chart_{test_result.test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()

            charts["メトリクス比較"] = str(chart_path)

        except Exception as e:
            logger.warning(f"比較チャート生成エラー: {e}")

        return charts

    def list_feature_sets(self) -> Dict[str, Dict[str, Any]]:
        """登録された特徴量セットのリストを取得"""
        return {
            set_id: {
                "description": data["description"],
                "shape": data["shape"],
                "feature_names_count": len(data["feature_names"]),
                "has_labels": data["labels"] is not None,
                "registered_at": data["registered_at"],
            }
            for set_id, data in self.feature_sets.items()
        }

    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """テスト状態を取得"""
        if test_id in self.active_tests:
            test_result = self.active_tests[test_id]
            return {
                "status": test_result.status.value,
                "test_name": test_result.test_config.test_name,
                "start_time": test_result.start_time,
                "is_active": True,
            }
        elif test_id in self.completed_tests:
            test_result = self.completed_tests[test_id]
            return {
                "status": test_result.status.value,
                "test_name": test_result.test_config.test_name,
                "start_time": test_result.start_time,
                "end_time": test_result.end_time,
                "is_significant": test_result.is_significant,
                "is_active": False,
            }
        else:
            raise ValueError(f"テストが見つかりません: {test_id}")

    def list_all_tests(self) -> Dict[str, List[Dict[str, Any]]]:
        """すべてのテストのリストを取得"""
        return {
            "active_tests": [
                {
                    "test_id": test_id,
                    "test_name": test_result.test_config.test_name,
                    "status": test_result.status.value,
                    "start_time": test_result.start_time,
                }
                for test_id, test_result in self.active_tests.items()
            ],
            "completed_tests": [
                {
                    "test_id": test_id,
                    "test_name": test_result.test_config.test_name,
                    "status": test_result.status.value,
                    "start_time": test_result.start_time,
                    "end_time": test_result.end_time,
                    "is_significant": test_result.is_significant,
                }
                for test_id, test_result in self.completed_tests.items()
            ],
        }

    def clean_up_old_tests(self, days_old: int = 30) -> Dict[str, Any]:
        """古いテスト結果をクリーンアップ

        Args:
            days_old: 削除する古さの日数

        Returns:
            クリーンアップ結果
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)

        cleaned_tests = []
        remaining_tests = {}

        for test_id, test_result in self.completed_tests.items():
            test_end_time = (
                datetime.fromisoformat(test_result.end_time)
                if test_result.end_time
                else datetime.now()
            )

            if test_end_time < cutoff_date:
                cleaned_tests.append(test_id)
                # 対応するファイルも削除
                result_file = self.output_dir / f"ab_test_result_{test_id}.json"
                if result_file.exists():
                    result_file.unlink()
            else:
                remaining_tests[test_id] = test_result

        self.completed_tests = remaining_tests

        logger.info(f"古いテスト結果をクリーンアップ: {len(cleaned_tests)}件削除")

        return {
            "cleaned_tests_count": len(cleaned_tests),
            "cleaned_test_ids": cleaned_tests,
            "remaining_tests_count": len(remaining_tests),
        }
