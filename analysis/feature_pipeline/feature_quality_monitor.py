"""
特徴量品質監視器
===============

特徴量品質の継続監視、自動品質レポート生成、品質劣化アラート機能を実装します。
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

logger = logging.getLogger(__name__)


class QualityMetricType(Enum):
    """品質メトリクスの種類"""
    STABILITY = "stability"
    INFORMATION_GAIN = "information_gain"
    REDUNDANCY = "redundancy"
    CORRELATION = "correlation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    VARIANCE = "variance"


class AlertSeverity(Enum):
    """アラートの重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """品質メトリクス情報"""
    metric_type: QualityMetricType
    value: float
    threshold: float
    timestamp: str
    feature_names: Optional[List[str]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class QualityAlert:
    """品質アラート情報"""
    alert_id: str
    severity: AlertSeverity
    metric_type: QualityMetricType
    message: str
    current_value: float
    threshold: float
    timestamp: str
    feature_names: Optional[List[str]] = None
    suggested_actions: Optional[List[str]] = None


class FeatureQualityMonitor:
    """特徴量品質監視器
    
    特徴量品質の継続監視、自動品質レポート生成、
    品質劣化アラート機能を実装。
    """
    
    def __init__(self, 
                 monitoring_config: Optional[Dict[str, Any]] = None,
                 output_dir: Optional[str] = None):
        """
        Args:
            monitoring_config: 監視設定
            output_dir: 出力ディレクトリのパス
        """
        self.config = monitoring_config or self._get_default_config()
        self.output_dir = Path(output_dir) if output_dir else Path('./quality_monitoring')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 監視状態
        self.is_monitoring = False
        self.baseline_metrics = {}
        self.historical_metrics = []
        self.active_alerts = []
        self.monitoring_start_time = None
        
        # 品質メトリクス計算器
        self.metric_calculators = {
            QualityMetricType.STABILITY: self._calculate_stability,
            QualityMetricType.INFORMATION_GAIN: self._calculate_information_gain,
            QualityMetricType.REDUNDANCY: self._calculate_redundancy,
            QualityMetricType.CORRELATION: self._calculate_correlation_quality,
            QualityMetricType.DISTRIBUTION_SHIFT: self._calculate_distribution_shift,
            QualityMetricType.MISSING_VALUES: self._calculate_missing_values,
            QualityMetricType.OUTLIERS: self._calculate_outliers,
            QualityMetricType.VARIANCE: self._calculate_variance
        }
        
        # アラート生成器
        self.alert_handlers = []
        
        logger.info(f"FeatureQualityMonitor初期化完了: output_dir={self.output_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト監視設定を取得"""
        return {
            'monitoring_interval_seconds': 300,  # 5分間隔
            'quality_thresholds': {
                'stability': 0.8,
                'information_gain': 0.1,
                'redundancy': 0.7,
                'correlation': 0.9,
                'distribution_shift': 0.3,
                'missing_values': 0.1,
                'outliers': 0.05,
                'variance': 0.01
            },
            'alert_thresholds': {
                'low': 0.1,      # 閾値からの乖離率
                'medium': 0.2,
                'high': 0.4,
                'critical': 0.6
            },
            'historical_window_hours': 24,
            'baseline_update_frequency': 7,  # 日数
            'auto_report_frequency': 24,     # 時間
            'enable_alerts': True,
            'enable_auto_reports': True,
            'enable_visualization': True
        }
    
    def start_monitoring(self, 
                        baseline_features: np.ndarray,
                        baseline_labels: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None) -> 'FeatureQualityMonitor':
        """品質監視を開始
        
        Args:
            baseline_features: ベースライン特徴量
            baseline_labels: ベースラインラベル（情報利得計算用）
            feature_names: 特徴量名のリスト
            
        Returns:
            自身のインスタンス
        """
        try:
            # ベースラインメトリクスを計算
            self.baseline_metrics = self._calculate_all_metrics(
                baseline_features, baseline_labels, feature_names
            )
            
            # 監視状態を更新
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            self.historical_metrics = []
            self.active_alerts = []
            
            # ベースライン保存
            self._save_baseline_metrics()
            
            logger.info(f"品質監視開始: {len(self.baseline_metrics)} メトリクス設定完了")
            return self
            
        except Exception as e:
            logger.error(f"監視開始エラー: {e}")
            raise
    
    def monitor_current_quality(self,
                              current_features: np.ndarray,
                              current_labels: Optional[np.ndarray] = None,
                              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """現在の特徴量品質を監視
        
        Args:
            current_features: 現在の特徴量
            current_labels: 現在のラベル
            feature_names: 特徴量名のリスト
            
        Returns:
            監視結果
        """
        if not self.is_monitoring:
            raise ValueError("監視が開始されていません。start_monitoring()を先に実行してください。")
        
        try:
            # 現在のメトリクスを計算
            current_metrics = self._calculate_all_metrics(
                current_features, current_labels, feature_names
            )
            
            # 品質変化を評価
            quality_changes = self._evaluate_quality_changes(current_metrics)
            
            # アラートをチェック
            new_alerts = self._check_for_alerts(current_metrics, quality_changes)
            
            # 履歴に追加
            timestamp = datetime.now().isoformat()
            self.historical_metrics.append({
                'timestamp': timestamp,
                'metrics': current_metrics,
                'quality_changes': quality_changes
            })
            
            # 古いデータを削除
            self._cleanup_historical_data()
            
            # 結果を整理
            monitoring_result = {
                'timestamp': timestamp,
                'current_metrics': current_metrics,
                'quality_changes': quality_changes,
                'new_alerts': [asdict(alert) for alert in new_alerts],
                'active_alerts_count': len(self.active_alerts),
                'overall_quality_score': self._calculate_overall_quality_score(current_metrics),
                'recommendations': self._generate_recommendations(current_metrics, quality_changes)
            }
            
            # アラート処理
            if new_alerts and self.config.get('enable_alerts', True):
                self._handle_alerts(new_alerts)
            
            logger.info(f"品質監視完了: 品質スコア={monitoring_result['overall_quality_score']:.3f}")
            return monitoring_result
            
        except Exception as e:
            logger.error(f"品質監視エラー: {e}")
            raise
    
    def _calculate_all_metrics(self, 
                              features: np.ndarray,
                              labels: Optional[np.ndarray] = None,
                              feature_names: Optional[List[str]] = None) -> Dict[str, QualityMetric]:
        """すべての品質メトリクスを計算"""
        metrics = {}
        timestamp = datetime.now().isoformat()
        
        for metric_type in QualityMetricType:
            try:
                calculator = self.metric_calculators.get(metric_type)
                if calculator:
                    value = calculator(features, labels, feature_names)
                    threshold = self.config['quality_thresholds'].get(metric_type.value, 0.5)
                    
                    metrics[metric_type.value] = QualityMetric(
                        metric_type=metric_type,
                        value=value,
                        threshold=threshold,
                        timestamp=timestamp,
                        feature_names=feature_names
                    )
                    
            except Exception as e:
                logger.warning(f"メトリクス計算失敗 ({metric_type.value}): {e}")
        
        return metrics
    
    def _calculate_stability(self, 
                           features: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None) -> float:
        """特徴量の安定性を計算"""
        if len(self.historical_metrics) == 0:
            return 1.0  # 最初の計算では最高値
        
        # 最近のメトリクスと比較
        recent_metrics = self.historical_metrics[-5:]  # 直近5回
        if not recent_metrics:
            return 1.0
        
        # 特徴量の統計量の変動を計算
        current_mean = np.mean(features, axis=0)
        current_std = np.std(features, axis=0)
        
        stability_scores = []
        for historical_entry in recent_metrics:
            hist_metrics = historical_entry.get('metrics', {})
            if 'stability' in hist_metrics:
                # 簡略化: 現在と過去の標準偏差の比較
                stability_score = 1.0 - min(1.0, np.mean(np.abs(current_std - np.mean(current_std))))
                stability_scores.append(stability_score)
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _calculate_information_gain(self, 
                                  features: np.ndarray,
                                  labels: Optional[np.ndarray] = None,
                                  feature_names: Optional[List[str]] = None) -> float:
        """情報利得を計算"""
        if labels is None:
            return 0.5  # ラベルがない場合はデフォルト値
        
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import LabelEncoder

            # ラベルを数値に変換
            if labels.dtype == 'object' or labels.dtype.kind in 'SU':
                le = LabelEncoder()
                numeric_labels = le.fit_transform(labels)
            else:
                numeric_labels = labels
            
            # 相互情報量を計算
            mi_scores = mutual_info_classif(features, numeric_labels, random_state=42)
            
            # 平均情報利得を返す
            return np.mean(mi_scores)
            
        except Exception as e:
            logger.warning(f"情報利得計算エラー: {e}")
            return 0.5
    
    def _calculate_redundancy(self, 
                            features: np.ndarray,
                            labels: Optional[np.ndarray] = None,
                            feature_names: Optional[List[str]] = None) -> float:
        """特徴量の冗長性を計算"""
        try:
            # 特徴量間の相関行列を計算
            correlation_matrix = np.corrcoef(features.T)
            
            # 対角要素を除く
            mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
            correlations = correlation_matrix[mask]
            
            # 高い相関（>0.8）の割合を冗長性とする
            high_correlation_ratio = np.sum(np.abs(correlations) > 0.8) / len(correlations)
            
            return high_correlation_ratio
            
        except Exception as e:
            logger.warning(f"冗長性計算エラー: {e}")
            return 0.5
    
    def _calculate_correlation_quality(self, 
                                     features: np.ndarray,
                                     labels: Optional[np.ndarray] = None,
                                     feature_names: Optional[List[str]] = None) -> float:
        """特徴量とラベル間の相関品質を計算"""
        if labels is None:
            return 0.5
        
        try:
            # 各特徴量とラベルの相関を計算
            correlations = []
            for i in range(features.shape[1]):
                corr, _ = stats.pearsonr(features[:, i], labels)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.5
            
        except Exception as e:
            logger.warning(f"相関品質計算エラー: {e}")
            return 0.5
    
    def _calculate_distribution_shift(self, 
                                    features: np.ndarray,
                                    labels: Optional[np.ndarray] = None,
                                    feature_names: Optional[List[str]] = None) -> float:
        """分布シフトを計算"""
        if not hasattr(self, 'baseline_features'):
            return 0.0  # ベースラインがない場合
        
        try:
            # KLダイバージェンスやKSテストで分布シフトを測定
            shift_scores = []
            
            for i in range(min(features.shape[1], 10)):  # 最初の10特徴量をサンプリング
                # KSテスト
                ks_stat, p_value = stats.ks_2samp(
                    self.baseline_features[:, i] if hasattr(self, 'baseline_features') else features[:, i],
                    features[:, i]
                )
                shift_scores.append(ks_stat)
            
            return np.mean(shift_scores) if shift_scores else 0.0
            
        except Exception as e:
            logger.warning(f"分布シフト計算エラー: {e}")
            return 0.0
    
    def _calculate_missing_values(self, 
                                features: np.ndarray,
                                labels: Optional[np.ndarray] = None,
                                feature_names: Optional[List[str]] = None) -> float:
        """欠損値の割合を計算"""
        try:
            # NaNやInfの割合を計算
            missing_ratio = (np.isnan(features).sum() + np.isinf(features).sum()) / features.size
            return missing_ratio
            
        except Exception as e:
            logger.warning(f"欠損値計算エラー: {e}")
            return 0.0
    
    def _calculate_outliers(self, 
                          features: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None) -> float:
        """外れ値の割合を計算"""
        try:
            # IQRベースの外れ値検出
            outlier_ratios = []
            
            for i in range(features.shape[1]):
                q1 = np.percentile(features[:, i], 25)
                q3 = np.percentile(features[:, i], 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = (features[:, i] < lower_bound) | (features[:, i] > upper_bound)
                outlier_ratio = np.sum(outliers) / len(features[:, i])
                outlier_ratios.append(outlier_ratio)
            
            return np.mean(outlier_ratios)
            
        except Exception as e:
            logger.warning(f"外れ値計算エラー: {e}")
            return 0.0
    
    def _calculate_variance(self, 
                          features: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None) -> float:
        """特徴量の分散を計算"""
        try:
            # 各特徴量の分散を計算
            variances = np.var(features, axis=0)
            
            # 低分散特徴量の割合
            low_variance_ratio = np.sum(variances < 0.01) / len(variances)
            
            return 1.0 - low_variance_ratio  # 高分散ほど良い品質
            
        except Exception as e:
            logger.warning(f"分散計算エラー: {e}")
            return 0.5
    
    def _evaluate_quality_changes(self, current_metrics: Dict[str, QualityMetric]) -> Dict[str, float]:
        """品質変化を評価"""
        changes = {}
        
        for metric_name, current_metric in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name].value
                current_value = current_metric.value
                
                # 変化率を計算
                if baseline_value != 0:
                    change_ratio = (current_value - baseline_value) / baseline_value
                else:
                    change_ratio = current_value
                
                changes[metric_name] = change_ratio
        
        return changes
    
    def _check_for_alerts(self, 
                         current_metrics: Dict[str, QualityMetric],
                         quality_changes: Dict[str, float]) -> List[QualityAlert]:
        """アラートをチェック"""
        new_alerts = []
        alert_thresholds = self.config.get('alert_thresholds', {})
        
        for metric_name, current_metric in current_metrics.items():
            threshold = current_metric.threshold
            current_value = current_metric.value
            
            # 閾値からの乖離を計算
            if threshold != 0:
                deviation = abs(current_value - threshold) / threshold
            else:
                deviation = current_value
            
            # アラートレベルを判定
            severity = self._determine_alert_severity(deviation, alert_thresholds)
            
            if severity:
                alert_id = f"{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                alert = QualityAlert(
                    alert_id=alert_id,
                    severity=severity,
                    metric_type=current_metric.metric_type,
                    message=self._generate_alert_message(current_metric, deviation),
                    current_value=current_value,
                    threshold=threshold,
                    timestamp=datetime.now().isoformat(),
                    feature_names=current_metric.feature_names,
                    suggested_actions=self._generate_suggested_actions(current_metric, deviation)
                )
                
                new_alerts.append(alert)
                self.active_alerts.append(alert)
        
        return new_alerts
    
    def _determine_alert_severity(self, deviation: float, thresholds: Dict[str, float]) -> Optional[AlertSeverity]:
        """アラートの重要度を判定"""
        if deviation >= thresholds.get('critical', 0.6):
            return AlertSeverity.CRITICAL
        elif deviation >= thresholds.get('high', 0.4):
            return AlertSeverity.HIGH
        elif deviation >= thresholds.get('medium', 0.2):
            return AlertSeverity.MEDIUM
        elif deviation >= thresholds.get('low', 0.1):
            return AlertSeverity.LOW
        else:
            return None
    
    def _generate_alert_message(self, metric: QualityMetric, deviation: float) -> str:
        """アラートメッセージを生成"""
        metric_name = metric.metric_type.value
        current_value = metric.value
        threshold = metric.threshold
        
        if current_value < threshold:
            direction = "below"
            action = "improve"
        else:
            direction = "above"
            action = "reduce"
        
        return (f"Quality metric '{metric_name}' is {direction} threshold: "
                f"current={current_value:.3f}, threshold={threshold:.3f}, "
                f"deviation={deviation:.1%}. Consider actions to {action} this metric.")
    
    def _generate_suggested_actions(self, metric: QualityMetric, deviation: float) -> List[str]:
        """推奨アクションを生成"""
        actions = []
        metric_type = metric.metric_type
        
        if metric_type == QualityMetricType.STABILITY:
            actions.extend([
                "Review data preprocessing pipeline",
                "Check for data source changes",
                "Validate feature extraction logic"
            ])
        elif metric_type == QualityMetricType.INFORMATION_GAIN:
            actions.extend([
                "Re-evaluate feature selection criteria",
                "Consider additional feature engineering",
                "Review target variable distribution"
            ])
        elif metric_type == QualityMetricType.REDUNDANCY:
            actions.extend([
                "Apply correlation-based feature removal",
                "Use dimensionality reduction techniques",
                "Review feature selection threshold"
            ])
        elif metric_type == QualityMetricType.MISSING_VALUES:
            actions.extend([
                "Investigate data collection issues",
                "Update missing value handling strategy",
                "Validate data pipeline integrity"
            ])
        elif metric_type == QualityMetricType.OUTLIERS:
            actions.extend([
                "Review outlier detection thresholds",
                "Investigate data quality issues",
                "Consider robust scaling methods"
            ])
        
        return actions
    
    def _handle_alerts(self, alerts: List[QualityAlert]) -> None:
        """アラートを処理"""
        for alert in alerts:
            try:
                # アラートをログに記録
                logger.warning(f"Quality Alert [{alert.severity.value.upper()}]: {alert.message}")
                
                # アラートファイルに保存
                self._save_alert(alert)
                
                # カスタムアラートハンドラーを実行
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"アラートハンドラーエラー: {e}")
                        
            except Exception as e:
                logger.error(f"アラート処理エラー: {e}")
    
    def _save_alert(self, alert: QualityAlert) -> None:
        """アラートをファイルに保存"""
        alert_file = self.output_dir / "alerts.jsonl"
        
        try:
            with open(alert_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(alert), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"アラート保存エラー: {e}")
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, QualityMetric]) -> float:
        """総合品質スコアを計算"""
        if not metrics:
            return 0.0
        
        scores = []
        weights = {
            'stability': 0.2,
            'information_gain': 0.2,
            'redundancy': 0.15,
            'correlation': 0.15,
            'distribution_shift': 0.1,
            'missing_values': 0.1,
            'outliers': 0.05,
            'variance': 0.05
        }
        
        for metric_name, metric in metrics.items():
            weight = weights.get(metric_name, 1.0 / len(metrics))
            
            # メトリクスを0-1スケールに正規化
            if metric_name in ['redundancy', 'missing_values', 'outliers', 'distribution_shift']:
                # 低い方が良いメトリクス
                score = max(0.0, 1.0 - metric.value)
            else:
                # 高い方が良いメトリクス
                score = min(1.0, metric.value)
            
            scores.append(score * weight)
        
        return sum(scores)
    
    def _generate_recommendations(self, 
                                metrics: Dict[str, QualityMetric],
                                quality_changes: Dict[str, float]) -> List[str]:
        """品質改善の推奨事項を生成"""
        recommendations = []
        
        # 品質が低いメトリクスを特定
        low_quality_metrics = []
        for metric_name, metric in metrics.items():
            if metric.value < metric.threshold:
                low_quality_metrics.append((metric_name, metric.value, metric.threshold))
        
        # 品質が大幅に悪化したメトリクスを特定
        degraded_metrics = []
        for metric_name, change in quality_changes.items():
            if change < -0.1:  # 10%以上の悪化
                degraded_metrics.append((metric_name, change))
        
        # 推奨事項を生成
        if low_quality_metrics:
            recommendations.append(
                f"以下のメトリクスが閾値を下回っています: "
                f"{', '.join([f'{name}({value:.3f}<{threshold:.3f})' for name, value, threshold in low_quality_metrics])}"
            )
        
        if degraded_metrics:
            recommendations.append(
                f"以下のメトリクスが大幅に悪化しています: "
                f"{', '.join([f'{name}({change:.1%})' for name, change in degraded_metrics])}"
            )
        
        # 具体的な改善アクション
        if any(name == 'redundancy' for name, _, _ in low_quality_metrics):
            recommendations.append("冗長な特徴量を削除または次元削減を適用してください")
        
        if any(name == 'stability' for name, _ in degraded_metrics):
            recommendations.append("データパイプラインの安定性を確認してください")
        
        if any(name == 'information_gain' for name, _, _ in low_quality_metrics):
            recommendations.append("特徴量エンジニアリングを見直してください")
        
        return recommendations
    
    def _cleanup_historical_data(self) -> None:
        """古い履歴データを削除"""
        if not self.historical_metrics:
            return
        
        window_hours = self.config.get('historical_window_hours', 24)
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        
        # 古いデータを削除
        self.historical_metrics = [
            entry for entry in self.historical_metrics
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
    
    def _save_baseline_metrics(self) -> None:
        """ベースラインメトリクスを保存"""
        baseline_file = self.output_dir / "baseline_metrics.json"
        
        try:
            baseline_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    name: asdict(metric) for name, metric in self.baseline_metrics.items()
                }
            }
            
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"ベースラインメトリクス保存完了: {baseline_file}")
            
        except Exception as e:
            logger.error(f"ベースライン保存エラー: {e}")
    
    def generate_quality_report(self, 
                              include_visualizations: bool = True,
                              report_format: str = 'html') -> str:
        """品質レポートを生成
        
        Args:
            include_visualizations: 可視化を含めるかどうか
            report_format: レポート形式 ('html', 'json', 'txt')
            
        Returns:
            生成されたレポートファイルのパス
        """
        if not self.is_monitoring:
            raise ValueError("監視が開始されていません")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"quality_report_{timestamp}.{report_format}"
        
        try:
            if report_format == 'html':
                self._generate_html_report(report_file, include_visualizations)
            elif report_format == 'json':
                self._generate_json_report(report_file)
            elif report_format == 'txt':
                self._generate_text_report(report_file)
            else:
                raise ValueError(f"サポートされていないレポート形式: {report_format}")
            
            logger.info(f"品質レポート生成完了: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            raise
    
    def _generate_html_report(self, report_file: Path, include_visualizations: bool) -> None:
        """HTMLレポートを生成"""
        html_content = self._build_html_report_content(include_visualizations)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _build_html_report_content(self, include_visualizations: bool) -> str:
        """HTMLレポートコンテンツを構築"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 最新のメトリクス
        latest_metrics = self.historical_metrics[-1]['metrics'] if self.historical_metrics else {}
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>特徴量品質監視レポート</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-good {{ background-color: #d4edda; }}
                .metric-warning {{ background-color: #fff3cd; }}
                .metric-danger {{ background-color: #f8d7da; }}
                .alert {{ padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; background-color: #f8f9fa; }}
                .alert-critical {{ border-left-color: #dc3545; }}
                .alert-high {{ border-left-color: #fd7e14; }}
                .alert-medium {{ border-left-color: #ffc107; }}
                .alert-low {{ border-left-color: #28a745; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>特徴量品質監視レポート</h1>
                <p>生成日時: {current_time}</p>
                <p>監視開始: {self.monitoring_start_time.strftime("%Y-%m-%d %H:%M:%S") if self.monitoring_start_time else "N/A"}</p>
                <p>アクティブアラート数: {len(self.active_alerts)}</p>
            </div>
        """
        
        # 現在の品質メトリクス
        if latest_metrics:
            html += "<h2>現在の品質メトリクス</h2>\n<table>\n<tr><th>メトリクス</th><th>現在値</th><th>閾値</th><th>状態</th></tr>\n"
            
            for metric_name, metric in latest_metrics.items():
                status = "良好" if metric.value >= metric.threshold else "要注意"
                status_class = "metric-good" if metric.value >= metric.threshold else "metric-warning"
                
                html += f"""
                <tr class="{status_class}">
                    <td>{metric_name}</td>
                    <td>{metric.value:.3f}</td>
                    <td>{metric.threshold:.3f}</td>
                    <td>{status}</td>
                </tr>
                """
            
            html += "</table>\n"
        
        # アクティブアラート
        if self.active_alerts:
            html += "<h2>アクティブアラート</h2>\n"
            
            for alert in self.active_alerts[-10:]:  # 最新10件
                severity_class = f"alert alert-{alert.severity.value}"
                html += f"""
                <div class="{severity_class}">
                    <strong>[{alert.severity.value.upper()}]</strong> {alert.message}
                    <br><small>時刻: {alert.timestamp}</small>
                </div>
                """
        
        # 品質トレンド（簡略化）
        if len(self.historical_metrics) > 1:
            html += "<h2>品質トレンド</h2>\n"
            html += "<p>過去24時間の品質変化を表示しています。</p>\n"
            
            # 簡単なトレンド表示
            if include_visualizations:
                trend_chart_path = self._generate_trend_chart()
                if trend_chart_path:
                    html += f'<img src="{trend_chart_path}" alt="品質トレンドチャート" style="max-width: 100%;">\n'
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_json_report(self, report_file: Path) -> None:
        """JSONレポートを生成"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_start_time': self.monitoring_start_time.isoformat() if self.monitoring_start_time else None,
            'baseline_metrics': {
                name: asdict(metric) for name, metric in self.baseline_metrics.items()
            },
            'historical_metrics': self.historical_metrics,
            'active_alerts': [asdict(alert) for alert in self.active_alerts],
            'summary': {
                'total_monitoring_time_hours': (
                    (datetime.now() - self.monitoring_start_time).total_seconds() / 3600
                    if self.monitoring_start_time else 0
                ),
                'total_alerts': len(self.active_alerts),
                'latest_quality_score': (
                    self._calculate_overall_quality_score(self.historical_metrics[-1]['metrics'])
                    if self.historical_metrics else 0
                )
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    def _generate_text_report(self, report_file: Path) -> None:
        """テキストレポートを生成"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = [
            "特徴量品質監視レポート",
            "=" * 50,
            f"生成日時: {current_time}",
            f"監視開始: {self.monitoring_start_time.strftime('%Y-%m-%d %H:%M:%S') if self.monitoring_start_time else 'N/A'}",
            f"アクティブアラート数: {len(self.active_alerts)}",
            ""
        ]
        
        # 最新メトリクス
        if self.historical_metrics:
            latest_metrics = self.historical_metrics[-1]['metrics']
            content.extend([
                "現在の品質メトリクス:",
                "-" * 30
            ])
            
            for metric_name, metric in latest_metrics.items():
                status = "OK" if metric.value >= metric.threshold else "NG"
                content.append(f"{metric_name}: {metric.value:.3f} (閾値: {metric.threshold:.3f}) [{status}]")
            
            content.append("")
        
        # アクティブアラート
        if self.active_alerts:
            content.extend([
                "アクティブアラート:",
                "-" * 30
            ])
            
            for alert in self.active_alerts[-5:]:  # 最新5件
                content.append(f"[{alert.severity.value.upper()}] {alert.message}")
            
            content.append("")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
    
    def _generate_trend_chart(self) -> Optional[str]:
        """トレンドチャートを生成"""
        if len(self.historical_metrics) < 2:
            return None
        
        try:
            # 時系列データを準備
            timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in self.historical_metrics]
            
            # 主要メトリクスのトレンドをプロット
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('品質メトリクストレンド')
            
            metrics_to_plot = ['stability', 'information_gain', 'redundancy', 'correlation']
            
            for idx, metric_name in enumerate(metrics_to_plot):
                ax = axes[idx // 2, idx % 2]
                
                values = []
                for entry in self.historical_metrics:
                    if metric_name in entry['metrics']:
                        values.append(entry['metrics'][metric_name].value)
                    else:
                        values.append(None)
                
                # Noneを除去
                valid_data = [(t, v) for t, v in zip(timestamps, values) if v is not None]
                if valid_data:
                    times, vals = zip(*valid_data)
                    ax.plot(times, vals, marker='o')
                    ax.set_title(metric_name)
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # チャートを保存
            chart_path = self.output_dir / f"trend_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            logger.warning(f"トレンドチャート生成エラー: {e}")
            return None
    
    def add_alert_handler(self, handler: Callable[[QualityAlert], None]) -> 'FeatureQualityMonitor':
        """カスタムアラートハンドラーを追加
        
        Args:
            handler: アラートを処理するコールバック関数
            
        Returns:
            自身のインスタンス
        """
        self.alert_handlers.append(handler)
        logger.info("アラートハンドラーを追加しました")
        return self
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状態を取得"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitoring_start_time': self.monitoring_start_time.isoformat() if self.monitoring_start_time else None,
            'total_monitoring_time_hours': (
                (datetime.now() - self.monitoring_start_time).total_seconds() / 3600
                if self.monitoring_start_time else 0
            ),
            'baseline_metrics_count': len(self.baseline_metrics),
            'historical_records_count': len(self.historical_metrics),
            'active_alerts_count': len(self.active_alerts),
            'latest_quality_score': (
                self._calculate_overall_quality_score(self.historical_metrics[-1]['metrics'])
                if self.historical_metrics else None
            ),
            'output_directory': str(self.output_dir)
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """監視を停止
        
        Returns:
            最終監視レポート
        """
        if not self.is_monitoring:
            logger.warning("監視は既に停止しています")
            return {}
        
        # 最終レポートを生成
        final_report_path = self.generate_quality_report(
            include_visualizations=True,
            report_format='html'
        )
        
        # 監視状態をリセット
        monitoring_duration = (datetime.now() - self.monitoring_start_time).total_seconds() / 3600
        
        final_summary = {
            'monitoring_stopped_at': datetime.now().isoformat(),
            'total_monitoring_duration_hours': monitoring_duration,
            'total_alerts_generated': len(self.active_alerts),
            'final_report_path': final_report_path,
            'historical_records_count': len(self.historical_metrics)
        }
        
        self.is_monitoring = False
        self.current_stage = None
        
        logger.info(f"品質監視停止: {monitoring_duration:.2f}時間の監視完了")
        return final_summary
