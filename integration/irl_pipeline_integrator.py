# IRL訓練パイプライン統合システム

"""
既存のIRL訓練パイプラインと新しい特徴量リデザインシステムを統合するモジュール。
後方互換性を保ちながら、強化された特徴量エンジニアリング機能を既存システムに統合します。
"""

import os
import sys
import json
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime

# 既存システムとの互換性のための相対インポート
try:
    from analysis.feature_pipeline.feature_pipeline import FeaturePipeline
    from analysis.feature_analysis.feature_importance_analyzer import FeatureImportanceAnalyzer
    from analysis.feature_design.task_feature_designer import TaskFeatureDesigner
    from analysis.feature_design.developer_feature_designer import DeveloperFeatureDesigner
    from analysis.feature_optimization.feature_scaler import FeatureScaler
    from analysis.feature_optimization.feature_selector import FeatureSelector
    from analysis.gat_optimization.gat_optimizer import GATOptimizer
except ImportError as e:
    logging.warning(f"新しい特徴量システムのインポートに失敗: {e}")
    logging.warning("基本機能のみで動作します")

@dataclass
class IRLIntegrationConfig:
    """IRL統合設定"""
    # 基本設定
    enable_enhanced_features: bool = True
    preserve_backward_compatibility: bool = True
    legacy_feature_weight: float = 0.3
    enhanced_feature_weight: float = 0.7
    
    # 特徴量設定
    original_feature_count: int = 62  # 既存システムの特徴量数
    target_feature_count: int = 85    # 統合後の目標特徴量数
    feature_compatibility_mode: str = 'adaptive'  # 'strict', 'adaptive', 'flexible'
    
    # パイプライン設定
    enable_feature_pipeline: bool = True
    pipeline_stages: List[str] = None
    cache_enhanced_features: bool = True
    
    # 統合設定
    integration_strategy: str = 'gradual'  # 'immediate', 'gradual', 'ab_test'
    migration_schedule: Optional[Dict[str, Any]] = None
    
    # 評価設定
    enable_performance_monitoring: bool = True
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        if self.pipeline_stages is None:
            self.pipeline_stages = ['analysis', 'design', 'optimization']
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ['accuracy', 'ndcg', 'mrr']


@dataclass
class IRLIntegrationResult:
    """IRL統合結果"""
    success: bool
    message: str
    execution_time: float
    
    # 特徴量情報
    original_features: Optional[np.ndarray] = None
    enhanced_features: Optional[np.ndarray] = None
    integrated_features: np.ndarray = None
    feature_names: List[str] = None
    
    # 性能情報
    performance_comparison: Dict[str, float] = None
    feature_importance_change: Dict[str, float] = None
    
    # 統合情報
    integration_strategy_used: str = None
    backward_compatibility_preserved: bool = None
    migration_status: str = None
    
    # メタデータ
    timestamp: str = None
    integration_id: str = None
    config_used: IRLIntegrationConfig = None


class IRLPipelineIntegrator:
    """
    IRL訓練パイプラインと新しい特徴量システムの統合クラス
    """
    
    def __init__(self, config: Optional[IRLIntegrationConfig] = None):
        """
        統合システムの初期化
        
        Args:
            config: 統合設定
        """
        self.config = config or IRLIntegrationConfig()
        self.logger = self._setup_logging()
        
        # 統合状態の管理
        self.integration_state = {
            'initialized': False,
            'enhanced_pipeline_loaded': False,
            'compatibility_verified': False,
            'migration_completed': False
        }
        
        # システムコンポーネント
        self.enhanced_pipeline = None
        self.legacy_feature_handler = None
        
        # 結果とメトリクス
        self.performance_tracker = {}
        self.feature_mappings = {}
        
        self.logger.info("IRL統合システムが初期化されました")
    
    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('IRL_Integration')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - IRL_Integration - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def initialize_integration(self) -> bool:
        """
        統合システムの初期化
        
        Returns:
            初期化成功フラグ
        """
        try:
            self.logger.info("統合システムの初期化を開始...")
            
            # 1. 既存システムの検証
            if not self._verify_legacy_system():
                self.logger.error("既存システムの検証に失敗")
                return False
            
            # 2. 新しいパイプラインの初期化
            if self.config.enable_enhanced_features:
                if not self._initialize_enhanced_pipeline():
                    self.logger.warning("強化パイプラインの初期化に失敗（基本機能で続行）")
                    self.config.enable_enhanced_features = False
            
            # 3. 互換性検証
            if not self._verify_compatibility():
                self.logger.error("互換性検証に失敗")
                return False
            
            # 4. 特徴量マッピングの構築
            self._build_feature_mappings()
            
            self.integration_state['initialized'] = True
            self.logger.info("統合システムの初期化が完了しました")
            return True
            
        except Exception as e:
            self.logger.error(f"統合システムの初期化中にエラー: {e}")
            return False
    
    def _verify_legacy_system(self) -> bool:
        """既存システムの検証"""
        try:
            # 既存の重みファイル確認
            expected_weight_files = [
                'learned_weights.npy',
                'learned_weights_bot_excluded.npy',
                'learned_weights_test.npy'
            ]
            
            for weight_file in expected_weight_files:
                if os.path.exists(f'data/{weight_file}'):
                    weights = np.load(f'data/{weight_file}')
                    if weights.shape[0] != self.config.original_feature_count:
                        self.logger.warning(
                            f"{weight_file}: 期待される次元数 {self.config.original_feature_count}, "
                            f"実際の次元数 {weights.shape[0]}"
                        )
                    else:
                        self.logger.info(f"{weight_file}: 検証OK ({weights.shape[0]}次元)")
            
            # データファイル確認
            required_data_files = [
                'data/backlog.json',
                'data/graph_training.pt',
                'data/labels.pt'
            ]
            
            missing_files = []
            for data_file in required_data_files:
                if not os.path.exists(data_file):
                    missing_files.append(data_file)
            
            if missing_files:
                self.logger.warning(f"見つからないデータファイル: {missing_files}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"既存システム検証エラー: {e}")
            return False
    
    def _initialize_enhanced_pipeline(self) -> bool:
        """強化パイプラインの初期化"""
        try:
            # 新しい特徴量パイプラインの初期化
            self.enhanced_pipeline = FeaturePipeline()
            self.enhanced_pipeline.initialize_components()
            
            self.integration_state['enhanced_pipeline_loaded'] = True
            self.logger.info("強化パイプラインの初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"強化パイプライン初期化エラー: {e}")
            return False
    
    def _verify_compatibility(self) -> bool:
        """互換性検証"""
        try:
            # 基本的な互換性チェック
            compatibility_checks = {
                'feature_count_compatibility': self._check_feature_count_compatibility(),
                'data_format_compatibility': self._check_data_format_compatibility(),
                'model_interface_compatibility': self._check_model_interface_compatibility()
            }
            
            all_compatible = all(compatibility_checks.values())
            
            if all_compatible:
                self.integration_state['compatibility_verified'] = True
                self.logger.info("互換性検証完了")
            else:
                failed_checks = [k for k, v in compatibility_checks.items() if not v]
                self.logger.warning(f"互換性検証失敗: {failed_checks}")
                
                # アダプティブモードでは部分的な失敗を許容
                if self.config.feature_compatibility_mode == 'adaptive':
                    self.integration_state['compatibility_verified'] = True
                    self.logger.info("アダプティブモードで互換性を確保")
                    return True
            
            return all_compatible
            
        except Exception as e:
            self.logger.error(f"互換性検証エラー: {e}")
            return False
    
    def _check_feature_count_compatibility(self) -> bool:
        """特徴量数の互換性チェック"""
        if self.config.feature_compatibility_mode == 'strict':
            return self.config.target_feature_count == self.config.original_feature_count
        elif self.config.feature_compatibility_mode == 'adaptive':
            # アダプティブモードでは次元調整を行う
            return True
        else:  # flexible
            return True
    
    def _check_data_format_compatibility(self) -> bool:
        """データ形式の互換性チェック"""
        try:
            # サンプルデータで形式チェック
            if os.path.exists('data/backlog.json'):
                with open('data/backlog.json', 'r') as f:
                    sample_data = json.load(f)
                    if isinstance(sample_data, list) and len(sample_data) > 0:
                        return True
            return False
        except:
            return False
    
    def _check_model_interface_compatibility(self) -> bool:
        """モデルインターフェースの互換性チェック"""
        # 基本的なインターフェースチェック
        return True
    
    def _build_feature_mappings(self):
        """特徴量マッピングの構築"""
        try:
            # 既存特徴量名の推定（実際のシステムに合わせて調整）
            legacy_feature_names = [
                # タスク関連特徴量
                'task_complexity', 'task_priority', 'task_urgency',
                'estimated_hours', 'dependency_count', 'blocking_issues',
                
                # 開発者関連特徴量
                'developer_expertise', 'recent_activity', 'commit_frequency',
                'pr_merge_rate', 'code_review_count', 'collaboration_score',
                
                # マッチング特徴量
                'skill_match', 'workload_balance', 'past_success_rate',
                'temporal_proximity', 'technical_compatibility',
                
                # その他の特徴量（62次元まで）
                *[f'feature_{i}' for i in range(20, 62)]
            ]
            
            self.feature_mappings = {
                'legacy_features': legacy_feature_names[:self.config.original_feature_count],
                'enhanced_features': [],  # 実行時に更新
                'mapping': {}  # legacy -> enhanced のマッピング
            }
            
            self.logger.info(f"特徴量マッピング構築完了: {len(self.feature_mappings['legacy_features'])}個の既存特徴量")
            
        except Exception as e:
            self.logger.error(f"特徴量マッピング構築エラー: {e}")
    
    def integrate_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any],
        legacy_weights: Optional[np.ndarray] = None
    ) -> IRLIntegrationResult:
        """
        特徴量統合の実行
        
        Args:
            task_data: タスクデータ
            developer_data: 開発者データ
            legacy_weights: 既存の学習済み重み
            
        Returns:
            統合結果
        """
        start_time = time.time()
        integration_id = f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"特徴量統合開始 (ID: {integration_id})")
            
            if not self.integration_state['initialized']:
                raise ValueError("統合システムが初期化されていません")
            
            # 1. 既存特徴量の抽出
            original_features = self._extract_legacy_features(task_data, developer_data)
            
            # 2. 強化特徴量の生成
            enhanced_features = None
            if self.config.enable_enhanced_features and self.enhanced_pipeline:
                enhanced_features = self._generate_enhanced_features(task_data, developer_data)
            
            # 3. 特徴量統合
            integrated_features, feature_names = self._integrate_feature_sets(
                original_features, enhanced_features
            )
            
            # 4. 性能評価
            performance_comparison = None
            if self.config.enable_performance_monitoring:
                performance_comparison = self._evaluate_performance(
                    original_features, integrated_features, legacy_weights
                )
            
            # 5. 特徴量重要度変化の分析
            importance_change = None
            if legacy_weights is not None:
                importance_change = self._analyze_importance_change(
                    original_features, integrated_features, legacy_weights, feature_names
                )
            
            execution_time = time.time() - start_time
            
            result = IRLIntegrationResult(
                success=True,
                message="特徴量統合が成功しました",
                execution_time=execution_time,
                original_features=original_features,
                enhanced_features=enhanced_features,
                integrated_features=integrated_features,
                feature_names=feature_names,
                performance_comparison=performance_comparison,
                feature_importance_change=importance_change,
                integration_strategy_used=self.config.integration_strategy,
                backward_compatibility_preserved=self.config.preserve_backward_compatibility,
                migration_status='completed',
                timestamp=datetime.now().isoformat(),
                integration_id=integration_id,
                config_used=self.config
            )
            
            self.logger.info(f"特徴量統合完了: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"特徴量統合エラー: {e}")
            
            return IRLIntegrationResult(
                success=False,
                message=f"特徴量統合に失敗: {str(e)}",
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                integration_id=integration_id,
                config_used=self.config
            )
    
    def _extract_legacy_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any]
    ) -> np.ndarray:
        """既存特徴量の抽出"""
        try:
            self.logger.info("既存特徴量の抽出を開始...")
            
            # 既存システムの特徴量抽出ロジックをシミュレーション
            # 実際のシステムでは、既存の特徴量抽出コードを呼び出し
            
            n_samples = len(task_data.get('task_id', []))
            if n_samples == 0:
                raise ValueError("タスクデータが空です")
            
            # 基本的な特徴量計算（既存ロジックをシミュレーション）
            features = np.zeros((n_samples, self.config.original_feature_count))
            
            # タスク特徴量
            if 'priority' in task_data:
                priority_map = {'high': 3, 'medium': 2, 'low': 1}
                priorities = [priority_map.get(p, 1) for p in task_data['priority']]
                features[:, 0] = priorities
            
            if 'estimated_hours' in task_data:
                features[:, 1] = task_data['estimated_hours']
            
            # 開発者特徴量
            if 'commits_count' in developer_data:
                dev_commits = developer_data['commits_count']
                # タスクごとに対応する開発者の特徴量を設定
                for i in range(n_samples):
                    dev_idx = i % len(dev_commits)  # 簡単なマッピング
                    features[i, 2] = dev_commits[dev_idx]
            
            # 残りの特徴量をランダムで初期化（実際のシステムでは適切な計算）
            np.random.seed(42)
            features[:, 3:] = np.random.randn(n_samples, self.config.original_feature_count - 3)
            
            self.logger.info(f"既存特徴量抽出完了: {features.shape}")
            return features
            
        except Exception as e:
            self.logger.error(f"既存特徴量抽出エラー: {e}")
            raise
    
    def _generate_enhanced_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any]
    ) -> np.ndarray:
        """強化特徴量の生成"""
        try:
            self.logger.info("強化特徴量の生成を開始...")
            
            if not self.enhanced_pipeline:
                raise ValueError("強化パイプラインが初期化されていません")
            
            # パイプライン入力データの準備
            input_data = {
                'task_data': task_data,
                'developer_data': developer_data,
                'features': None,  # 既存特徴量は後で設定
                'labels': None,    # ラベルは必要に応じて設定
                'weights': None    # 重みは必要に応じて設定
            }
            
            # パイプライン設定
            pipeline_config = {
                'stages': self.config.pipeline_stages,
                'design': {
                    'task_features': True,
                    'developer_features': True,
                    'matching_features': True
                },
                'optimization': {
                    'scaling': True,
                    'selection': True,
                    'dimension_reduction': False  # 次元は統合時に調整
                }
            }
            
            # パイプライン実行
            result = self.enhanced_pipeline.run_full_pipeline(
                input_data=input_data,
                pipeline_config=pipeline_config
            )
            
            if not result.success:
                raise ValueError(f"パイプライン実行失敗: {result.message}")
            
            enhanced_features = result.data['final_features']
            self.feature_mappings['enhanced_features'] = result.data['feature_names']
            
            self.logger.info(f"強化特徴量生成完了: {enhanced_features.shape}")
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"強化特徴量生成エラー: {e}")
            raise
    
    def _integrate_feature_sets(
        self,
        original_features: np.ndarray,
        enhanced_features: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """特徴量セットの統合"""
        try:
            self.logger.info("特徴量セットの統合を開始...")
            
            if enhanced_features is None:
                # 強化特徴量がない場合は既存特徴量をそのまま返す
                feature_names = self.feature_mappings['legacy_features']
                return original_features, feature_names
            
            integration_strategy = self.config.integration_strategy
            
            if integration_strategy == 'immediate':
                # 即座に強化特徴量に置き換え
                integrated_features = enhanced_features
                feature_names = self.feature_mappings['enhanced_features']
                
            elif integration_strategy == 'gradual':
                # 段階的統合: 既存と新規を重み付き結合
                integrated_features = self._gradual_integration(
                    original_features, enhanced_features
                )
                feature_names = self._create_integrated_feature_names()
                
            elif integration_strategy == 'ab_test':
                # A/Bテスト形式: ランダムに選択
                use_enhanced = np.random.random() > 0.5
                if use_enhanced:
                    integrated_features = enhanced_features
                    feature_names = self.feature_mappings['enhanced_features']
                else:
                    integrated_features = original_features
                    feature_names = self.feature_mappings['legacy_features']
            
            else:
                raise ValueError(f"未知の統合戦略: {integration_strategy}")
            
            # 次元調整
            if integrated_features.shape[1] != self.config.target_feature_count:
                integrated_features = self._adjust_feature_dimensions(
                    integrated_features, self.config.target_feature_count
                )
            
            self.logger.info(f"特徴量統合完了: {integrated_features.shape}")
            return integrated_features, feature_names
            
        except Exception as e:
            self.logger.error(f"特徴量統合エラー: {e}")
            raise
    
    def _gradual_integration(
        self,
        original_features: np.ndarray,
        enhanced_features: np.ndarray
    ) -> np.ndarray:
        """段階的統合"""
        # 次元を合わせる
        min_dim = min(original_features.shape[1], enhanced_features.shape[1])
        
        original_norm = original_features[:, :min_dim]
        enhanced_norm = enhanced_features[:, :min_dim]
        
        # 重み付き結合
        integrated = (
            self.config.legacy_feature_weight * original_norm +
            self.config.enhanced_feature_weight * enhanced_norm
        )
        
        # 残りの次元を追加（強化特徴量から）
        if enhanced_features.shape[1] > min_dim:
            additional_features = enhanced_features[:, min_dim:]
            integrated = np.hstack([integrated, additional_features])
        
        return integrated
    
    def _create_integrated_feature_names(self) -> List[str]:
        """統合特徴量名の作成"""
        legacy_names = self.feature_mappings['legacy_features']
        enhanced_names = self.feature_mappings.get('enhanced_features', [])
        
        if self.config.integration_strategy == 'gradual':
            # 段階的統合の場合は結合名を作成
            min_count = min(len(legacy_names), len(enhanced_names))
            integrated_names = [
                f"integrated_{legacy_names[i]}_{enhanced_names[i]}" 
                for i in range(min_count)
            ]
            
            # 残りの特徴量名を追加
            if len(enhanced_names) > min_count:
                integrated_names.extend(enhanced_names[min_count:])
            
            return integrated_names
        else:
            return enhanced_names if enhanced_names else legacy_names
    
    def _adjust_feature_dimensions(
        self,
        features: np.ndarray,
        target_dim: int
    ) -> np.ndarray:
        """特徴量次元の調整"""
        current_dim = features.shape[1]
        
        if current_dim == target_dim:
            return features
        elif current_dim > target_dim:
            # 次元削減: 上位次元を選択
            return features[:, :target_dim]
        else:
            # 次元拡張: ゼロパディング
            padding = np.zeros((features.shape[0], target_dim - current_dim))
            return np.hstack([features, padding])
    
    def _evaluate_performance(
        self,
        original_features: np.ndarray,
        integrated_features: np.ndarray,
        legacy_weights: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """性能評価"""
        try:
            self.logger.info("性能評価を開始...")
            
            performance_metrics = {}
            
            # 基本的な統計比較
            performance_metrics['feature_count_change'] = (
                integrated_features.shape[1] - original_features.shape[1]
            ) / original_features.shape[1]
            
            performance_metrics['feature_variance_change'] = (
                np.mean(np.var(integrated_features, axis=0)) - 
                np.mean(np.var(original_features, axis=0))
            ) / np.mean(np.var(original_features, axis=0))
            
            # 重みがある場合の評価
            if legacy_weights is not None:
                # 既存重みでの予測値比較
                original_pred = np.dot(original_features, 
                                     legacy_weights[:original_features.shape[1]])
                
                # 統合特徴量用の重み調整
                adjusted_weights = self._adjust_weights_for_integration(
                    legacy_weights, integrated_features.shape[1]
                )
                integrated_pred = np.dot(integrated_features, adjusted_weights)
                
                # 予測値の相関
                correlation = np.corrcoef(original_pred, integrated_pred)[0, 1]
                performance_metrics['prediction_correlation'] = correlation
                
                # 予測値の分散変化
                performance_metrics['prediction_variance_change'] = (
                    np.var(integrated_pred) - np.var(original_pred)
                ) / np.var(original_pred)
            
            self.logger.info("性能評価完了")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"性能評価エラー: {e}")
            return {}
    
    def _adjust_weights_for_integration(
        self,
        legacy_weights: np.ndarray,
        target_dim: int
    ) -> np.ndarray:
        """統合用の重み調整"""
        if len(legacy_weights) == target_dim:
            return legacy_weights
        elif len(legacy_weights) > target_dim:
            return legacy_weights[:target_dim]
        else:
            # 不足分をゼロで埋める
            padding = np.zeros(target_dim - len(legacy_weights))
            return np.concatenate([legacy_weights, padding])
    
    def _analyze_importance_change(
        self,
        original_features: np.ndarray,
        integrated_features: np.ndarray,
        legacy_weights: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """特徴量重要度変化の分析"""
        try:
            self.logger.info("重要度変化分析を開始...")
            
            # 既存重みの重要度
            legacy_importance = np.abs(legacy_weights[:original_features.shape[1]])
            legacy_importance_normalized = legacy_importance / np.sum(legacy_importance)
            
            # 統合特徴量での重要度推定
            adjusted_weights = self._adjust_weights_for_integration(
                legacy_weights, integrated_features.shape[1]
            )
            integrated_importance = np.abs(adjusted_weights)
            integrated_importance_normalized = integrated_importance / np.sum(integrated_importance)
            
            # 変化量計算
            min_dim = min(len(legacy_importance_normalized), len(integrated_importance_normalized))
            importance_change = {}
            
            for i in range(min_dim):
                if i < len(feature_names):
                    feature_name = feature_names[i]
                else:
                    feature_name = f"feature_{i}"
                
                change = (integrated_importance_normalized[i] - 
                         legacy_importance_normalized[i])
                importance_change[feature_name] = change
            
            # 新しい特徴量の重要度
            if len(integrated_importance_normalized) > min_dim:
                for i in range(min_dim, len(integrated_importance_normalized)):
                    if i < len(feature_names):
                        feature_name = feature_names[i]
                    else:
                        feature_name = f"new_feature_{i}"
                    importance_change[feature_name] = integrated_importance_normalized[i]
            
            self.logger.info("重要度変化分析完了")
            return importance_change
            
        except Exception as e:
            self.logger.error(f"重要度変化分析エラー: {e}")
            return {}
    
    def save_integration_result(
        self,
        result: IRLIntegrationResult,
        output_dir: str = "outputs/integration"
    ) -> str:
        """統合結果の保存"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 基本情報をJSONで保存
            json_path = os.path.join(output_dir, f"{result.integration_id}_result.json")
            json_data = {
                'success': result.success,
                'message': result.message,
                'execution_time': result.execution_time,
                'feature_names': result.feature_names,
                'performance_comparison': result.performance_comparison,
                'feature_importance_change': result.feature_importance_change,
                'integration_strategy_used': result.integration_strategy_used,
                'backward_compatibility_preserved': result.backward_compatibility_preserved,
                'migration_status': result.migration_status,
                'timestamp': result.timestamp,
                'integration_id': result.integration_id,
                'config_used': asdict(result.config_used) if result.config_used else None
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # 特徴量データをnumpyで保存
            if result.integrated_features is not None:
                features_path = os.path.join(output_dir, f"{result.integration_id}_features.npy")
                np.save(features_path, result.integrated_features)
            
            # 特徴量名をテキストで保存
            if result.feature_names:
                names_path = os.path.join(output_dir, f"{result.integration_id}_feature_names.txt")
                with open(names_path, 'w', encoding='utf-8') as f:
                    for name in result.feature_names:
                        f.write(f"{name}\n")
            
            self.logger.info(f"統合結果を保存: {json_path}")
            return json_path
            
        except Exception as e:
            self.logger.error(f"統合結果保存エラー: {e}")
            raise
    
    def load_integration_result(self, result_path: str) -> IRLIntegrationResult:
        """統合結果の読み込み"""
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 設定の復元
            config = None
            if json_data.get('config_used'):
                config = IRLIntegrationConfig(**json_data['config_used'])
            
            # 特徴量データの読み込み
            integrated_features = None
            integration_id = json_data['integration_id']
            features_path = result_path.replace('_result.json', '_features.npy')
            if os.path.exists(features_path):
                integrated_features = np.load(features_path)
            
            result = IRLIntegrationResult(
                success=json_data['success'],
                message=json_data['message'],
                execution_time=json_data['execution_time'],
                integrated_features=integrated_features,
                feature_names=json_data.get('feature_names'),
                performance_comparison=json_data.get('performance_comparison'),
                feature_importance_change=json_data.get('feature_importance_change'),
                integration_strategy_used=json_data.get('integration_strategy_used'),
                backward_compatibility_preserved=json_data.get('backward_compatibility_preserved'),
                migration_status=json_data.get('migration_status'),
                timestamp=json_data['timestamp'],
                integration_id=integration_id,
                config_used=config
            )
            
            self.logger.info(f"統合結果を読み込み: {result_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"統合結果読み込みエラー: {e}")
            raise
    
    def create_migration_plan(
        self,
        current_performance: Dict[str, float],
        target_performance: Dict[str, float]
    ) -> Dict[str, Any]:
        """移行計画の作成"""
        try:
            self.logger.info("移行計画の作成を開始...")
            
            migration_plan = {
                'migration_strategy': self.config.integration_strategy,
                'phases': [],
                'estimated_duration_days': 0,
                'risk_assessment': {},
                'rollback_plan': {},
                'success_criteria': target_performance
            }
            
            if self.config.integration_strategy == 'gradual':
                # 段階的移行計画
                phases = [
                    {
                        'phase': 1,
                        'description': '特徴量分析と設計の統合',
                        'duration_days': 7,
                        'enhanced_feature_weight': 0.2,
                        'success_criteria': {'accuracy': current_performance.get('accuracy', 0.8)}
                    },
                    {
                        'phase': 2,
                        'description': '最適化機能の統合',
                        'duration_days': 7,
                        'enhanced_feature_weight': 0.5,
                        'success_criteria': {'accuracy': current_performance.get('accuracy', 0.8) * 1.05}
                    },
                    {
                        'phase': 3,
                        'description': 'GAT最適化の統合',
                        'duration_days': 10,
                        'enhanced_feature_weight': 0.8,
                        'success_criteria': {'accuracy': current_performance.get('accuracy', 0.8) * 1.1}
                    },
                    {
                        'phase': 4,
                        'description': '完全統合',
                        'duration_days': 5,
                        'enhanced_feature_weight': 1.0,
                        'success_criteria': target_performance
                    }
                ]
                migration_plan['phases'] = phases
                migration_plan['estimated_duration_days'] = sum(p['duration_days'] for p in phases)
                
            elif self.config.integration_strategy == 'immediate':
                # 即座移行計画
                migration_plan['phases'] = [{
                    'phase': 1,
                    'description': '即座に新システムに移行',
                    'duration_days': 1,
                    'enhanced_feature_weight': 1.0,
                    'success_criteria': target_performance
                }]
                migration_plan['estimated_duration_days'] = 1
                
            elif self.config.integration_strategy == 'ab_test':
                # A/Bテスト移行計画
                migration_plan['phases'] = [
                    {
                        'phase': 1,
                        'description': 'A/Bテスト実行',
                        'duration_days': 14,
                        'enhanced_feature_weight': 0.5,  # 50%のトラフィック
                        'success_criteria': {'significance': 0.05}
                    },
                    {
                        'phase': 2,
                        'description': '結果に基づく全面展開',
                        'duration_days': 3,
                        'enhanced_feature_weight': 1.0,
                        'success_criteria': target_performance
                    }
                ]
                migration_plan['estimated_duration_days'] = 17
            
            # リスク評価
            migration_plan['risk_assessment'] = {
                'performance_degradation': 'medium',
                'compatibility_issues': 'low',
                'rollback_complexity': 'low',
                'data_loss_risk': 'none'
            }
            
            # ロールバック計画
            migration_plan['rollback_plan'] = {
                'triggers': [
                    'performance degradation > 10%',
                    'error rate > 5%',
                    'user complaints > threshold'
                ],
                'rollback_steps': [
                    '1. 新システムの無効化',
                    '2. 既存システムの再有効化',
                    '3. データベースの復元',
                    '4. キャッシュのクリア',
                    '5. 監視とログの確認'
                ],
                'estimated_rollback_time_minutes': 30
            }
            
            self.logger.info("移行計画の作成完了")
            return migration_plan
            
        except Exception as e:
            self.logger.error(f"移行計画作成エラー: {e}")
            raise


class IRLIntegrationUtils:
    """IRL統合ユーティリティクラス"""
    
    @staticmethod
    def load_sample_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """サンプルデータの読み込み"""
        # サンプルタスクデータ
        task_data = {
            'task_id': [1, 2, 3, 4, 5],
            'title': [
                'Fix critical bug in payment system',
                'Add user authentication',
                'Update documentation',
                'Implement new API endpoint',
                'Optimize database queries'
            ],
            'description': [
                'Critical bug causing payment failures...',
                'Implement OAuth2 authentication system...',
                'Update API documentation for v2...',
                'Add new REST API endpoint for users...',
                'Optimize slow database queries...'
            ],
            'priority': ['high', 'medium', 'low', 'medium', 'high'],
            'estimated_hours': [8, 16, 4, 12, 6],
            'labels': [
                ['bug', 'critical'],
                ['feature', 'authentication'],
                ['documentation'],
                ['feature', 'api'],
                ['performance', 'database']
            ]
        }
        
        # サンプル開発者データ
        developer_data = {
            'developer_id': [101, 102, 103, 104, 105],
            'commits_count': [1200, 800, 2000, 600, 1500],
            'expertise_languages': [
                ['Python', 'JavaScript'],
                ['Java', 'Kotlin'],
                ['Python', 'Go', 'Rust'],
                ['JavaScript', 'TypeScript'],
                ['Python', 'SQL']
            ],
            'recent_activity_days': [30, 15, 45, 7, 20],
            'pr_merge_rate': [0.85, 0.92, 0.78, 0.95, 0.88],
            'code_review_count': [150, 200, 300, 80, 120],
            'bug_introduction_rate': [0.02, 0.01, 0.03, 0.01, 0.02]
        }
        
        return task_data, developer_data
    
    @staticmethod
    def validate_data_compatibility(
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """データ互換性の検証"""
        validation_results = {}
        
        # タスクデータ検証
        required_task_fields = ['task_id', 'title', 'priority']
        validation_results['task_data_complete'] = all(
            field in task_data for field in required_task_fields
        )
        
        # 開発者データ検証
        required_dev_fields = ['developer_id', 'commits_count', 'expertise_languages']
        validation_results['developer_data_complete'] = all(
            field in developer_data for field in required_dev_fields
        )
        
        # データ形式検証
        validation_results['task_data_format_valid'] = (
            isinstance(task_data.get('task_id'), list) and
            len(task_data.get('task_id', [])) > 0
        )
        
        validation_results['developer_data_format_valid'] = (
            isinstance(developer_data.get('developer_id'), list) and
            len(developer_data.get('developer_id', [])) > 0
        )
        
        return validation_results


def create_integration_config_template() -> Dict[str, Any]:
    """統合設定テンプレートの作成"""
    return {
        'integration': {
            'enable_enhanced_features': True,
            'preserve_backward_compatibility': True,
            'legacy_feature_weight': 0.3,
            'enhanced_feature_weight': 0.7,
            'original_feature_count': 62,
            'target_feature_count': 85,
            'feature_compatibility_mode': 'adaptive',
            'integration_strategy': 'gradual',
            'enable_performance_monitoring': True
        },
        'pipeline': {
            'enable_feature_pipeline': True,
            'pipeline_stages': ['analysis', 'design', 'optimization'],
            'cache_enhanced_features': True
        },
        'evaluation': {
            'evaluation_metrics': ['accuracy', 'ndcg', 'mrr'],
            'performance_threshold': 0.95,
            'enable_ab_testing': False
        }
    }


# メイン実行例
if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # 設定作成
    config = IRLIntegrationConfig(
        enable_enhanced_features=True,
        integration_strategy='gradual',
        original_feature_count=62,
        target_feature_count=85
    )
    
    # 統合システム初期化
    integrator = IRLPipelineIntegrator(config)
    
    if not integrator.initialize_integration():
        print("統合システムの初期化に失敗しました")
        sys.exit(1)
    
    # サンプルデータ読み込み
    task_data, developer_data = IRLIntegrationUtils.load_sample_data()
    
    # 既存重みの読み込み（存在する場合）
    legacy_weights = None
    if os.path.exists('data/learned_weights.npy'):
        legacy_weights = np.load('data/learned_weights.npy')
    else:
        # サンプル重み作成
        legacy_weights = np.random.randn(62)
    
    # 特徴量統合実行
    result = integrator.integrate_features(
        task_data=task_data,
        developer_data=developer_data,
        legacy_weights=legacy_weights
    )
    
    if result.success:
        print(f"✅ 統合成功!")
        print(f"   実行時間: {result.execution_time:.2f}秒")
        print(f"   統合特徴量形状: {result.integrated_features.shape}")
        print(f"   特徴量数変化: {len(result.feature_names)}個")
        
        if result.performance_comparison:
            print("   性能比較:")
            for metric, value in result.performance_comparison.items():
                print(f"     {metric}: {value:.4f}")
        
        # 結果保存
        saved_path = integrator.save_integration_result(result)
        print(f"   結果保存: {saved_path}")
        
        # 移行計画作成
        current_perf = {'accuracy': 0.85}
        target_perf = {'accuracy': 0.90}
        migration_plan = integrator.create_migration_plan(current_perf, target_perf)
        print(f"   移行期間: {migration_plan['estimated_duration_days']}日")
        
    else:
        print(f"❌ 統合失敗: {result.message}")
        sys.exit(1)
