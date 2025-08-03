"""
GAT特徴量統合最適化器
==================

基本特徴量とGAT特徴量の最適な組み合わせ探索、冗長性除去、動的次元調整を実装します。
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .gat_interpreter import GATInterpreter
from .gat_optimizer import GATOptimizer

logger = logging.getLogger(__name__)


class GATIntegratedOptimizer:
    """GAT特徴量統合最適化器
    
    基本特徴量とGAT特徴量の最適な組み合わせ探索、GAT特徴量の冗長性除去と情報量最大化、
    GAT特徴量の動的次元調整を実装。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（統合最適化手法、パラメータなど）
        """
        self.config = config or {}
        
        # 統合最適化設定
        self.integration_config = self.config.get('integration_config', {
            'combination_search': {
                'search_strategy': 'exhaustive',  # 'exhaustive', 'greedy', 'genetic'
                'max_features': 100,              # 最大特徴量数
                'performance_metric': 'r2',       # 'r2', 'mse', 'cross_val'
                'cv_folds': 5                     # クロスバリデーション分割数
            },
            'redundancy_removal': {
                'correlation_threshold': 0.8,     # 冗長性判定の相関閾値
                'mutual_info_threshold': 0.1,     # 相互情報量の閾値
                'variance_threshold': 0.01,       # 分散の最小閾値
                'method': 'correlation_first'     # 'correlation_first', 'mutual_info_first'
            },
            'dimension_adjustment': {
                'adjustment_strategy': 'performance_based',  # 'performance_based', 'information_based'
                'target_reduction': 0.3,          # 目標削減率
                'min_dimensions': 8,              # 最小次元数
                'performance_tolerance': 0.05     # 性能低下許容率
            }
        })
        
        # 初期化
        self.gat_optimizer = None
        self.gat_interpreter = None
        self.basic_features = None
        self.gat_features = None
        self.target_values = None
        self.optimization_results = {}
        self.is_fitted = False
        
        logger.info("GATIntegratedOptimizer初期化完了")
    
    def load_components(self, gat_optimizer: GATOptimizer, 
                       gat_interpreter: GATInterpreter) -> 'GATIntegratedOptimizer':
        """GAT最適化器と解釈器を読み込み
        
        Args:
            gat_optimizer: GAT最適化器インスタンス
            gat_interpreter: GAT解釈器インスタンス
            
        Returns:
            自身のインスタンス
        """
        self.gat_optimizer = gat_optimizer
        self.gat_interpreter = gat_interpreter
        
        logger.info("GAT最適化コンポーネント読み込み完了")
        return self
    
    def load_features(self, basic_features: Union[np.ndarray, Dict[str, np.ndarray]],
                     gat_features: Union[np.ndarray, Dict[str, np.ndarray]],
                     target_values: Optional[np.ndarray] = None,
                     node_type: str = 'dev') -> 'GATIntegratedOptimizer':
        """基本特徴量とGAT特徴量を読み込み
        
        Args:
            basic_features: 基本特徴量データ
            gat_features: GAT特徴量データ
            target_values: ターゲット値（性能評価用、オプション）
            node_type: ノードタイプ（辞書形式の場合のキー）
            
        Returns:
            自身のインスタンス
        """
        try:
            # 特徴量データの処理
            if isinstance(basic_features, dict):
                self.basic_features = basic_features.get(node_type, np.array([]))
            else:
                self.basic_features = basic_features
            
            if isinstance(gat_features, dict):
                self.gat_features = gat_features.get(node_type, np.array([]))
            else:
                self.gat_features = gat_features
            
            self.target_values = target_values
            
            # データの整合性チェック
            if self.basic_features.shape[0] != self.gat_features.shape[0]:
                raise ValueError(f"基本特徴量とGAT特徴量のサンプル数が一致しません: "
                               f"{self.basic_features.shape[0]} vs {self.gat_features.shape[0]}")
            
            if target_values is not None and len(target_values) != self.basic_features.shape[0]:
                raise ValueError(f"ターゲット値とサンプル数が一致しません: "
                               f"{len(target_values)} vs {self.basic_features.shape[0]}")
            
            self.is_fitted = True
            logger.info(f"特徴量読み込み完了: 基本={self.basic_features.shape}, GAT={self.gat_features.shape}")
            return self
            
        except Exception as e:
            logger.error(f"特徴量読み込みエラー: {e}")
            raise
    
    def find_optimal_combination(self, basic_feature_names: Optional[List[str]] = None,
                               gat_feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """基本特徴量とGAT特徴量の最適な組み合わせ探索
        
        Args:
            basic_feature_names: 基本特徴量の名前リスト
            gat_feature_names: GAT特徴量の名前リスト
            
        Returns:
            最適組み合わせ探索結果
        """
        if not self.is_fitted:
            raise ValueError("特徴量データが読み込まれていません。")
        
        config = self.integration_config['combination_search']
        
        # デフォルトの特徴量名を生成
        if basic_feature_names is None:
            basic_feature_names = [f'basic_feature_{i}' for i in range(self.basic_features.shape[1])]
        if gat_feature_names is None:
            gat_feature_names = [f'gat_feature_{i}' for i in range(self.gat_features.shape[1])]
        
        results = {
            'search_strategy': config['search_strategy'],
            'basic_features_count': self.basic_features.shape[1],
            'gat_features_count': self.gat_features.shape[1],
            'combination_results': {},
            'optimal_combination': None,
            'performance_comparison': {}
        }
        
        # 全特徴量を結合
        all_features = np.hstack([self.basic_features, self.gat_features])
        all_feature_names = basic_feature_names + gat_feature_names
        
        # 探索戦略に応じた最適化
        if config['search_strategy'] == 'exhaustive':
            combination_results = self._exhaustive_combination_search(
                all_features, all_feature_names, config
            )
        elif config['search_strategy'] == 'greedy':
            combination_results = self._greedy_combination_search(
                all_features, all_feature_names, config
            )
        elif config['search_strategy'] == 'genetic':
            combination_results = self._genetic_combination_search(
                all_features, all_feature_names, config
            )
        else:
            raise ValueError(f"未知の探索戦略: {config['search_strategy']}")
        
        results['combination_results'] = combination_results
        
        # 最適組み合わせの選択
        if combination_results:
            best_combination = max(combination_results, 
                                 key=lambda x: combination_results[x]['performance_score'])
            results['optimal_combination'] = {
                'combination_id': best_combination,
                'selected_features': combination_results[best_combination]['selected_features'],
                'performance_score': combination_results[best_combination]['performance_score'],
                'feature_count': len(combination_results[best_combination]['selected_features'])
            }
        
        # 性能比較
        results['performance_comparison'] = self._compare_feature_combinations(
            basic_feature_names, gat_feature_names, results.get('optimal_combination')
        )
        
        self.optimization_results['combination_search'] = results
        logger.info(f"最適組み合わせ探索完了: {config['search_strategy']} 戦略")
        
        return results
    
    def _exhaustive_combination_search(self, features: np.ndarray, 
                                     feature_names: List[str],
                                     config: Dict[str, Any]) -> Dict[str, Any]:
        """全探索による組み合わせ探索"""
        if self.target_values is None:
            logger.warning("ターゲット値がないため、相関ベースの評価を使用")
            return self._correlation_based_search(features, feature_names, config)
        
        from itertools import combinations
        
        max_features = min(config['max_features'], len(feature_names))
        results = {}
        
        # 特徴量数を段階的に増やして探索
        for n_features in range(1, min(max_features + 1, 21)):  # 計算量制限のため上限設定
            logger.info(f"特徴量数 {n_features} の組み合わせを探索中...")
            
            best_score = -float('inf')
            best_combination = None
            
            # すべての組み合わせをテスト（計算量制限あり）
            combinations_tested = 0
            max_combinations = 1000  # 計算量制限
            
            for feature_indices in combinations(range(len(feature_names)), n_features):
                if combinations_tested >= max_combinations:
                    break
                
                selected_features = features[:, list(feature_indices)]
                selected_names = [feature_names[i] for i in feature_indices]
                
                # 性能評価
                score = self._evaluate_feature_combination(selected_features)
                
                if score > best_score:
                    best_score = score
                    best_combination = {
                        'selected_features': selected_names,
                        'performance_score': score,
                        'feature_indices': list(feature_indices)
                    }
                
                combinations_tested += 1
            
            if best_combination:
                results[f'n_features_{n_features}'] = best_combination
        
        return results
    
    def _greedy_combination_search(self, features: np.ndarray, 
                                 feature_names: List[str],
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """貪欲法による組み合わせ探索"""
        if self.target_values is None:
            logger.warning("ターゲット値がないため、相関ベースの評価を使用")
            return self._correlation_based_search(features, feature_names, config)
        
        max_features = min(config['max_features'], len(feature_names))
        
        selected_indices = []
        remaining_indices = list(range(len(feature_names)))
        results = {}
        
        best_score = -float('inf')
        
        for step in range(max_features):
            if not remaining_indices:
                break
            
            best_addition = None
            best_addition_score = -float('inf')
            
            # 残りの特徴量から最良の追加候補を探索
            for candidate_idx in remaining_indices:
                test_indices = selected_indices + [candidate_idx]
                test_features = features[:, test_indices]
                
                score = self._evaluate_feature_combination(test_features)
                
                if score > best_addition_score:
                    best_addition_score = score
                    best_addition = candidate_idx
            
            # 最良の特徴量を追加
            if best_addition is not None:
                selected_indices.append(best_addition)
                remaining_indices.remove(best_addition)
                
                selected_names = [feature_names[i] for i in selected_indices]
                
                results[f'step_{step + 1}'] = {
                    'selected_features': selected_names,
                    'performance_score': best_addition_score,
                    'feature_indices': selected_indices.copy()
                }
                
                # 性能が向上しなくなったら停止
                if best_addition_score <= best_score:
                    logger.info(f"性能向上が見られないため、ステップ {step + 1} で探索終了")
                    break
                
                best_score = best_addition_score
                logger.info(f"ステップ {step + 1}: 特徴量 '{feature_names[best_addition]}' を追加 (スコア: {best_addition_score:.4f})")
        
        return results
    
    def _genetic_combination_search(self, features: np.ndarray, 
                                  feature_names: List[str],
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """遺伝的アルゴリズムによる組み合わせ探索"""
        if self.target_values is None:
            logger.warning("ターゲット値がないため、相関ベースの評価を使用")
            return self._correlation_based_search(features, feature_names, config)
        
        # 簡易版遺伝的アルゴリズム
        population_size = 50
        generations = 20
        mutation_rate = 0.1
        
        n_features = len(feature_names)
        max_selected = min(config['max_features'], n_features)
        
        # 初期個体群の生成
        population = []
        for _ in range(population_size):
            # 各個体は特徴量選択のバイナリマスク
            individual = np.random.choice([0, 1], size=n_features, 
                                        p=[0.7, 0.3])  # 30%の確率で特徴量を選択
            
            # 最大特徴量数制限
            selected_count = np.sum(individual)
            if selected_count > max_selected:
                # ランダムに特徴量を削除
                selected_indices = np.where(individual == 1)[0]
                to_remove = np.random.choice(selected_indices, 
                                           size=selected_count - max_selected, 
                                           replace=False)
                individual[to_remove] = 0
            
            # 最低1つの特徴量は選択
            if np.sum(individual) == 0:
                individual[np.random.randint(n_features)] = 1
            
            population.append(individual)
        
        results = {}
        
        # 世代の進化
        for generation in range(generations):
            # 適応度評価
            fitness_scores = []
            for individual in population:
                selected_indices = np.where(individual == 1)[0]
                if len(selected_indices) == 0:
                    fitness_scores.append(-float('inf'))
                else:
                    selected_features = features[:, selected_indices]
                    score = self._evaluate_feature_combination(selected_features)
                    fitness_scores.append(score)
            
            fitness_scores = np.array(fitness_scores)
            
            # 最良個体の記録
            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            best_score = fitness_scores[best_idx]
            
            selected_indices = np.where(best_individual == 1)[0]
            selected_names = [feature_names[i] for i in selected_indices]
            
            results[f'generation_{generation + 1}'] = {
                'selected_features': selected_names,
                'performance_score': best_score,
                'feature_indices': selected_indices.tolist()
            }
            
            # 選択・交叉・突然変異
            new_population = []
            
            # エリート保存
            elite_count = population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 残りの個体を生成
            while len(new_population) < population_size:
                # トーナメント選択
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # 交叉
                child = self._crossover(parent1, parent2)
                
                # 突然変異
                if np.random.random() < mutation_rate:
                    child = self._mutate(child, max_selected)
                
                new_population.append(child)
            
            population = new_population
            
            logger.info(f"世代 {generation + 1}: 最良スコア = {best_score:.4f}, 特徴量数 = {len(selected_indices)}")
        
        return results
    
    def _tournament_selection(self, population: List[np.ndarray], 
                            fitness_scores: np.ndarray, 
                            tournament_size: int = 3) -> np.ndarray:
        """トーナメント選択"""
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_scores = fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """一点交叉"""
        crossover_point = np.random.randint(1, len(parent1))
        child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return child
    
    def _mutate(self, individual: np.ndarray, max_selected: int) -> np.ndarray:
        """突然変異"""
        mutated = individual.copy()
        
        # ランダムに1つのビットを反転
        mutation_point = np.random.randint(len(individual))
        mutated[mutation_point] = 1 - mutated[mutation_point]
        
        # 最大特徴量数制限
        selected_count = np.sum(mutated)
        if selected_count > max_selected:
            selected_indices = np.where(mutated == 1)[0]
            to_remove = np.random.choice(selected_indices, 
                                       size=selected_count - max_selected, 
                                       replace=False)
            mutated[to_remove] = 0
        
        # 最低1つの特徴量は選択
        if np.sum(mutated) == 0:
            mutated[np.random.randint(len(mutated))] = 1
        
        return mutated
    
    def _correlation_based_search(self, features: np.ndarray, 
                                feature_names: List[str],
                                config: Dict[str, Any]) -> Dict[str, Any]:
        """相関ベースの特徴量選択（ターゲット値がない場合）"""
        # 特徴量間の相関分析に基づく選択
        correlation_matrix = np.corrcoef(features.T)
        
        # 分散による特徴量選択
        variances = np.var(features, axis=0)
        variance_threshold = self.integration_config['redundancy_removal']['variance_threshold']
        
        valid_features = variances >= variance_threshold
        
        results = {
            'correlation_based_selection': {
                'selected_features': [feature_names[i] for i in range(len(feature_names)) if valid_features[i]],
                'performance_score': float(np.sum(valid_features)) / len(feature_names),  # 選択された特徴量の割合
                'feature_indices': np.where(valid_features)[0].tolist()
            }
        }
        
        return results
    
    def _evaluate_feature_combination(self, features: np.ndarray) -> float:
        """特徴量組み合わせの性能評価"""
        if self.target_values is None:
            # ターゲット値がない場合は分散の合計を使用
            return float(np.sum(np.var(features, axis=0)))
        
        config = self.integration_config['combination_search']
        
        try:
            if config['performance_metric'] == 'r2':
                # R²スコアによる評価
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                
                if config.get('cv_folds', 0) > 1:
                    scores = cross_val_score(model, features, self.target_values, 
                                           cv=config['cv_folds'], scoring='r2')
                    return float(np.mean(scores))
                else:
                    model.fit(features, self.target_values)
                    predictions = model.predict(features)
                    return float(r2_score(self.target_values, predictions))
                    
            elif config['performance_metric'] == 'mse':
                # 平均二乗誤差による評価（負の値を返す、大きいほど良い）
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                
                if config.get('cv_folds', 0) > 1:
                    scores = cross_val_score(model, features, self.target_values, 
                                           cv=config['cv_folds'], scoring='neg_mean_squared_error')
                    return float(np.mean(scores))
                else:
                    model.fit(features, self.target_values)
                    predictions = model.predict(features)
                    return float(-mean_squared_error(self.target_values, predictions))
            
            else:
                # デフォルトは相関係数の絶対値の平均
                correlations = [np.corrcoef(features[:, i], self.target_values)[0, 1] 
                               for i in range(features.shape[1])]
                return float(np.mean(np.abs(correlations)))
                
        except Exception as e:
            logger.warning(f"性能評価エラー: {e}")
            return -float('inf')
    
    def _compare_feature_combinations(self, basic_feature_names: List[str],
                                    gat_feature_names: List[str],
                                    optimal_combination: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """特徴量組み合わせの性能比較"""
        comparison = {}
        
        try:
            # 基本特徴量のみ
            basic_score = self._evaluate_feature_combination(self.basic_features)
            comparison['basic_features_only'] = {
                'score': basic_score,
                'feature_count': self.basic_features.shape[1],
                'features': basic_feature_names
            }
            
            # GAT特徴量のみ
            gat_score = self._evaluate_feature_combination(self.gat_features)
            comparison['gat_features_only'] = {
                'score': gat_score,
                'feature_count': self.gat_features.shape[1],
                'features': gat_feature_names
            }
            
            # 全特徴量
            all_features = np.hstack([self.basic_features, self.gat_features])
            all_score = self._evaluate_feature_combination(all_features)
            comparison['all_features'] = {
                'score': all_score,
                'feature_count': all_features.shape[1],
                'features': basic_feature_names + gat_feature_names
            }
            
            # 最適組み合わせ（存在する場合）
            if optimal_combination:
                comparison['optimal_combination'] = optimal_combination
                
                # 性能向上率の計算
                baseline_score = max(basic_score, gat_score)
                if baseline_score != 0:
                    improvement = (optimal_combination['performance_score'] - baseline_score) / abs(baseline_score)
                    comparison['performance_improvement'] = {
                        'improvement_rate': improvement,
                        'baseline_score': baseline_score,
                        'optimal_score': optimal_combination['performance_score']
                    }
            
        except Exception as e:
            logger.error(f"性能比較エラー: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def remove_redundant_features(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """GAT特徴量の冗長性除去と情報量最大化
        
        Args:
            feature_names: 特徴量名のリスト
            
        Returns:
            冗長性除去結果
        """
        if not self.is_fitted:
            raise ValueError("特徴量データが読み込まれていません。")
        
        config = self.integration_config['redundancy_removal']
        
        # デフォルトの特徴量名を生成
        if feature_names is None:
            feature_names = [f'gat_feature_{i}' for i in range(self.gat_features.shape[1])]
        
        results = {
            'original_feature_count': self.gat_features.shape[1],
            'redundancy_removal_method': config['method'],
            'removed_features': [],
            'remaining_features': [],
            'redundancy_analysis': {}
        }
        
        # 冗長性分析
        results['redundancy_analysis'] = self._analyze_feature_redundancy_detailed(
            self.gat_features, feature_names, config
        )
        
        # 冗長性除去の実行
        if config['method'] == 'correlation_first':
            remaining_features = self._remove_redundancy_correlation_first(
                self.gat_features, feature_names, config
            )
        elif config['method'] == 'mutual_info_first':
            remaining_features = self._remove_redundancy_mutual_info_first(
                self.gat_features, feature_names, config
            )
        else:
            logger.warning(f"未知の冗長性除去手法: {config['method']}")
            remaining_features = list(range(len(feature_names)))
        
        # 結果の整理
        removed_indices = [i for i in range(len(feature_names)) if i not in remaining_features]
        
        results['removed_features'] = [feature_names[i] for i in removed_indices]
        results['remaining_features'] = [feature_names[i] for i in remaining_features]
        results['final_feature_count'] = len(remaining_features)
        results['reduction_rate'] = len(removed_indices) / len(feature_names)
        
        # 情報量の評価
        if remaining_features:
            remaining_gat_features = self.gat_features[:, remaining_features]
            results['information_analysis'] = self._analyze_information_content(
                remaining_gat_features, [feature_names[i] for i in remaining_features]
            )
        
        self.optimization_results['redundancy_removal'] = results
        logger.info(f"冗長性除去完了: {len(feature_names)} → {len(remaining_features)} 特徴量")
        
        return results
    
    def _analyze_feature_redundancy_detailed(self, features: np.ndarray,
                                           feature_names: List[str],
                                           config: Dict[str, Any]) -> Dict[str, Any]:
        """詳細な特徴量冗長性分析"""
        analysis = {}
        
        try:
            # 相関分析
            correlation_matrix = np.corrcoef(features.T)
            high_corr_pairs = []
            
            n_features = len(feature_names)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr_value = correlation_matrix[i, j]
                    if abs(corr_value) >= config['correlation_threshold']:
                        high_corr_pairs.append({
                            'feature_i': feature_names[i],
                            'feature_j': feature_names[j],
                            'correlation': float(corr_value),
                            'abs_correlation': float(abs(corr_value))
                        })
            
            analysis['correlation_analysis'] = {
                'high_correlation_pairs': high_corr_pairs,
                'n_high_correlation_pairs': len(high_corr_pairs),
                'correlation_threshold': config['correlation_threshold']
            }
            
            # 分散分析
            variances = np.var(features, axis=0)
            low_variance_features = []
            
            for i, (name, variance) in enumerate(zip(feature_names, variances)):
                if variance < config['variance_threshold']:
                    low_variance_features.append({
                        'feature_name': name,
                        'variance': float(variance),
                        'feature_index': i
                    })
            
            analysis['variance_analysis'] = {
                'low_variance_features': low_variance_features,
                'n_low_variance_features': len(low_variance_features),
                'variance_threshold': config['variance_threshold'],
                'variance_statistics': {
                    'mean': float(np.mean(variances)),
                    'std': float(np.std(variances)),
                    'min': float(np.min(variances)),
                    'max': float(np.max(variances))
                }
            }
            
            # 相互情報量分析（ターゲット値がある場合）
            if self.target_values is not None:
                mutual_info_scores = mutual_info_classif(features, self.target_values.astype(int), 
                                                       discrete_features=False)
                
                low_info_features = []
                for i, (name, score) in enumerate(zip(feature_names, mutual_info_scores)):
                    if score < config['mutual_info_threshold']:
                        low_info_features.append({
                            'feature_name': name,
                            'mutual_info_score': float(score),
                            'feature_index': i
                        })
                
                analysis['mutual_info_analysis'] = {
                    'low_info_features': low_info_features,
                    'n_low_info_features': len(low_info_features),
                    'mutual_info_threshold': config['mutual_info_threshold'],
                    'mutual_info_statistics': {
                        'mean': float(np.mean(mutual_info_scores)),
                        'std': float(np.std(mutual_info_scores)),
                        'min': float(np.min(mutual_info_scores)),
                        'max': float(np.max(mutual_info_scores))
                    }
                }
            
        except Exception as e:
            logger.error(f"冗長性分析エラー: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _remove_redundancy_correlation_first(self, features: np.ndarray,
                                           feature_names: List[str],
                                           config: Dict[str, Any]) -> List[int]:
        """相関ベースの冗長性除去"""
        correlation_matrix = np.corrcoef(features.T)
        n_features = len(feature_names)
        
        # 除去する特徴量のセット
        features_to_remove = set()
        
        # 高相関ペアの処理
        for i in range(n_features):
            if i in features_to_remove:
                continue
                
            for j in range(i + 1, n_features):
                if j in features_to_remove:
                    continue
                
                if abs(correlation_matrix[i, j]) >= config['correlation_threshold']:
                    # 分散の小さい方を除去
                    var_i = np.var(features[:, i])
                    var_j = np.var(features[:, j])
                    
                    if var_i <= var_j:
                        features_to_remove.add(i)
                    else:
                        features_to_remove.add(j)
        
        # 低分散特徴量の除去
        variances = np.var(features, axis=0)
        for i, variance in enumerate(variances):
            if variance < config['variance_threshold']:
                features_to_remove.add(i)
        
        # 残存特徴量のインデックス
        remaining_features = [i for i in range(n_features) if i not in features_to_remove]
        
        return remaining_features
    
    def _remove_redundancy_mutual_info_first(self, features: np.ndarray,
                                           feature_names: List[str],
                                           config: Dict[str, Any]) -> List[int]:
        """相互情報量ベースの冗長性除去"""
        n_features = len(feature_names)
        features_to_remove = set()
        
        # 低分散特徴量の除去
        variances = np.var(features, axis=0)
        for i, variance in enumerate(variances):
            if variance < config['variance_threshold']:
                features_to_remove.add(i)
        
        # 相互情報量による評価（ターゲット値がある場合）
        if self.target_values is not None:
            try:
                mutual_info_scores = mutual_info_classif(features, self.target_values.astype(int), 
                                                       discrete_features=False)
                
                # 低情報量特徴量の除去
                for i, score in enumerate(mutual_info_scores):
                    if score < config['mutual_info_threshold']:
                        features_to_remove.add(i)
                
                # 高相関ペアでは情報量の高い方を残す
                correlation_matrix = np.corrcoef(features.T)
                
                for i in range(n_features):
                    if i in features_to_remove:
                        continue
                        
                    for j in range(i + 1, n_features):
                        if j in features_to_remove:
                            continue
                        
                        if abs(correlation_matrix[i, j]) >= config['correlation_threshold']:
                            if mutual_info_scores[i] <= mutual_info_scores[j]:
                                features_to_remove.add(i)
                            else:
                                features_to_remove.add(j)
                                
            except Exception as e:
                logger.warning(f"相互情報量計算エラー: {e}")
                # フォールバックとして相関ベースの除去を実行
                return self._remove_redundancy_correlation_first(features, feature_names, config)
        
        else:
            # ターゲット値がない場合は相関ベースの除去
            return self._remove_redundancy_correlation_first(features, feature_names, config)
        
        remaining_features = [i for i in range(n_features) if i not in features_to_remove]
        return remaining_features
    
    def _analyze_information_content(self, features: np.ndarray, 
                                   feature_names: List[str]) -> Dict[str, Any]:
        """特徴量の情報量分析"""
        analysis = {}
        
        try:
            # 各特徴量のエントロピー計算
            entropies = []
            for i in range(features.shape[1]):
                hist, _ = np.histogram(features[:, i], bins=50, density=True)
                hist = hist + 1e-10  # ゼロ対策
                hist = hist / np.sum(hist)
                entropy = -np.sum(hist * np.log2(hist))
                entropies.append(entropy)
            
            analysis['entropy_analysis'] = {
                'entropies': entropies,
                'feature_names': feature_names,
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
                'total_information': float(np.sum(entropies))
            }
            
            # 主成分分析による情報量分析
            pca = PCA()
            pca.fit(features)
            
            analysis['pca_analysis'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_95': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)) + 1,
                'n_components_99': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99)) + 1
            }
            
        except Exception as e:
            logger.error(f"情報量分析エラー: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def adjust_dimensions_dynamically(self, target_performance: Optional[float] = None) -> Dict[str, Any]:
        """GAT特徴量の動的次元調整
        
        Args:
            target_performance: 目標性能値
            
        Returns:
            次元調整結果
        """
        if not self.is_fitted:
            raise ValueError("特徴量データが読み込まれていません。")
        
        config = self.integration_config['dimension_adjustment']
        
        results = {
            'original_dimensions': self.gat_features.shape[1],
            'adjustment_strategy': config['adjustment_strategy'],
            'dimension_reduction_results': {},
            'optimal_dimensions': None,
            'performance_analysis': {}
        }
        
        # 現在の性能をベースラインとして設定
        baseline_performance = self._evaluate_feature_combination(self.gat_features)
        results['baseline_performance'] = baseline_performance
        
        # 次元数を段階的に削減してテスト
        original_dims = self.gat_features.shape[1]
        min_dims = max(config['min_dimensions'], 1)
        target_reduction = config['target_reduction']
        
        # テストする次元数のリスト
        test_dimensions = []
        current_dims = original_dims
        
        while current_dims >= min_dims:
            test_dimensions.append(current_dims)
            current_dims = int(current_dims * (1 - target_reduction))
        
        test_dimensions.append(min_dims)
        test_dimensions = sorted(list(set(test_dimensions)), reverse=True)
        
        # 各次元数での性能評価
        for n_dims in test_dimensions:
            if n_dims >= original_dims:
                # 元の次元数以上の場合はスキップ
                continue
            
            logger.info(f"次元数 {n_dims} での性能評価中...")
            
            # 次元削減の実行
            if config['adjustment_strategy'] == 'performance_based':
                reduced_features = self._reduce_dimensions_performance_based(
                    self.gat_features, n_dims
                )
            elif config['adjustment_strategy'] == 'information_based':
                reduced_features = self._reduce_dimensions_information_based(
                    self.gat_features, n_dims
                )
            else:
                # デフォルトはPCA
                reduced_features = self._reduce_dimensions_pca(self.gat_features, n_dims)
            
            if reduced_features is None:
                continue
            
            # 削減後の性能評価
            reduced_performance = self._evaluate_feature_combination(reduced_features)
            
            # 性能低下率の計算
            performance_loss = (baseline_performance - reduced_performance) / abs(baseline_performance) if baseline_performance != 0 else 0
            
            results['dimension_reduction_results'][n_dims] = {
                'reduced_performance': reduced_performance,
                'performance_loss': performance_loss,
                'reduction_ratio': 1 - (n_dims / original_dims),
                'acceptable': performance_loss <= config['performance_tolerance']
            }
            
            logger.info(f"次元数 {n_dims}: 性能={reduced_performance:.4f}, 低下率={performance_loss:.2%}")
        
        # 最適次元数の決定
        acceptable_dimensions = [
            dims for dims, result in results['dimension_reduction_results'].items()
            if result['acceptable']
        ]
        
        if acceptable_dimensions:
            # 許容範囲内で最も次元数の少ないものを選択
            optimal_dims = min(acceptable_dimensions)
            results['optimal_dimensions'] = optimal_dims
            results['recommended_reduction'] = results['dimension_reduction_results'][optimal_dims]
        else:
            # 許容範囲内がない場合は最も性能低下の少ないものを選択
            if results['dimension_reduction_results']:
                optimal_dims = min(results['dimension_reduction_results'].keys(),
                                 key=lambda d: results['dimension_reduction_results'][d]['performance_loss'])
                results['optimal_dimensions'] = optimal_dims
                results['recommended_reduction'] = results['dimension_reduction_results'][optimal_dims]
        
        # 性能分析
        results['performance_analysis'] = self._analyze_dimension_performance_tradeoff(
            results['dimension_reduction_results']
        )
        
        self.optimization_results['dimension_adjustment'] = results
        logger.info(f"動的次元調整完了: 推奨次元数 = {results.get('optimal_dimensions', 'N/A')}")
        
        return results
    
    def _reduce_dimensions_performance_based(self, features: np.ndarray, 
                                           target_dims: int) -> Optional[np.ndarray]:
        """性能ベースの次元削減"""
        if self.target_values is None:
            # ターゲット値がない場合はPCAを使用
            return self._reduce_dimensions_pca(features, target_dims)
        
        try:
            # 特徴量選択による次元削減
            selector = SelectKBest(score_func=f_classif, k=target_dims)
            reduced_features = selector.fit_transform(features, self.target_values.astype(int))
            return reduced_features
            
        except Exception as e:
            logger.error(f"性能ベース次元削減エラー: {e}")
            return None
    
    def _reduce_dimensions_information_based(self, features: np.ndarray, 
                                           target_dims: int) -> Optional[np.ndarray]:
        """情報量ベースの次元削減"""
        try:
            # PCAによる次元削減（情報量を最大化）
            pca = PCA(n_components=target_dims)
            reduced_features = pca.fit_transform(features)
            return reduced_features
            
        except Exception as e:
            logger.error(f"情報量ベース次元削減エラー: {e}")
            return None
    
    def _reduce_dimensions_pca(self, features: np.ndarray, 
                             target_dims: int) -> Optional[np.ndarray]:
        """PCAによる次元削減"""
        try:
            pca = PCA(n_components=target_dims)
            reduced_features = pca.fit_transform(features)
            return reduced_features
            
        except Exception as e:
            logger.error(f"PCA次元削減エラー: {e}")
            return None
    
    def _analyze_dimension_performance_tradeoff(self, 
                                              reduction_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """次元数と性能のトレードオフ分析"""
        if not reduction_results:
            return {}
        
        dimensions = sorted(reduction_results.keys())
        performances = [reduction_results[d]['reduced_performance'] for d in dimensions]
        performance_losses = [reduction_results[d]['performance_loss'] for d in dimensions]
        reduction_ratios = [reduction_results[d]['reduction_ratio'] for d in dimensions]
        
        analysis = {
            'dimension_performance_correlation': float(np.corrcoef(dimensions, performances)[0, 1]),
            'performance_loss_trend': {
                'mean_loss': float(np.mean(performance_losses)),
                'max_loss': float(np.max(performance_losses)),
                'min_loss': float(np.min(performance_losses))
            },
            'efficiency_analysis': {}
        }
        
        # 効率性分析（削減率あたりの性能低下）
        for dims in dimensions:
            result = reduction_results[dims]
            efficiency = result['reduction_ratio'] / (result['performance_loss'] + 1e-10)
            analysis['efficiency_analysis'][dims] = {
                'efficiency_score': float(efficiency),
                'reduction_ratio': result['reduction_ratio'],
                'performance_loss': result['performance_loss']
            }
        
        # 最も効率的な次元数
        if analysis['efficiency_analysis']:
            most_efficient_dims = max(analysis['efficiency_analysis'].keys(),
                                    key=lambda d: analysis['efficiency_analysis'][d]['efficiency_score'])
            analysis['most_efficient_dimensions'] = most_efficient_dims
        
        return analysis
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """統合最適化結果の要約情報を取得
        
        Returns:
            統合最適化要約情報の辞書
        """
        if not self.optimization_results:
            raise ValueError("統合最適化が実行されていません。")
        
        summary = {
            'integration_completed': True,
            'results_summary': {},
            'recommendations': {}
        }
        
        # 組み合わせ探索結果の要約
        if 'combination_search' in self.optimization_results:
            combo_result = self.optimization_results['combination_search']
            
            summary['results_summary']['combination_search'] = {
                'search_strategy': combo_result['search_strategy'],
                'basic_features_count': combo_result['basic_features_count'],
                'gat_features_count': combo_result['gat_features_count'],
                'optimal_combination_found': combo_result['optimal_combination'] is not None
            }
            
            if combo_result['optimal_combination']:
                optimal = combo_result['optimal_combination']
                summary['recommendations']['optimal_feature_combination'] = {
                    'feature_count': optimal['feature_count'],
                    'performance_score': optimal['performance_score']
                }
        
        # 冗長性除去結果の要約
        if 'redundancy_removal' in self.optimization_results:
            redundancy_result = self.optimization_results['redundancy_removal']
            
            summary['results_summary']['redundancy_removal'] = {
                'original_count': redundancy_result['original_feature_count'],
                'final_count': redundancy_result['final_feature_count'],
                'reduction_rate': redundancy_result['reduction_rate']
            }
            
            summary['recommendations']['redundancy_reduction'] = {
                'recommended_features': len(redundancy_result['remaining_features']),
                'reduction_achieved': redundancy_result['reduction_rate']
            }
        
        # 次元調整結果の要約
        if 'dimension_adjustment' in self.optimization_results:
            dimension_result = self.optimization_results['dimension_adjustment']
            
            summary['results_summary']['dimension_adjustment'] = {
                'original_dimensions': dimension_result['original_dimensions'],
                'optimal_dimensions': dimension_result.get('optimal_dimensions'),
                'baseline_performance': dimension_result['baseline_performance']
            }
            
            if dimension_result.get('optimal_dimensions'):
                summary['recommendations']['optimal_dimensions'] = dimension_result['optimal_dimensions']
        
        return summary
    
    def save(self, filepath: str) -> None:
        """統合最適化器を保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        save_data = {
            'config': self.config,
            'optimization_results': self.optimization_results,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"GATIntegratedOptimizer保存完了: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GATIntegratedOptimizer':
        """統合最適化器を読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
            
        Returns:
            読み込まれたGATIntegratedOptimizerインスタンス
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        instance = cls(config=save_data['config'])
        instance.optimization_results = save_data['optimization_results']
        instance.is_fitted = save_data['is_fitted']
        
        logger.info(f"GATIntegratedOptimizer読み込み完了: {filepath}")
        return instance
    
    def generate_integration_report(self, output_path: Optional[str] = None) -> str:
        """統合最適化レポートを生成
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            
        Returns:
            生成されたレポートファイルのパス
        """
        if not self.optimization_results:
            raise ValueError("統合最適化が実行されていません。")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"gat_integration_report_{timestamp}.txt"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_integration_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("GAT特徴量統合最適化レポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 要約
            f.write("【統合最適化要約】\n")
            for key, value in summary['results_summary'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 推奨事項
            if summary['recommendations']:
                f.write("【推奨事項】\n")
                for key, value in summary['recommendations'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # 詳細結果
            f.write("【詳細統合結果】\n")
            for result_key, result_data in self.optimization_results.items():
                f.write(f"\n--- {result_key} ---\n")
                self._write_integration_details(f, result_data)
        
        logger.info(f"統合最適化レポート生成完了: {output_path}")
        return str(output_path)
    
    def _write_integration_details(self, file_handle, result_data: Dict[str, Any]) -> None:
        """統合結果詳細をファイルに書き込み"""
        for key, value in result_data.items():
            if isinstance(value, dict):
                file_handle.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (dict, list)) and len(str(sub_value)) > 100:
                        file_handle.write(f"    {sub_key}: [詳細データ - 省略]\n")
                    else:
                        file_handle.write(f"    {sub_key}: {sub_value}\n")
            elif isinstance(value, list) and len(str(value)) > 100:
                file_handle.write(f"  {key}: [データリスト - 省略 (長さ: {len(value)})]\n")
            else:
                file_handle.write(f"  {key}: {value}\n")
