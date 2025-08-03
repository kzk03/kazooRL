"""
特徴量選択器
============

単変量統計検定、再帰的特徴量除去、L1正則化による特徴量選択を実装。
重要度ベースの特徴量選択と相互作用項生成機能を提供します。
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class FeatureSelector:
    """特徴量選択器
    
    単変量統計検定（chi-square、ANOVA）による特徴量選択を実装。
    再帰的特徴量除去（RFE）とL1正則化による自動選択機能を提供。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（選択手法、パラメータなど）
        """
        self.config = config or {}
        
        # 選択手法の設定
        self.selection_methods = self.config.get('selection_methods', {
            'univariate': {
                'score_func': 'auto',  # 'chi2', 'f_classif', 'f_regression', 'mutual_info'
                'k': 'auto',           # 選択する特徴量数
                'percentile': 50       # パーセンタイル選択
            },
            'rfe': {
                'estimator': 'auto',   # 'rf', 'logistic', 'lasso'
                'n_features_to_select': 'auto',
                'step': 1,
                'cv': 5
            },
            'l1_regularization': {
                'alpha': 'auto',       # 正則化パラメータ
                'cv': 5,
                'max_iter': 1000
            },
            'importance_based': {
                'estimator': 'rf',     # 'rf', 'xgb', 'lgb'
                'threshold': 'auto',   # 重要度閾値
                'top_k': 'auto'        # 上位k個選択
            }
        })
        
        # 相互作用項生成の設定
        self.interaction_config = self.config.get('interaction_config', {
            'degree': 2,               # 多項式の次数
            'interaction_only': True,  # 相互作用項のみ
            'include_bias': False,     # バイアス項を含むか
            'max_features': 1000       # 最大特徴量数
        })
        
        # 初期化
        self.selectors = {}
        self.selected_features = {}
        self.feature_scores = {}
        self.feature_importances = {}
        self.is_fitted = False
        self.task_type = None  # 'classification' or 'regression'
        
        logger.info(f"FeatureSelector初期化完了")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            feature_names: Optional[List[str]] = None,
            task_type: Optional[str] = None,
            methods: Optional[List[str]] = None) -> 'FeatureSelector':
        """特徴量選択器を学習データに適合
        
        Args:
            X: 特徴量データ
            y: 目的変数
            feature_names: 特徴量名のリスト
            task_type: タスクタイプ ('classification' or 'regression')
            methods: 使用する選択手法のリスト
            
        Returns:
            自身のインスタンス
        """
        # データ形式の統一
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        self.feature_names = feature_names
        self.n_features_original = len(feature_names)
        
        # タスクタイプの自動判定
        if task_type:
            self.task_type = task_type
        else:
            self.task_type = self._detect_task_type(y_array)
        
        # 使用する手法の決定
        if methods is None:
            methods = ['univariate', 'rfe', 'l1_regularization', 'importance_based']
        
        # 各手法で特徴量選択を実行
        for method in methods:
            if method == 'univariate':
                self._fit_univariate_selection(X_array, y_array)
            elif method == 'rfe':
                self._fit_rfe_selection(X_array, y_array)
            elif method == 'l1_regularization':
                self._fit_l1_selection(X_array, y_array)
            elif method == 'importance_based':
                self._fit_importance_selection(X_array, y_array)
        
        self.is_fitted = True
        logger.info(f"FeatureSelector学習完了: {len(methods)}手法")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                 method: str = 'ensemble') -> np.ndarray:
        """特徴量選択を適用
        
        Args:
            X: 変換する特徴量データ
            method: 使用する選択手法 ('univariate', 'rfe', 'l1', 'importance', 'ensemble')
            
        Returns:
            選択された特徴量データ
        """
        if not self.is_fitted:
            raise ValueError("特徴量選択器が学習されていません。先にfit()を呼び出してください。")
        
        # データ形式の統一
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if X_array.shape[1] != self.n_features_original:
            raise ValueError(f"入力データの次元数({X_array.shape[1]})が学習時({self.n_features_original})と異なります")
        
        # 選択手法に応じた変換
        if method == 'ensemble':
            selected_indices = self._get_ensemble_selection()
        else:
            if method not in self.selected_features:
                raise ValueError(f"手法 '{method}' は学習されていません")
            selected_indices = self.selected_features[method]
        
        X_selected = X_array[:, selected_indices]
        
        logger.debug(f"特徴量選択変換完了: {X_array.shape[1]} → {X_selected.shape[1]} ({method})")
        return X_selected
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series],
                     feature_names: Optional[List[str]] = None,
                     task_type: Optional[str] = None,
                     methods: Optional[List[str]] = None,
                     transform_method: str = 'ensemble') -> np.ndarray:
        """学習と変換を同時実行
        
        Args:
            X: 特徴量データ
            y: 目的変数
            feature_names: 特徴量名のリスト
            task_type: タスクタイプ
            methods: 使用する選択手法のリスト
            transform_method: 変換に使用する手法
            
        Returns:
            選択された特徴量データ
        """
        return self.fit(X, y, feature_names, task_type, methods).transform(X, transform_method)
    
    def _detect_task_type(self, y: np.ndarray) -> str:
        """タスクタイプの自動判定
        
        Args:
            y: 目的変数
            
        Returns:
            タスクタイプ ('classification' or 'regression')
        """
        unique_values = np.unique(y)
        
        # 整数値で値の種類が少ない場合は分類
        if len(unique_values) <= 20 and np.all(y == y.astype(int)):
            return 'classification'
        # 連続値の場合は回帰
        else:
            return 'regression'
    
    def _fit_univariate_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """単変量統計検定による特徴量選択
        
        Args:
            X: 特徴量データ
            y: 目的変数
        """
        config = self.selection_methods['univariate']
        
        # スコア関数の自動選択
        if config['score_func'] == 'auto':
            if self.task_type == 'classification':
                # 非負値のみの場合はchi2、そうでなければf_classif
                if np.all(X >= 0):
                    score_func = chi2
                else:
                    score_func = f_classif
            else:
                score_func = f_regression
        else:
            score_func_map = {
                'chi2': chi2,
                'f_classif': f_classif,
                'f_regression': f_regression,
                'mutual_info_classif': mutual_info_classif,
                'mutual_info_regression': mutual_info_regression
            }
            score_func = score_func_map[config['score_func']]
        
        # 特徴量数の自動決定
        if config['k'] == 'auto':
            k = min(int(self.n_features_original * 0.5), 50)  # 50%または50個の小さい方
        else:
            k = config['k']
        
        # 選択器の学習
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # 結果の保存
        self.selectors['univariate'] = selector
        self.selected_features['univariate'] = selector.get_support(indices=True).tolist()
        self.feature_scores['univariate'] = selector.scores_.tolist()
        
        logger.debug(f"単変量選択完了: {len(self.selected_features['univariate'])}特徴量選択")
    
    def _fit_rfe_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """再帰的特徴量除去による選択
        
        Args:
            X: 特徴量データ
            y: 目的変数
        """
        config = self.selection_methods['rfe']
        
        # 推定器の自動選択
        if config['estimator'] == 'auto':
            if self.task_type == 'classification':
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = Lasso(random_state=42, max_iter=1000)
        else:
            estimator_map = {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42) if self.task_type == 'classification' 
                      else RandomForestRegressor(n_estimators=100, random_state=42),
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'lasso': Lasso(random_state=42, max_iter=1000)
            }
            estimator = estimator_map[config['estimator']]
        
        # 選択特徴量数の自動決定
        if config['n_features_to_select'] == 'auto':
            n_features = min(int(self.n_features_original * 0.3), 30)  # 30%または30個の小さい方
        else:
            n_features = config['n_features_to_select']
        
        # RFECVを使用して最適な特徴量数を決定
        try:
            selector = RFECV(
                estimator=estimator,
                step=config['step'],
                cv=config['cv'],
                scoring='accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error',
                n_jobs=-1
            )
            selector.fit(X, y)
        except Exception as e:
            logger.warning(f"RFECV失敗、RFEにフォールバック: {e}")
            # RFECVが失敗した場合はRFEを使用
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=config['step']
            )
            selector.fit(X, y)
        
        # 結果の保存
        self.selectors['rfe'] = selector
        self.selected_features['rfe'] = selector.get_support(indices=True).tolist()
        
        # 特徴量重要度の取得（可能な場合）
        if hasattr(selector.estimator_, 'feature_importances_'):
            selected_importances = selector.estimator_.feature_importances_
            full_importances = np.zeros(self.n_features_original)
            full_importances[self.selected_features['rfe']] = selected_importances
            self.feature_importances['rfe'] = full_importances.tolist()
        elif hasattr(selector.estimator_, 'coef_'):
            selected_coefs = np.abs(selector.estimator_.coef_).flatten()
            full_coefs = np.zeros(self.n_features_original)
            full_coefs[self.selected_features['rfe']] = selected_coefs
            self.feature_importances['rfe'] = full_coefs.tolist()
        
        logger.debug(f"RFE選択完了: {len(self.selected_features['rfe'])}特徴量選択")
    
    def _fit_l1_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """L1正則化による特徴量選択
        
        Args:
            X: 特徴量データ
            y: 目的変数
        """
        config = self.selection_methods['l1_regularization']
        
        # L1正則化モデルの学習
        if self.task_type == 'classification':
            if config['alpha'] == 'auto':
                # LogisticRegressionCVは存在しないため、手動でCV
                alphas = np.logspace(-4, 1, 20)
                best_alpha = None
                best_score = -np.inf
                
                for alpha in alphas:
                    model = LogisticRegression(
                        penalty='l1', 
                        C=1/alpha, 
                        solver='liblinear',
                        random_state=42,
                        max_iter=config['max_iter']
                    )
                    try:
                        scores = cross_val_score(model, X, y, cv=config['cv'], scoring='accuracy')
                        score = np.mean(scores)
                        if score > best_score:
                            best_score = score
                            best_alpha = alpha
                    except:
                        continue
                
                alpha = best_alpha if best_alpha is not None else 0.01
            else:
                alpha = config['alpha']
            
            model = LogisticRegression(
                penalty='l1',
                C=1/alpha,
                solver='liblinear',
                random_state=42,
                max_iter=config['max_iter']
            )
        else:
            if config['alpha'] == 'auto':
                model = LassoCV(
                    cv=config['cv'],
                    random_state=42,
                    max_iter=config['max_iter']
                )
            else:
                model = Lasso(
                    alpha=config['alpha'],
                    random_state=42,
                    max_iter=config['max_iter']
                )
        
        # モデルの学習
        model.fit(X, y)
        
        # 非ゼロ係数の特徴量を選択
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = np.abs(coefs).max(axis=0)  # 多クラス分類の場合
            else:
                coefs = np.abs(coefs)
            
            selected_indices = np.where(coefs > 1e-8)[0].tolist()  # 実質的に非ゼロ
        else:
            selected_indices = list(range(self.n_features_original))
        
        # 結果の保存
        self.selectors['l1_regularization'] = model
        self.selected_features['l1_regularization'] = selected_indices
        self.feature_importances['l1_regularization'] = coefs.tolist()
        
        logger.debug(f"L1正則化選択完了: {len(selected_indices)}特徴量選択")
    
    def _fit_importance_selection(self, X: np.ndarray, y: np.ndarray) -> None:
        """重要度ベースの特徴量選択
        
        Args:
            X: 特徴量データ
            y: 目的変数
        """
        config = self.selection_methods['importance_based']
        
        # 推定器の選択
        if config['estimator'] == 'rf':
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # デフォルトはRandomForest
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # モデルの学習
        model.fit(X, y)
        importances = model.feature_importances_
        
        # 閾値の自動決定
        if config['threshold'] == 'auto':
            # 平均重要度を閾値とする
            threshold = np.mean(importances)
        else:
            threshold = config['threshold']
        
        # 上位k個の選択
        if config['top_k'] == 'auto':
            top_k = min(int(self.n_features_original * 0.4), 40)  # 40%または40個の小さい方
        else:
            top_k = config['top_k']
        
        # 閾値ベースの選択
        threshold_selected = np.where(importances >= threshold)[0]
        
        # 上位k個の選択
        top_k_indices = np.argsort(importances)[::-1][:top_k]
        
        # 両方の条件を満たす特徴量を選択（和集合）
        selected_indices = np.unique(np.concatenate([threshold_selected, top_k_indices])).tolist()
        
        # 結果の保存
        self.selectors['importance_based'] = model
        self.selected_features['importance_based'] = selected_indices
        self.feature_importances['importance_based'] = importances.tolist()
        
        logger.debug(f"重要度ベース選択完了: {len(selected_indices)}特徴量選択")
    
    def _get_ensemble_selection(self) -> List[int]:
        """アンサンブル選択（複数手法の結果を統合）
        
        Returns:
            選択された特徴量のインデックスリスト
        """
        if not self.selected_features:
            raise ValueError("特徴量選択が実行されていません")
        
        # 各手法で選択された特徴量の投票
        feature_votes = np.zeros(self.n_features_original)
        
        for method, selected_indices in self.selected_features.items():
            feature_votes[selected_indices] += 1
        
        # 過半数の手法で選択された特徴量を採用
        n_methods = len(self.selected_features)
        threshold = max(1, n_methods // 2)  # 最低1票、過半数
        
        ensemble_selected = np.where(feature_votes >= threshold)[0].tolist()
        
        # 最低限の特徴量数を保証
        min_features = min(10, self.n_features_original)
        if len(ensemble_selected) < min_features:
            # 投票数の多い順に追加
            sorted_indices = np.argsort(feature_votes)[::-1]
            ensemble_selected = sorted_indices[:min_features].tolist()
        
        self.selected_features['ensemble'] = ensemble_selected
        return ensemble_selected
    
    def generate_interaction_features(self, X: Union[np.ndarray, pd.DataFrame],
                                    selected_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, List[str]]:
        """相互作用項の生成
        
        Args:
            X: 特徴量データ
            selected_indices: 相互作用を生成する特徴量のインデックス
            
        Returns:
            相互作用項を含む特徴量データと特徴量名のタプル
        """
        # データ形式の統一
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 選択された特徴量のみを使用
        if selected_indices is not None:
            X_selected = X_array[:, selected_indices]
            selected_names = [self.feature_names[i] for i in selected_indices]
        else:
            X_selected = X_array
            selected_names = self.feature_names
        
        # 特徴量数の制限
        max_features_for_interaction = min(len(selected_names), 20)  # 計算量を考慮
        if len(selected_names) > max_features_for_interaction:
            # 重要度が高い特徴量を優先
            if 'importance_based' in self.feature_importances:
                importances = np.array(self.feature_importances['importance_based'])
                if selected_indices is not None:
                    importances = importances[selected_indices]
                top_indices = np.argsort(importances)[::-1][:max_features_for_interaction]
                X_selected = X_selected[:, top_indices]
                selected_names = [selected_names[i] for i in top_indices]
        
        # 相互作用項の生成
        poly = PolynomialFeatures(
            degree=self.interaction_config['degree'],
            interaction_only=self.interaction_config['interaction_only'],
            include_bias=self.interaction_config['include_bias']
        )
        
        X_poly = poly.fit_transform(X_selected)
        
        # 特徴量名の生成
        poly_feature_names = poly.get_feature_names_out(selected_names)
        
        # 最大特徴量数の制限
        if X_poly.shape[1] > self.interaction_config['max_features']:
            logger.warning(f"相互作用項が多すぎます({X_poly.shape[1]})。最初の{self.interaction_config['max_features']}個のみ使用。")
            X_poly = X_poly[:, :self.interaction_config['max_features']]
            poly_feature_names = poly_feature_names[:self.interaction_config['max_features']]
        
        logger.debug(f"相互作用項生成完了: {X_selected.shape[1]} → {X_poly.shape[1]}")
        return X_poly, poly_feature_names.tolist()
    
    def get_selected_feature_names(self, method: str = 'ensemble') -> List[str]:
        """選択された特徴量名を取得
        
        Args:
            method: 選択手法
            
        Returns:
            選択された特徴量名のリスト
        """
        if not self.is_fitted:
            raise ValueError("特徴量選択器が学習されていません。")
        
        if method == 'ensemble':
            selected_indices = self._get_ensemble_selection()
        else:
            if method not in self.selected_features:
                raise ValueError(f"手法 '{method}' は学習されていません")
            selected_indices = self.selected_features[method]
        
        return [self.feature_names[i] for i in selected_indices]
    
    def get_feature_ranking(self, method: str = 'ensemble') -> List[Tuple[str, float]]:
        """特徴量の重要度ランキングを取得
        
        Args:
            method: 選択手法
            
        Returns:
            (特徴量名, スコア)のタプルリスト（スコア降順）
        """
        if not self.is_fitted:
            raise ValueError("特徴量選択器が学習されていません。")
        
        if method == 'ensemble':
            # アンサンブルスコアを計算
            scores = np.zeros(self.n_features_original)
            for m in self.selected_features.keys():
                if m in self.feature_importances:
                    scores += np.array(self.feature_importances[m])
                elif m in self.feature_scores:
                    scores += np.array(self.feature_scores[m])
            scores = scores / len(self.selected_features)
        else:
            if method in self.feature_importances:
                scores = np.array(self.feature_importances[method])
            elif method in self.feature_scores:
                scores = np.array(self.feature_scores[method])
            else:
                raise ValueError(f"手法 '{method}' のスコア情報がありません")
        
        # ランキング作成
        ranking = [(self.feature_names[i], float(scores[i])) for i in range(len(scores))]
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """特徴量選択の要約情報を取得
        
        Returns:
            選択要約情報の辞書
        """
        if not self.is_fitted:
            raise ValueError("特徴量選択器が学習されていません。")
        
        summary = {
            'n_features_original': self.n_features_original,
            'task_type': self.task_type,
            'methods_used': list(self.selected_features.keys()),
            'selection_results': {}
        }
        
        for method, selected_indices in self.selected_features.items():
            summary['selection_results'][method] = {
                'n_selected': len(selected_indices),
                'selection_ratio': len(selected_indices) / self.n_features_original,
                'selected_features': [self.feature_names[i] for i in selected_indices[:10]]  # 最初の10個
            }
        
        return summary
    
    def save(self, filepath: str) -> None:
        """特徴量選択器を保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        if not self.is_fitted:
            raise ValueError("特徴量選択器が学習されていません。")
        
        save_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'n_features_original': self.n_features_original,
            'task_type': self.task_type,
            'selectors': self.selectors,
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'feature_importances': self.feature_importances,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"FeatureSelector保存完了: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureSelector':
        """特徴量選択器を読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
            
        Returns:
            読み込まれたFeatureSelectorインスタンス
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        instance = cls(config=save_data['config'])
        instance.feature_names = save_data['feature_names']
        instance.n_features_original = save_data['n_features_original']
        instance.task_type = save_data['task_type']
        instance.selectors = save_data['selectors']
        instance.selected_features = save_data['selected_features']
        instance.feature_scores = save_data['feature_scores']
        instance.feature_importances = save_data['feature_importances']
        instance.is_fitted = save_data['is_fitted']
        
        logger.info(f"FeatureSelector読み込み完了: {filepath}")
        return instance
    
    def generate_selection_report(self, output_path: Optional[str] = None) -> str:
        """特徴量選択分析レポートを生成
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            
        Returns:
            生成されたレポートファイルのパス
        """
        if not self.is_fitted:
            raise ValueError("特徴量選択器が学習されていません。")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"feature_selection_report_{timestamp}.txt"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_selection_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("特徴量選択分析レポート\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本情報
            f.write("【基本情報】\n")
            f.write(f"元特徴量数: {summary['n_features_original']}\n")
            f.write(f"タスクタイプ: {summary['task_type']}\n")
            f.write(f"使用手法: {', '.join(summary['methods_used'])}\n\n")
            
            # 手法別結果
            f.write("【手法別選択結果】\n")
            for method, result in summary['selection_results'].items():
                f.write(f"\n{method}:\n")
                f.write(f"  選択特徴量数: {result['n_selected']}個\n")
                f.write(f"  選択率: {result['selection_ratio']:.1%}\n")
                f.write(f"  主要選択特徴量: {', '.join(result['selected_features'])}\n")
            
            # アンサンブル結果
            if len(summary['methods_used']) > 1:
                ensemble_features = self.get_selected_feature_names('ensemble')
                f.write(f"\n【アンサンブル選択結果】\n")
                f.write(f"選択特徴量数: {len(ensemble_features)}個\n")
                f.write(f"選択率: {len(ensemble_features) / summary['n_features_original']:.1%}\n")
                f.write(f"選択特徴量: {', '.join(ensemble_features[:20])}\n")
                if len(ensemble_features) > 20:
                    f.write(f"... 他 {len(ensemble_features) - 20} 個\n")
            
            # 重要度ランキング
            try:
                ranking = self.get_feature_ranking('ensemble')
                f.write(f"\n【重要度ランキング TOP20】\n")
                for i, (name, score) in enumerate(ranking[:20], 1):
                    f.write(f"{i:2d}. {name:40s} | スコア:{score:8.4f}\n")
            except Exception as e:
                f.write(f"\n重要度ランキング生成エラー: {e}\n")
        
        logger.info(f"特徴量選択分析レポート生成完了: {output_path}")
        return str(output_path)