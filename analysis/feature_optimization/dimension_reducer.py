"""
次元削減器
==========

PCA、UMAP、t-SNEによる次元削減機能を実装。
最適次元数の自動決定機能と次元削減後の特徴量解釈機能を提供します。
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DimensionReducer:
    """次元削減器
    
    PCA、UMAP、t-SNEによる次元削減機能を実装。
    最適次元数の自動決定機能と次元削減後の特徴量解釈機能を提供。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（次元削減手法、パラメータなど）
        """
        self.config = config or {}
        
        # 次元削減手法の設定
        self.reduction_methods = self.config.get('reduction_methods', {
            'pca': {
                'n_components': 'auto',    # 'auto', int, float(0-1)
                'variance_threshold': 0.95, # 累積寄与率の閾値
                'svd_solver': 'auto'
            },
            'truncated_svd': {
                'n_components': 'auto',
                'n_iter': 5,
                'random_state': 42
            },
            'ica': {
                'n_components': 'auto',
                'algorithm': 'parallel',
                'whiten': True,
                'random_state': 42
            },
            'tsne': {
                'n_components': 2,
                'perplexity': 30.0,
                'learning_rate': 200.0,
                'n_iter': 1000,
                'random_state': 42
            }
        })
        
        # 最適次元数決定の設定
        self.auto_dimension_config = self.config.get('auto_dimension_config', {
            'min_components': 2,
            'max_components_ratio': 0.8,  # 元特徴量数の80%まで
            'variance_threshold': 0.95,   # PCAの累積寄与率
            'elbow_sensitivity': 0.1      # エルボー法の感度
        })
        
        # 初期化
        self.reducers = {}
        self.scalers = {}
        self.reduction_info = {}
        self.is_fitted = False
        self.original_feature_names = None
        self.n_features_original = None
        
        logger.info(f"DimensionReducer初期化完了")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            feature_names: Optional[List[str]] = None,
            methods: Optional[List[str]] = None,
            scale_data: bool = True) -> 'DimensionReducer':
        """次元削減器を学習データに適合
        
        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト
            methods: 使用する次元削減手法のリスト
            scale_data: データをスケーリングするかどうか
            
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
        
        self.original_feature_names = feature_names
        self.n_features_original = len(feature_names)
        
        # データのスケーリング
        if scale_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            self.scalers['standard'] = scaler
        else:
            X_scaled = X_array
            self.scalers['standard'] = None
        
        # 使用する手法の決定
        if methods is None:
            methods = ['pca', 'truncated_svd']  # デフォルトは軽量な手法
        
        # 各手法で次元削減を実行
        for method in methods:
            if method == 'pca':
                self._fit_pca(X_scaled)
            elif method == 'truncated_svd':
                self._fit_truncated_svd(X_scaled)
            elif method == 'ica':
                self._fit_ica(X_scaled)
            elif method == 'tsne':
                self._fit_tsne(X_scaled)
            elif method == 'umap':
                self._fit_umap(X_scaled)
        
        self.is_fitted = True
        logger.info(f"DimensionReducer学習完了: {len(methods)}手法")
        
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], 
                 method: str = 'pca') -> np.ndarray:
        """次元削減を適用
        
        Args:
            X: 変換する特徴量データ
            method: 使用する次元削減手法
            
        Returns:
            次元削減後のデータ
        """
        if not self.is_fitted:
            raise ValueError("次元削減器が学習されていません。先にfit()を呼び出してください。")
        
        if method not in self.reducers:
            raise ValueError(f"手法 '{method}' は学習されていません")
        
        # データ形式の統一
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if X_array.shape[1] != self.n_features_original:
            raise ValueError(f"入力データの次元数({X_array.shape[1]})が学習時({self.n_features_original})と異なります")
        
        # スケーリング（学習時と同じ設定）
        if self.scalers['standard'] is not None:
            X_scaled = self.scalers['standard'].transform(X_array)
        else:
            X_scaled = X_array
        
        # 次元削減の適用
        reducer = self.reducers[method]
        X_reduced = reducer.transform(X_scaled)
        
        logger.debug(f"次元削減変換完了: {X_array.shape[1]} → {X_reduced.shape[1]} ({method})")
        return X_reduced
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     feature_names: Optional[List[str]] = None,
                     methods: Optional[List[str]] = None,
                     transform_method: str = 'pca',
                     scale_data: bool = True) -> np.ndarray:
        """学習と変換を同時実行
        
        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト
            methods: 使用する次元削減手法のリスト
            transform_method: 変換に使用する手法
            scale_data: データをスケーリングするかどうか
            
        Returns:
            次元削減後のデータ
        """
        return self.fit(X, feature_names, methods, scale_data).transform(X, transform_method)
    
    def _fit_pca(self, X: np.ndarray) -> None:
        """PCAによる次元削減
        
        Args:
            X: スケーリング済み特徴量データ
        """
        config = self.reduction_methods['pca']
        
        # 最適次元数の決定
        n_components = self._determine_pca_components(X, config)
        
        # PCAの学習
        pca = PCA(
            n_components=n_components,
            svd_solver=config['svd_solver'],
            random_state=42
        )
        pca.fit(X)
        
        # 結果の保存
        self.reducers['pca'] = pca
        self.reduction_info['pca'] = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'singular_values': pca.singular_values_.tolist(),
            'components': pca.components_.tolist()
        }
        
        logger.debug(f"PCA学習完了: {X.shape[1]} → {n_components}次元")
    
    def _fit_truncated_svd(self, X: np.ndarray) -> None:
        """Truncated SVDによる次元削減
        
        Args:
            X: スケーリング済み特徴量データ
        """
        config = self.reduction_methods['truncated_svd']
        
        # 最適次元数の決定
        if config['n_components'] == 'auto':
            max_components = min(X.shape[0] - 1, X.shape[1] - 1)
            n_components = min(int(max_components * self.auto_dimension_config['max_components_ratio']), 50)
        else:
            n_components = config['n_components']
        
        # Truncated SVDの学習
        svd = TruncatedSVD(
            n_components=n_components,
            n_iter=config['n_iter'],
            random_state=config['random_state']
        )
        svd.fit(X)
        
        # 結果の保存
        self.reducers['truncated_svd'] = svd
        self.reduction_info['truncated_svd'] = {
            'n_components': n_components,
            'explained_variance_ratio': svd.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(svd.explained_variance_ratio_).tolist(),
            'singular_values': svd.singular_values_.tolist(),
            'components': svd.components_.tolist()
        }
        
        logger.debug(f"Truncated SVD学習完了: {X.shape[1]} → {n_components}次元")
    
    def _fit_ica(self, X: np.ndarray) -> None:
        """ICAによる次元削減
        
        Args:
            X: スケーリング済み特徴量データ
        """
        config = self.reduction_methods['ica']
        
        # 最適次元数の決定
        if config['n_components'] == 'auto':
            max_components = min(X.shape[0], X.shape[1])
            n_components = min(int(max_components * self.auto_dimension_config['max_components_ratio']), 30)
        else:
            n_components = config['n_components']
        
        # ICAの学習
        ica = FastICA(
            n_components=n_components,
            algorithm=config['algorithm'],
            whiten=config['whiten'],
            random_state=config['random_state'],
            max_iter=1000
        )
        
        try:
            ica.fit(X)
            
            # 結果の保存
            self.reducers['ica'] = ica
            self.reduction_info['ica'] = {
                'n_components': n_components,
                'components': ica.components_.tolist(),
                'mixing_matrix': ica.mixing_.tolist() if hasattr(ica, 'mixing_') else None,
                'n_iter': ica.n_iter_
            }
            
            logger.debug(f"ICA学習完了: {X.shape[1]} → {n_components}次元")
            
        except Exception as e:
            logger.warning(f"ICA学習失敗: {e}")
            # ICAが失敗した場合は結果を保存しない
    
    def _fit_tsne(self, X: np.ndarray) -> None:
        """t-SNEによる次元削減
        
        Args:
            X: スケーリング済み特徴量データ
        """
        config = self.reduction_methods['tsne']
        
        # t-SNEは計算量が多いため、サンプル数を制限
        if X.shape[0] > 5000:
            logger.warning(f"t-SNE: サンプル数が多いため({X.shape[0]})、最初の5000サンプルのみ使用")
            X_sample = X[:5000]
        else:
            X_sample = X
        
        # t-SNEの学習
        tsne = TSNE(
            n_components=config['n_components'],
            perplexity=min(config['perplexity'], X_sample.shape[0] - 1),
            learning_rate=config['learning_rate'],
            n_iter=config['n_iter'],
            random_state=config['random_state']
        )
        
        try:
            X_embedded = tsne.fit_transform(X_sample)
            
            # t-SNEは変換のみ可能（新しいデータに適用不可）
            self.reduction_info['tsne'] = {
                'n_components': config['n_components'],
                'perplexity': tsne.perplexity,
                'learning_rate': tsne.learning_rate,
                'n_iter': tsne.n_iter_,
                'kl_divergence': tsne.kl_divergence_,
                'embedded_data': X_embedded.tolist()  # 学習データの埋め込み結果
            }
            
            logger.debug(f"t-SNE学習完了: {X.shape[1]} → {config['n_components']}次元")
            
        except Exception as e:
            logger.warning(f"t-SNE学習失敗: {e}")
    
    def _fit_umap(self, X: np.ndarray) -> None:
        """UMAPによる次元削減（umapライブラリが利用可能な場合）
        
        Args:
            X: スケーリング済み特徴量データ
        """
        try:
            import umap
            
            config = self.config.get('umap', {
                'n_components': 2,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'random_state': 42
            })
            
            # UMAPの学習
            umap_reducer = umap.UMAP(
                n_components=config['n_components'],
                n_neighbors=config['n_neighbors'],
                min_dist=config['min_dist'],
                metric=config['metric'],
                random_state=config['random_state']
            )
            
            umap_reducer.fit(X)
            
            # 結果の保存
            self.reducers['umap'] = umap_reducer
            self.reduction_info['umap'] = {
                'n_components': config['n_components'],
                'n_neighbors': config['n_neighbors'],
                'min_dist': config['min_dist'],
                'metric': config['metric']
            }
            
            logger.debug(f"UMAP学習完了: {X.shape[1]} → {config['n_components']}次元")
            
        except ImportError:
            logger.warning("UMAPライブラリが利用できません。pip install umapでインストールしてください。")
        except Exception as e:
            logger.warning(f"UMAP学習失敗: {e}")
    
    def _determine_pca_components(self, X: np.ndarray, config: Dict[str, Any]) -> int:
        """PCAの最適次元数を決定
        
        Args:
            X: 特徴量データ
            config: PCA設定
            
        Returns:
            最適次元数
        """
        if isinstance(config['n_components'], int):
            return min(config['n_components'], X.shape[1])
        elif isinstance(config['n_components'], float):
            return int(X.shape[1] * config['n_components'])
        elif config['n_components'] == 'auto':
            # 累積寄与率による自動決定
            pca_full = PCA()
            pca_full.fit(X)
            
            cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum_ratio >= config['variance_threshold']) + 1
            
            # 最小・最大制約
            min_comp = self.auto_dimension_config['min_components']
            max_comp = int(X.shape[1] * self.auto_dimension_config['max_components_ratio'])
            
            n_components = max(min_comp, min(n_components, max_comp))
            
            return n_components
        else:
            return min(10, X.shape[1])  # デフォルト
    
    def get_component_interpretation(self, method: str = 'pca', 
                                   n_top_features: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """次元削減後の成分解釈
        
        Args:
            method: 次元削減手法
            n_top_features: 各成分で表示する上位特徴量数
            
        Returns:
            成分別の重要特徴量辞書
        """
        if not self.is_fitted:
            raise ValueError("次元削減器が学習されていません。")
        
        if method not in self.reduction_info:
            raise ValueError(f"手法 '{method}' は学習されていません")
        
        if 'components' not in self.reduction_info[method]:
            raise ValueError(f"手法 '{method}' には成分情報がありません")
        
        components = np.array(self.reduction_info[method]['components'])
        interpretation = {}
        
        for i, component in enumerate(components):
            # 絶対値で重要度をソート
            feature_importance = [(self.original_feature_names[j], float(component[j])) 
                                for j in range(len(component))]
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            interpretation[f'Component_{i+1}'] = feature_importance[:n_top_features]
        
        return interpretation
    
    def get_optimal_dimensions(self, method: str = 'pca', 
                             variance_threshold: float = 0.95) -> Dict[str, Any]:
        """最適次元数の分析
        
        Args:
            method: 次元削減手法
            variance_threshold: 分散説明率の閾値
            
        Returns:
            最適次元数の分析結果
        """
        if not self.is_fitted:
            raise ValueError("次元削減器が学習されていません。")
        
        if method not in self.reduction_info:
            raise ValueError(f"手法 '{method}' は学習されていません")
        
        info = self.reduction_info[method]
        result = {
            'current_dimensions': info['n_components'],
            'original_dimensions': self.n_features_original
        }
        
        if 'explained_variance_ratio' in info:
            explained_var = np.array(info['explained_variance_ratio'])
            cumulative_var = np.array(info['cumulative_variance_ratio'])
            
            # 指定された分散説明率を満たす最小次元数
            optimal_dims = np.argmax(cumulative_var >= variance_threshold) + 1
            
            result.update({
                'explained_variance_ratio': explained_var.tolist(),
                'cumulative_variance_ratio': cumulative_var.tolist(),
                'optimal_dimensions_for_threshold': int(optimal_dims),
                'variance_at_optimal': float(cumulative_var[optimal_dims - 1]),
                'current_variance_explained': float(cumulative_var[-1])
            })
            
            # エルボー法による推奨次元数
            elbow_point = self._find_elbow_point(explained_var)
            result['elbow_point_dimensions'] = elbow_point
        
        return result
    
    def _find_elbow_point(self, explained_variance: np.ndarray) -> int:
        """エルボー法による最適次元数の推定
        
        Args:
            explained_variance: 各成分の分散説明率
            
        Returns:
            エルボーポイントの次元数
        """
        if len(explained_variance) < 3:
            return len(explained_variance)
        
        # 2次微分を計算してエルボーポイントを検出
        second_derivative = np.diff(explained_variance, n=2)
        
        # 最大の2次微分の位置をエルボーポイントとする
        elbow_idx = np.argmax(np.abs(second_derivative)) + 2  # diff操作で2つ減るため
        
        return min(elbow_idx, len(explained_variance))
    
    def get_reduction_summary(self) -> Dict[str, Any]:
        """次元削減の要約情報を取得
        
        Returns:
            次元削減要約情報の辞書
        """
        if not self.is_fitted:
            raise ValueError("次元削減器が学習されていません。")
        
        summary = {
            'n_features_original': self.n_features_original,
            'methods_used': list(self.reduction_info.keys()),
            'reduction_results': {}
        }
        
        for method, info in self.reduction_info.items():
            result = {
                'n_components': info['n_components'],
                'reduction_ratio': info['n_components'] / self.n_features_original
            }
            
            if 'cumulative_variance_ratio' in info:
                result['variance_explained'] = info['cumulative_variance_ratio'][-1]
            
            summary['reduction_results'][method] = result
        
        return summary
    
    def save(self, filepath: str) -> None:
        """次元削減器を保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        if not self.is_fitted:
            raise ValueError("次元削減器が学習されていません。")
        
        save_data = {
            'config': self.config,
            'original_feature_names': self.original_feature_names,
            'n_features_original': self.n_features_original,
            'reducers': self.reducers,
            'scalers': self.scalers,
            'reduction_info': self.reduction_info,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"DimensionReducer保存完了: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DimensionReducer':
        """次元削減器を読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
            
        Returns:
            読み込まれたDimensionReducerインスタンス
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        instance = cls(config=save_data['config'])
        instance.original_feature_names = save_data['original_feature_names']
        instance.n_features_original = save_data['n_features_original']
        instance.reducers = save_data['reducers']
        instance.scalers = save_data['scalers']
        instance.reduction_info = save_data['reduction_info']
        instance.is_fitted = save_data['is_fitted']
        
        logger.info(f"DimensionReducer読み込み完了: {filepath}")
        return instance
    
    def generate_reduction_report(self, output_path: Optional[str] = None) -> str:
        """次元削減分析レポートを生成
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            
        Returns:
            生成されたレポートファイルのパス
        """
        if not self.is_fitted:
            raise ValueError("次元削減器が学習されていません。")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"dimension_reduction_report_{timestamp}.txt"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_reduction_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("次元削減分析レポート\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本情報
            f.write("【基本情報】\n")
            f.write(f"元特徴量数: {summary['n_features_original']}\n")
            f.write(f"使用手法: {', '.join(summary['methods_used'])}\n\n")
            
            # 手法別結果
            f.write("【手法別削減結果】\n")
            for method, result in summary['reduction_results'].items():
                f.write(f"\n{method}:\n")
                f.write(f"  削減後次元数: {result['n_components']}\n")
                f.write(f"  削減率: {1 - result['reduction_ratio']:.1%}\n")
                
                if 'variance_explained' in result:
                    f.write(f"  分散説明率: {result['variance_explained']:.1%}\n")
                
                # 最適次元数分析
                try:
                    optimal_info = self.get_optimal_dimensions(method)
                    f.write(f"  推奨次元数(95%分散): {optimal_info.get('optimal_dimensions_for_threshold', 'N/A')}\n")
                    f.write(f"  エルボーポイント: {optimal_info.get('elbow_point_dimensions', 'N/A')}\n")
                except:
                    pass
            
            # 成分解釈（PCAの場合）
            if 'pca' in summary['methods_used']:
                try:
                    interpretation = self.get_component_interpretation('pca', n_top_features=3)
                    f.write(f"\n【PCA成分解釈】\n")
                    for component, features in list(interpretation.items())[:5]:  # 最初の5成分
                        f.write(f"\n{component}:\n")
                        for feature_name, weight in features:
                            f.write(f"  {feature_name:30s}: {weight:7.4f}\n")
                except Exception as e:
                    f.write(f"\n成分解釈生成エラー: {e}\n")
        
        logger.info(f"次元削減分析レポート生成完了: {output_path}")
        return str(output_path)