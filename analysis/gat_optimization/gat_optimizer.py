"""
GAT特徴量最適化器
================

Graph Attention Network特徴量の最適次元数決定・アテンション重み分析・埋め込み品質評価を実装します。
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class GATOptimizer:
    """GAT特徴量最適化器

    GAT特徴量の最適次元数決定アルゴリズム、アテンション重みの分析と可視化、
    埋め込み品質評価メトリクス（分散、情報量、クラスタリング品質）を実装。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（最適化手法、パラメータなど）
        """
        self.config = config or {}

        # 最適化設定
        self.optimization_config = self.config.get(
            "optimization_config",
            {
                "dimension_search": {
                    "min_dimensions": 8,  # 最小次元数
                    "max_dimensions": 128,  # 最大次元数
                    "search_step": 8,  # 検索ステップ
                    "evaluation_methods": ["variance", "clustering", "information"],
                },
                "attention_analysis": {
                    "top_k_heads": 5,  # 分析する上位アテンションヘッド数
                    "layer_analysis": True,  # 層別分析を実行するか
                    "edge_type_analysis": True,  # エッジタイプ別分析を実行するか
                },
                "quality_metrics": {
                    "clustering_methods": ["kmeans", "spectral"],
                    "variance_threshold": 0.01,  # 分散の最小閾値
                    "information_bins": 50,  # 情報量計算時のビン数
                    "silhouette_min_samples": 10,  # シルエット分析の最小サンプル数
                },
            },
        )

        # 初期化
        self.gat_model = None
        self.embeddings = None
        self.attention_weights = None
        self.is_fitted = False
        self.optimization_results = {}
        self.original_dimensions = None

        logger.info("GATOptimizer初期化完了")

    def load_gat_model(
        self, model_path: str, graph_data_path: Optional[str] = None
    ) -> "GATOptimizer":
        """GATモデルと関連データを読み込み

        Args:
            model_path: GATモデルファイルのパス
            graph_data_path: グラフデータファイルのパス（オプション）

        Returns:
            自身のインスタンス
        """
        try:
            # GATモデルの読み込み
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"GATモデルファイルが見つかりません: {model_path}"
                )

            # モデルファイルの種類に応じて読み込み方法を調整
            if model_path.suffix == ".pt":
                # PyTorchモデルファイル
                model_data = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )

                if isinstance(model_data, dict):
                    # モデルの状態辞書の場合
                    self.gat_model = self._initialize_gat_model()
                    self.gat_model.load_state_dict(model_data)
                else:
                    # モデルインスタンスの場合
                    self.gat_model = model_data

                self.gat_model.eval()
                logger.info(f"GATモデル読み込み完了: {model_path}")

            # グラフデータの読み込み（オプション）
            if graph_data_path:
                graph_data_path = Path(graph_data_path)
                if graph_data_path.exists():
                    self.graph_data = torch.load(graph_data_path, weights_only=False)
                    logger.info(f"グラフデータ読み込み完了: {graph_data_path}")
                else:
                    logger.warning(
                        f"グラフデータファイルが見つかりません: {graph_data_path}"
                    )

            self.is_fitted = True
            return self

        except Exception as e:
            logger.error(f"GATモデル読み込みエラー: {e}")
            raise

    def _initialize_gat_model(self):
        """GATモデルを初期化（適切なアーキテクチャで）"""
        try:
            # GATモデルをインポート
            import sys
            from pathlib import Path

            sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
            from kazoo.GAT.GAT_model import GATModel

            # デフォルトのGATモデル設定
            model = GATModel(in_channels_dict={"dev": 8, "task": 9}, out_channels=32)
            return model

        except ImportError as e:
            logger.error(f"GATModelのインポートに失敗: {e}")
            raise

    def extract_embeddings(
        self, graph_data: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """GATモデルから埋め込みを抽出

        Args:
            graph_data: グラフデータ（Noneの場合は内部データを使用）

        Returns:
            抽出された埋め込み辞書
        """
        if not self.is_fitted or self.gat_model is None:
            raise ValueError("GATモデルが読み込まれていません。")

        try:
            graph_data = graph_data or getattr(self, "graph_data", None)
            if graph_data is None:
                raise ValueError("グラフデータが提供されていません。")

            with torch.no_grad():
                self.embeddings = self.gat_model(
                    graph_data.x_dict, graph_data.edge_index_dict
                )

            # 次元数を記録
            self.original_dimensions = {
                node_type: emb.shape[1] for node_type, emb in self.embeddings.items()
            }

            logger.info(f"埋め込み抽出完了: {self.original_dimensions}")
            return self.embeddings

        except Exception as e:
            logger.error(f"埋め込み抽出エラー: {e}")
            raise

    def analyze_optimal_dimensions(
        self, node_type: str = "dev", methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """最適次元数の分析

        Args:
            node_type: 分析対象のノードタイプ（'dev', 'task'など）
            methods: 使用する評価手法のリスト

        Returns:
            最適次元数分析の結果
        """
        if self.embeddings is None:
            raise ValueError("埋め込みが抽出されていません。")

        if node_type not in self.embeddings:
            raise ValueError(f"ノードタイプ '{node_type}' が見つかりません")

        methods = (
            methods
            or self.optimization_config["dimension_search"]["evaluation_methods"]
        )
        embeddings = self.embeddings[node_type].detach().cpu().numpy()

        config = self.optimization_config["dimension_search"]
        min_dims = config["min_dimensions"]
        max_dims = min(config["max_dimensions"], embeddings.shape[1])
        step = config["search_step"]

        dimensions_to_test = list(range(min_dims, max_dims + 1, step))

        results = {
            "node_type": node_type,
            "original_dimensions": embeddings.shape[1],
            "tested_dimensions": dimensions_to_test,
            "evaluation_results": {},
            "optimal_dimensions": {},
        }

        for method in methods:
            logger.info(f"{method}による次元数評価を実行中...")

            if method == "variance":
                method_results = self._analyze_dimensions_by_variance(
                    embeddings, dimensions_to_test
                )
            elif method == "clustering":
                method_results = self._analyze_dimensions_by_clustering(
                    embeddings, dimensions_to_test
                )
            elif method == "information":
                method_results = self._analyze_dimensions_by_information(
                    embeddings, dimensions_to_test
                )
            else:
                logger.warning(f"未知の評価手法: {method}")
                continue

            results["evaluation_results"][method] = method_results
            results["optimal_dimensions"][method] = method_results["optimal_dimension"]

        # 総合的な最適次元数を決定
        optimal_dims = list(results["optimal_dimensions"].values())
        if optimal_dims:
            results["recommended_dimension"] = int(np.median(optimal_dims))

        self.optimization_results[f"dimension_analysis_{node_type}"] = results
        logger.info(
            f"次元数分析完了: 推奨次元数 = {results.get('recommended_dimension', 'N/A')}"
        )

        return results

    def _analyze_dimensions_by_variance(
        self, embeddings: np.ndarray, dimensions: List[int]
    ) -> Dict[str, Any]:
        """分散ベースの次元数分析"""
        results = {
            "method": "variance",
            "dimension_scores": {},
            "optimal_dimension": None,
        }

        threshold = self.optimization_config["quality_metrics"]["variance_threshold"]

        for dim in dimensions:
            if dim > embeddings.shape[1]:
                continue

            # 主成分分析で次元削減
            from sklearn.decomposition import PCA

            pca = PCA(n_components=dim)
            reduced_embeddings = pca.fit_transform(embeddings)

            # 各次元の分散を計算
            variances = np.var(reduced_embeddings, axis=0)

            # 有効次元数（分散が閾値以上の次元数）
            effective_dims = np.sum(variances >= threshold)

            # スコア：有効次元数の割合と累積寄与率のバランス
            cumulative_variance_ratio = np.sum(pca.explained_variance_ratio_)
            score = effective_dims / dim * cumulative_variance_ratio

            results["dimension_scores"][dim] = {
                "score": score,
                "effective_dimensions": effective_dims,
                "cumulative_variance_ratio": cumulative_variance_ratio,
                "mean_variance": np.mean(variances),
            }

        # 最適次元数を決定（スコアが最大となる次元数）
        if results["dimension_scores"]:
            optimal_dim = max(
                results["dimension_scores"].keys(),
                key=lambda d: results["dimension_scores"][d]["score"],
            )
            results["optimal_dimension"] = optimal_dim

        return results

    def _analyze_dimensions_by_clustering(
        self, embeddings: np.ndarray, dimensions: List[int]
    ) -> Dict[str, Any]:
        """クラスタリング品質ベースの次元数分析"""
        results = {
            "method": "clustering",
            "dimension_scores": {},
            "optimal_dimension": None,
        }

        min_samples = self.optimization_config["quality_metrics"][
            "silhouette_min_samples"
        ]
        if embeddings.shape[0] < min_samples:
            logger.warning(f"サンプル数が不足（{embeddings.shape[0]} < {min_samples}）")
            return results

        # クラスタ数を推定（サンプル数の平方根程度）
        n_clusters = max(2, min(10, int(np.sqrt(embeddings.shape[0]))))

        for dim in dimensions:
            if dim > embeddings.shape[1]:
                continue

            try:
                # 次元削減
                from sklearn.decomposition import PCA

                pca = PCA(n_components=dim)
                reduced_embeddings = pca.fit_transform(embeddings)

                # K-meansクラスタリング
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(reduced_embeddings)

                # シルエット係数を計算
                silhouette_avg = silhouette_score(reduced_embeddings, cluster_labels)

                # クラスタ内分散とクラスタ間分散の比率
                inertia = kmeans.inertia_ / embeddings.shape[0]  # 正規化

                # 総合スコア
                score = silhouette_avg * (1 / (1 + inertia))

                results["dimension_scores"][dim] = {
                    "score": score,
                    "silhouette_score": silhouette_avg,
                    "inertia": inertia,
                    "n_clusters": n_clusters,
                }

            except Exception as e:
                logger.warning(f"次元{dim}でのクラスタリング分析に失敗: {e}")
                continue

        # 最適次元数を決定
        if results["dimension_scores"]:
            optimal_dim = max(
                results["dimension_scores"].keys(),
                key=lambda d: results["dimension_scores"][d]["score"],
            )
            results["optimal_dimension"] = optimal_dim

        return results

    def _analyze_dimensions_by_information(
        self, embeddings: np.ndarray, dimensions: List[int]
    ) -> Dict[str, Any]:
        """情報量ベースの次元数分析"""
        results = {
            "method": "information",
            "dimension_scores": {},
            "optimal_dimension": None,
        }

        n_bins = self.optimization_config["quality_metrics"]["information_bins"]

        for dim in dimensions:
            if dim > embeddings.shape[1]:
                continue

            try:
                # 次元削減
                from sklearn.decomposition import PCA

                pca = PCA(n_components=dim)
                reduced_embeddings = pca.fit_transform(embeddings)

                # 各次元の情報エントロピーを計算
                entropies = []
                for i in range(dim):
                    # ヒストグラムを作成してエントロピー計算
                    hist, _ = np.histogram(
                        reduced_embeddings[:, i], bins=n_bins, density=True
                    )
                    hist = hist + 1e-10  # ゼロ対策
                    hist = hist / np.sum(hist)  # 正規化
                    entropy = -np.sum(hist * np.log2(hist))
                    entropies.append(entropy)

                # 総情報量と平均情報量
                total_information = np.sum(entropies)
                avg_information = np.mean(entropies)

                # 情報効率（情報量/次元数）
                information_efficiency = total_information / dim

                results["dimension_scores"][dim] = {
                    "score": information_efficiency,
                    "total_information": total_information,
                    "avg_information": avg_information,
                    "information_efficiency": information_efficiency,
                }

            except Exception as e:
                logger.warning(f"次元{dim}での情報量分析に失敗: {e}")
                continue

        # 最適次元数を決定
        if results["dimension_scores"]:
            optimal_dim = max(
                results["dimension_scores"].keys(),
                key=lambda d: results["dimension_scores"][d]["score"],
            )
            results["optimal_dimension"] = optimal_dim

        return results

    def analyze_attention_weights(self) -> Dict[str, Any]:
        """アテンション重みの分析と可視化

        Returns:
            アテンション重み分析結果
        """
        if not self.is_fitted or self.gat_model is None:
            raise ValueError("GATモデルが読み込まれていません。")

        results = {
            "attention_analysis": {},
            "layer_analysis": {},
            "edge_type_analysis": {},
        }

        try:
            # アテンション重みを抽出
            attention_weights = self._extract_attention_weights()
            if not attention_weights:
                logger.warning("アテンション重みの抽出に失敗")
                return results

            self.attention_weights = attention_weights

            # 層別分析
            if self.optimization_config["attention_analysis"]["layer_analysis"]:
                results["layer_analysis"] = self._analyze_attention_by_layer(
                    attention_weights
                )

            # エッジタイプ別分析
            if self.optimization_config["attention_analysis"]["edge_type_analysis"]:
                results["edge_type_analysis"] = self._analyze_attention_by_edge_type(
                    attention_weights
                )

            # 総合分析
            results["attention_analysis"] = self._analyze_attention_overall(
                attention_weights
            )

            self.optimization_results["attention_analysis"] = results
            logger.info("アテンション重み分析完了")

        except Exception as e:
            logger.error(f"アテンション重み分析エラー: {e}")

        return results

    def _extract_attention_weights(self) -> Dict[str, Any]:
        """GATモデルからアテンション重みを抽出"""
        attention_weights = {}

        try:
            # GATモデルの構造を調べてアテンション重みを抽出
            for name, module in self.gat_model.named_modules():
                # GATConvモジュールを探す
                if hasattr(module, "attention"):
                    attention_weights[name] = {
                        "weights": (
                            module.attention.detach().cpu().numpy()
                            if hasattr(module.attention, "detach")
                            else None
                        ),
                        "module_type": type(module).__name__,
                    }

            logger.info(
                f"アテンション重み抽出完了: {len(attention_weights)} モジュール"
            )

        except Exception as e:
            logger.error(f"アテンション重み抽出エラー: {e}")

        return attention_weights

    def _analyze_attention_by_layer(
        self, attention_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """層別アテンション分析"""
        layer_results = {}

        for layer_name, attention_data in attention_weights.items():
            if attention_data["weights"] is None:
                continue

            weights = attention_data["weights"]

            layer_results[layer_name] = {
                "mean_attention": float(np.mean(weights)),
                "std_attention": float(np.std(weights)),
                "max_attention": float(np.max(weights)),
                "min_attention": float(np.min(weights)),
                "attention_distribution": {
                    "q25": float(np.percentile(weights, 25)),
                    "q50": float(np.percentile(weights, 50)),
                    "q75": float(np.percentile(weights, 75)),
                },
            }

        return layer_results

    def _analyze_attention_by_edge_type(
        self, attention_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """エッジタイプ別アテンション分析"""
        edge_type_results = {}

        # エッジタイプをGATレイヤー名から推定
        edge_types = ["dev_writes_task", "task_written_by_dev", "dev_collaborates_dev"]

        for edge_type in edge_types:
            matching_layers = [
                name for name in attention_weights.keys() if edge_type in name
            ]

            if not matching_layers:
                continue

            edge_type_results[edge_type] = {}

            for layer_name in matching_layers:
                attention_data = attention_weights[layer_name]
                if attention_data["weights"] is None:
                    continue

                weights = attention_data["weights"]
                edge_type_results[edge_type][layer_name] = {
                    "attention_strength": float(np.mean(weights)),
                    "attention_variance": float(np.var(weights)),
                    "attention_concentration": float(
                        np.std(weights) / (np.mean(weights) + 1e-10)
                    ),
                }

        return edge_type_results

    def _analyze_attention_overall(
        self, attention_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """アテンション重みの総合分析"""
        all_weights = []
        layer_statistics = {}

        for layer_name, attention_data in attention_weights.items():
            if attention_data["weights"] is None:
                continue

            weights = attention_data["weights"]
            all_weights.extend(weights.flatten())

            layer_statistics[layer_name] = {
                "weight_count": weights.size,
                "mean_weight": float(np.mean(weights)),
                "contribution_ratio": (
                    float(np.sum(weights)) if len(all_weights) > 0 else 0.0
                ),
            }

        if not all_weights:
            return {}

        all_weights = np.array(all_weights)

        return {
            "overall_statistics": {
                "total_weights": len(all_weights),
                "mean_attention": float(np.mean(all_weights)),
                "std_attention": float(np.std(all_weights)),
                "attention_concentration": float(
                    np.std(all_weights) / (np.mean(all_weights) + 1e-10)
                ),
            },
            "layer_statistics": layer_statistics,
            "top_attention_layers": sorted(
                layer_statistics.keys(),
                key=lambda x: layer_statistics[x]["mean_weight"],
                reverse=True,
            )[: self.optimization_config["attention_analysis"]["top_k_heads"]],
        }

    def evaluate_embedding_quality(self, node_type: str = "dev") -> Dict[str, Any]:
        """埋め込み品質評価メトリクス

        Args:
            node_type: 評価対象のノードタイプ

        Returns:
            品質評価結果
        """
        if self.embeddings is None:
            raise ValueError("埋め込みが抽出されていません。")

        if node_type not in self.embeddings:
            raise ValueError(f"ノードタイプ '{node_type}' が見つかりません")

        embeddings = self.embeddings[node_type].detach().cpu().numpy()

        results = {
            "node_type": node_type,
            "embedding_shape": embeddings.shape,
            "variance_analysis": self._evaluate_variance_quality(embeddings),
            "information_analysis": self._evaluate_information_quality(embeddings),
            "clustering_analysis": self._evaluate_clustering_quality(embeddings),
            "overall_quality_score": None,
        }

        # 総合品質スコアを計算
        quality_scores = []

        if "quality_score" in results["variance_analysis"]:
            quality_scores.append(results["variance_analysis"]["quality_score"])
        if "quality_score" in results["information_analysis"]:
            quality_scores.append(results["information_analysis"]["quality_score"])
        if "quality_score" in results["clustering_analysis"]:
            quality_scores.append(results["clustering_analysis"]["quality_score"])

        if quality_scores:
            results["overall_quality_score"] = float(np.mean(quality_scores))

        self.optimization_results[f"quality_evaluation_{node_type}"] = results
        logger.info(
            f"埋め込み品質評価完了: 総合スコア = {results['overall_quality_score']:.3f}"
        )

        return results

    def _evaluate_variance_quality(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """分散品質の評価"""
        # 各次元の分散
        variances = np.var(embeddings, axis=0)

        # 分散の統計
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        min_variance = np.min(variances)
        max_variance = np.max(variances)

        # 有効次元数（閾値以上の分散を持つ次元数）
        threshold = self.optimization_config["quality_metrics"]["variance_threshold"]
        effective_dimensions = np.sum(variances >= threshold)

        # 分散の均一性（理想的には各次元が同程度の分散を持つ）
        variance_uniformity = 1.0 / (1.0 + std_variance / (mean_variance + 1e-10))

        # 品質スコア
        quality_score = (effective_dimensions / len(variances)) * variance_uniformity

        return {
            "mean_variance": float(mean_variance),
            "std_variance": float(std_variance),
            "min_variance": float(min_variance),
            "max_variance": float(max_variance),
            "effective_dimensions": int(effective_dimensions),
            "variance_uniformity": float(variance_uniformity),
            "quality_score": float(quality_score),
        }

    def _evaluate_information_quality(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """情報量品質の評価"""
        n_bins = self.optimization_config["quality_metrics"]["information_bins"]

        # 各次元のエントロピー計算
        entropies = []
        for i in range(embeddings.shape[1]):
            hist, _ = np.histogram(embeddings[:, i], bins=n_bins, density=True)
            hist = hist + 1e-10  # ゼロ対策
            hist = hist / np.sum(hist)  # 正規化
            entropy = -np.sum(hist * np.log2(hist))
            entropies.append(entropy)

        entropies = np.array(entropies)

        # 情報量統計
        total_information = np.sum(entropies)
        mean_information = np.mean(entropies)
        std_information = np.std(entropies)

        # 情報効率
        information_efficiency = total_information / embeddings.shape[1]

        # 情報の均一性
        information_uniformity = 1.0 / (
            1.0 + std_information / (mean_information + 1e-10)
        )

        # 品質スコア
        quality_score = information_efficiency * information_uniformity

        return {
            "total_information": float(total_information),
            "mean_information": float(mean_information),
            "std_information": float(std_information),
            "information_efficiency": float(information_efficiency),
            "information_uniformity": float(information_uniformity),
            "quality_score": float(quality_score),
        }

    def _evaluate_clustering_quality(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """クラスタリング品質の評価"""
        min_samples = self.optimization_config["quality_metrics"][
            "silhouette_min_samples"
        ]
        if embeddings.shape[0] < min_samples:
            return {"error": f"サンプル数不足 ({embeddings.shape[0]} < {min_samples})"}

        try:
            # 複数のクラスタ数でテスト
            n_samples = embeddings.shape[0]
            cluster_numbers = [2, 3, max(2, min(8, int(np.sqrt(n_samples))))]

            clustering_results = {}

            for n_clusters in cluster_numbers:
                if n_clusters >= n_samples:
                    continue

                # K-meansクラスタリング
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)

                # シルエット係数
                silhouette_avg = silhouette_score(embeddings, cluster_labels)

                # クラスタ内平均距離（慣性の正規化）
                inertia_normalized = kmeans.inertia_ / n_samples

                clustering_results[n_clusters] = {
                    "silhouette_score": float(silhouette_avg),
                    "inertia_normalized": float(inertia_normalized),
                    "score": float(silhouette_avg / (1.0 + inertia_normalized)),
                }

            if not clustering_results:
                return {"error": "クラスタリング分析を実行できませんでした"}

            # 最良の結果を選択
            best_clustering = max(
                clustering_results.items(), key=lambda x: x[1]["score"]
            )

            return {
                "clustering_results": clustering_results,
                "best_n_clusters": best_clustering[0],
                "best_silhouette_score": best_clustering[1]["silhouette_score"],
                "quality_score": best_clustering[1]["score"],
            }

        except Exception as e:
            logger.error(f"クラスタリング品質評価エラー: {e}")
            return {"error": str(e)}

    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化結果の要約情報を取得

        Returns:
            最適化要約情報の辞書
        """
        if not self.optimization_results:
            raise ValueError("最適化分析が実行されていません。")

        summary = {
            "optimization_completed": True,
            "results_summary": {},
            "recommendations": {},
        }

        # 次元分析結果の要約
        dimension_results = [
            k
            for k in self.optimization_results.keys()
            if k.startswith("dimension_analysis")
        ]
        if dimension_results:
            for result_key in dimension_results:
                result = self.optimization_results[result_key]
                node_type = result["node_type"]

                summary["results_summary"][f"dimension_analysis_{node_type}"] = {
                    "original_dimensions": result["original_dimensions"],
                    "recommended_dimension": result.get("recommended_dimension"),
                    "evaluation_methods": list(result["evaluation_results"].keys()),
                }

                if "recommended_dimension" in result:
                    summary["recommendations"][f"optimal_dimensions_{node_type}"] = (
                        result["recommended_dimension"]
                    )

        # アテンション分析結果の要約
        if "attention_analysis" in self.optimization_results:
            attention_result = self.optimization_results["attention_analysis"]

            summary["results_summary"]["attention_analysis"] = {
                "layers_analyzed": len(attention_result.get("layer_analysis", {})),
                "edge_types_analyzed": len(
                    attention_result.get("edge_type_analysis", {})
                ),
            }

            if "attention_analysis" in attention_result:
                overall = attention_result["attention_analysis"].get(
                    "overall_statistics", {}
                )
                if overall:
                    summary["recommendations"]["attention_concentration"] = overall.get(
                        "attention_concentration"
                    )

        # 品質評価結果の要約
        quality_results = [
            k
            for k in self.optimization_results.keys()
            if k.startswith("quality_evaluation")
        ]
        if quality_results:
            for result_key in quality_results:
                result = self.optimization_results[result_key]
                node_type = result["node_type"]

                summary["results_summary"][f"quality_evaluation_{node_type}"] = {
                    "overall_quality_score": result["overall_quality_score"],
                    "embedding_shape": result["embedding_shape"],
                }

                if result["overall_quality_score"] is not None:
                    summary["recommendations"][f"quality_score_{node_type}"] = result[
                        "overall_quality_score"
                    ]

        return summary

    def save(self, filepath: str) -> None:
        """最適化器を保存

        Args:
            filepath: 保存先ファイルパス
        """
        save_data = {
            "config": self.config,
            "optimization_results": self.optimization_results,
            "original_dimensions": self.original_dimensions,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"GATOptimizer保存完了: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "GATOptimizer":
        """最適化器を読み込み

        Args:
            filepath: 読み込み元ファイルパス

        Returns:
            読み込まれたGATOptimizerインスタンス
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        instance = cls(config=save_data["config"])
        instance.optimization_results = save_data["optimization_results"]
        instance.original_dimensions = save_data["original_dimensions"]
        instance.is_fitted = save_data["is_fitted"]

        logger.info(f"GATOptimizer読み込み完了: {filepath}")
        return instance

    def generate_optimization_report(self, output_path: Optional[str] = None) -> str:
        """最適化分析レポートを生成

        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）

        Returns:
            生成されたレポートファイルのパス
        """
        if not self.optimization_results:
            raise ValueError("最適化分析が実行されていません。")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"gat_optimization_report_{timestamp}.txt"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_optimization_summary()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("GAT特徴量最適化分析レポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 要約
            f.write("【分析要約】\n")
            for key, value in summary["results_summary"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # 推奨事項
            if summary["recommendations"]:
                f.write("【推奨事項】\n")
                for key, value in summary["recommendations"].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # 詳細結果
            f.write("【詳細分析結果】\n")
            for result_key, result_data in self.optimization_results.items():
                f.write(f"\n--- {result_key} ---\n")
                self._write_result_details(f, result_data)

        logger.info(f"最適化レポート生成完了: {output_path}")
        return str(output_path)

    def _write_result_details(self, file_handle, result_data: Dict[str, Any]) -> None:
        """結果詳細をファイルに書き込み"""
        for key, value in result_data.items():
            if isinstance(value, dict):
                file_handle.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    file_handle.write(f"    {sub_key}: {sub_value}\n")
            elif isinstance(value, list):
                file_handle.write(f"  {key}: {', '.join(map(str, value))}\n")
            else:
                file_handle.write(f"  {key}: {value}\n")
