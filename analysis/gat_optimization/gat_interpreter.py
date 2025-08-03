"""
GAT特徴量解釈器
==============

Graph Attention Network特徴量の意味解釈・パターン特定・協力関係可視化を実装します。
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class GATInterpreter:
    """GAT特徴量解釈器

    各GAT次元の意味解釈、重要なグラフパターンの特定、協力関係の可視化、
    GAT特徴量と基本特徴量の相関分析を実装。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（解釈手法、パラメータなど）
        """
        self.config = config or {}

        # 解釈設定
        self.interpretation_config = self.config.get(
            "interpretation_config",
            {
                "dimension_interpretation": {
                    "top_k_features": 10,  # 各次元で重要な上位特徴量数
                    "correlation_threshold": 0.3,  # 相関分析の閾値
                    "clustering_method": "kmeans",  # クラスタリング手法
                    "n_clusters": 5,  # クラスタ数
                },
                "pattern_analysis": {
                    "similarity_threshold": 0.7,  # 類似パターンの閾値
                    "min_pattern_size": 3,  # 最小パターンサイズ
                    "max_patterns": 20,  # 最大パターン数
                },
                "collaboration_analysis": {
                    "edge_weight_threshold": 0.1,  # エッジ重みの閾値
                    "community_detection": True,  # コミュニティ検出を実行するか
                    "centrality_measures": ["degree", "betweenness", "eigenvector"],
                },
                "visualization": {
                    "dimension_reduction": "2d",  # '2d' or '3d'
                    "plot_size": (12, 8),  # プロットサイズ
                    "color_scheme": "viridis",  # カラースキーム
                },
            },
        )

        # 初期化
        self.embeddings = None
        self.basic_features = None
        self.graph_data = None
        self.interpretation_results = {}
        self.is_fitted = False

        logger.info("GATInterpreter初期化完了")

    def load_data(
        self,
        embeddings: Dict[str, torch.Tensor],
        basic_features: Optional[Dict[str, np.ndarray]] = None,
        graph_data: Optional[Any] = None,
    ) -> "GATInterpreter":
        """GAT埋め込みと関連データを読み込み

        Args:
            embeddings: GAT埋め込み辞書
            basic_features: 基本特徴量辞書（オプション）
            graph_data: グラフデータ（オプション）

        Returns:
            自身のインスタンス
        """
        try:
            # 埋め込みをnumpy配列に変換
            self.embeddings = {}
            for node_type, emb in embeddings.items():
                if isinstance(emb, torch.Tensor):
                    self.embeddings[node_type] = emb.detach().cpu().numpy()
                else:
                    self.embeddings[node_type] = np.array(emb)

            self.basic_features = basic_features or {}
            self.graph_data = graph_data
            self.is_fitted = True

            logger.info(f"データ読み込み完了: 埋め込み={list(self.embeddings.keys())}")
            return self

        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise

    def interpret_dimensions(
        self, node_type: str = "dev", basic_feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """各GAT次元の意味解釈

        Args:
            node_type: 分析対象のノードタイプ
            basic_feature_names: 基本特徴量の名前リスト

        Returns:
            次元解釈結果
        """
        if not self.is_fitted or node_type not in self.embeddings:
            raise ValueError(
                f"データが読み込まれていないか、ノードタイプ '{node_type}' が見つかりません"
            )

        embeddings = self.embeddings[node_type]
        n_dimensions = embeddings.shape[1]

        results = {
            "node_type": node_type,
            "n_dimensions": n_dimensions,
            "dimension_interpretations": {},
            "dimension_clusters": {},
            "correlation_analysis": {},
        }

        config = self.interpretation_config["dimension_interpretation"]

        # 各次元の解釈
        for dim_idx in range(n_dimensions):
            dim_values = embeddings[:, dim_idx]

            interpretation = self._interpret_single_dimension(
                dim_values, dim_idx, node_type
            )
            results["dimension_interpretations"][f"dim_{dim_idx}"] = interpretation

        # 次元のクラスタリング
        results["dimension_clusters"] = self._cluster_dimensions(embeddings, config)

        # 基本特徴量との相関分析
        if node_type in self.basic_features and basic_feature_names:
            results["correlation_analysis"] = (
                self._analyze_correlations_with_basic_features(
                    embeddings, self.basic_features[node_type], basic_feature_names
                )
            )

        self.interpretation_results[f"dimension_interpretation_{node_type}"] = results
        logger.info(f"次元解釈完了: {node_type} ({n_dimensions}次元)")

        return results

    def _interpret_single_dimension(
        self, dim_values: np.ndarray, dim_idx: int, node_type: str
    ) -> Dict[str, Any]:
        """単一次元の解釈"""
        interpretation = {
            "dimension_index": dim_idx,
            "statistics": {
                "mean": float(np.mean(dim_values)),
                "std": float(np.std(dim_values)),
                "min": float(np.min(dim_values)),
                "max": float(np.max(dim_values)),
                "skewness": float(stats.skew(dim_values)),
                "kurtosis": float(stats.kurtosis(dim_values)),
            },
            "distribution_type": self._identify_distribution_type(dim_values),
            "extreme_values": self._find_extreme_values(dim_values),
            "semantic_category": self._infer_semantic_category(
                dim_values, dim_idx, node_type
            ),
        }

        return interpretation

    def _identify_distribution_type(self, values: np.ndarray) -> str:
        """分布タイプの特定"""
        # 正規性検定
        _, p_normal = stats.normaltest(values)

        # 歪度と尖度による分類
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)

        if p_normal > 0.05:
            return "normal"
        elif abs(skewness) > 1.0:
            return "skewed_right" if skewness > 0 else "skewed_left"
        elif kurtosis > 3.0:
            return "heavy_tailed"
        elif kurtosis < -1.0:
            return "light_tailed"
        else:
            return "unknown"

    def _find_extreme_values(
        self, values: np.ndarray, percentile: float = 95
    ) -> Dict[str, Any]:
        """極値の特定"""
        threshold_high = np.percentile(values, percentile)
        threshold_low = np.percentile(values, 100 - percentile)

        high_indices = np.where(values >= threshold_high)[0]
        low_indices = np.where(values <= threshold_low)[0]

        return {
            "high_threshold": float(threshold_high),
            "low_threshold": float(threshold_low),
            "high_value_indices": high_indices.tolist(),
            "low_value_indices": low_indices.tolist(),
            "n_high_values": len(high_indices),
            "n_low_values": len(low_indices),
        }

    def _infer_semantic_category(
        self, values: np.ndarray, dim_idx: int, node_type: str
    ) -> str:
        """セマンティックカテゴリの推定"""
        # 統計的特性に基づく簡単な分類
        mean_val = np.mean(values)
        std_val = np.std(values)

        # 値の範囲と分散に基づく分類
        if std_val < 0.1:
            return "constant_like"
        elif abs(mean_val) < 0.1 and std_val < 0.5:
            return "centered_low_variance"
        elif abs(mean_val) > 1.0:
            return "shifted_high_magnitude"
        elif std_val > 1.0:
            return "high_variance"
        else:
            return "moderate_pattern"

    def _cluster_dimensions(
        self, embeddings: np.ndarray, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """次元のクラスタリング"""
        try:
            # 次元を転置（次元×サンプル → サンプル×次元）
            dimensions_as_features = embeddings.T

            n_clusters = min(config["n_clusters"], dimensions_as_features.shape[0])

            if config["clustering_method"] == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(dimensions_as_features)

                # クラスタ中心の取得
                cluster_centers = clusterer.cluster_centers_

            else:
                # デフォルトはKMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(dimensions_as_features)
                cluster_centers = clusterer.cluster_centers_

            # クラスタごとの次元を分類
            dimension_clusters = {}
            for cluster_id in range(n_clusters):
                cluster_dimensions = np.where(cluster_labels == cluster_id)[0].tolist()

                dimension_clusters[f"cluster_{cluster_id}"] = {
                    "dimensions": cluster_dimensions,
                    "size": len(cluster_dimensions),
                    "center": (
                        cluster_centers[cluster_id].tolist()
                        if cluster_centers is not None
                        else None
                    ),
                }

            return {
                "method": config["clustering_method"],
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "dimension_clusters": dimension_clusters,
            }

        except Exception as e:
            logger.error(f"次元クラスタリングエラー: {e}")
            return {"error": str(e)}

    def _analyze_correlations_with_basic_features(
        self,
        embeddings: np.ndarray,
        basic_features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """基本特徴量との相関分析"""
        try:
            # 相関行列を計算
            correlations = np.corrcoef(embeddings.T, basic_features.T)

            # GAT次元と基本特徴量の相関部分を抽出
            n_gat_dims = embeddings.shape[1]
            n_basic_features = basic_features.shape[1]

            gat_basic_correlations = correlations[
                :n_gat_dims, n_gat_dims : n_gat_dims + n_basic_features
            ]

            config = self.interpretation_config["dimension_interpretation"]
            threshold = config["correlation_threshold"]

            # 高相関ペアの特定
            high_correlations = {}
            for gat_dim in range(n_gat_dims):
                high_corr_indices = np.where(
                    np.abs(gat_basic_correlations[gat_dim, :]) >= threshold
                )[0]

                if len(high_corr_indices) > 0:
                    high_correlations[f"gat_dim_{gat_dim}"] = [
                        {
                            "feature_name": feature_names[idx],
                            "correlation": float(gat_basic_correlations[gat_dim, idx]),
                        }
                        for idx in high_corr_indices
                    ]

            return {
                "correlation_matrix_shape": gat_basic_correlations.shape,
                "correlation_threshold": threshold,
                "high_correlations": high_correlations,
                "max_correlation": float(np.max(np.abs(gat_basic_correlations))),
                "mean_absolute_correlation": float(
                    np.mean(np.abs(gat_basic_correlations))
                ),
            }

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return {"error": str(e)}

    def identify_graph_patterns(self, node_type: str = "dev") -> Dict[str, Any]:
        """重要なグラフパターンの特定

        Args:
            node_type: 分析対象のノードタイプ

        Returns:
            グラフパターン分析結果
        """
        if not self.is_fitted or node_type not in self.embeddings:
            raise ValueError(
                f"データが読み込まれていないか、ノードタイプ '{node_type}' が見つかりません"
            )

        embeddings = self.embeddings[node_type]
        config = self.interpretation_config["pattern_analysis"]

        results = {
            "node_type": node_type,
            "similarity_patterns": self._find_similarity_patterns(embeddings, config),
            "cluster_patterns": self._find_cluster_patterns(embeddings, config),
            "outlier_patterns": self._find_outlier_patterns(embeddings, config),
        }

        self.interpretation_results[f"graph_patterns_{node_type}"] = results
        logger.info(f"グラフパターン特定完了: {node_type}")

        return results

    def _find_similarity_patterns(
        self, embeddings: np.ndarray, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """類似性パターンの発見"""
        similarity_matrix = cosine_similarity(embeddings)
        threshold = config["similarity_threshold"]
        min_size = config["min_pattern_size"]
        max_patterns = config["max_patterns"]

        # 高類似度ペアの特定
        high_similarity_pairs = []
        n_nodes = similarity_matrix.shape[0]

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if similarity_matrix[i, j] >= threshold:
                    high_similarity_pairs.append(
                        {
                            "node_i": i,
                            "node_j": j,
                            "similarity": float(similarity_matrix[i, j]),
                        }
                    )

        # 類似度でソート
        high_similarity_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        # 類似グループの形成
        similarity_groups = []
        used_nodes = set()

        for pair in high_similarity_pairs[:max_patterns]:
            node_i, node_j = pair["node_i"], pair["node_j"]

            if node_i not in used_nodes and node_j not in used_nodes:
                # 新しいグループを作成
                group = {
                    "nodes": [node_i, node_j],
                    "avg_similarity": pair["similarity"],
                    "size": 2,
                }

                # 近い他のノードを追加
                for k in range(n_nodes):
                    if k not in [node_i, node_j] and k not in used_nodes:
                        sim_to_group = np.mean(
                            [similarity_matrix[k, node_i], similarity_matrix[k, node_j]]
                        )
                        if sim_to_group >= threshold:
                            group["nodes"].append(k)
                            group["size"] += 1

                if group["size"] >= min_size:
                    similarity_groups.append(group)
                    used_nodes.update(group["nodes"])

        return {
            "threshold": threshold,
            "total_high_similarity_pairs": len(high_similarity_pairs),
            "similarity_groups": similarity_groups,
            "n_groups": len(similarity_groups),
        }

    def _find_cluster_patterns(
        self, embeddings: np.ndarray, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """クラスタパターンの発見"""
        try:
            # K-meansクラスタリング
            n_clusters = min(8, max(2, int(np.sqrt(embeddings.shape[0]))))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # 各クラスタの特性分析
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_embeddings = embeddings[cluster_mask]

                if len(cluster_embeddings) == 0:
                    continue

                # クラスタ内統計
                cluster_center = np.mean(cluster_embeddings, axis=0)
                cluster_std = np.std(cluster_embeddings, axis=0)
                cluster_size = len(cluster_embeddings)

                # クラスタの凝集度（内部距離の平均）
                if cluster_size > 1:
                    internal_distances = cosine_similarity(cluster_embeddings)
                    cohesion = np.mean(
                        internal_distances[
                            np.triu_indices_from(internal_distances, k=1)
                        ]
                    )
                else:
                    cohesion = 1.0

                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "size": cluster_size,
                    "nodes": np.where(cluster_mask)[0].tolist(),
                    "center": cluster_center.tolist(),
                    "cohesion": float(cohesion),
                    "mean_std": float(np.mean(cluster_std)),
                }

            return {
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_analysis": cluster_analysis,
            }

        except Exception as e:
            logger.error(f"クラスタパターン分析エラー: {e}")
            return {"error": str(e)}

    def _find_outlier_patterns(
        self, embeddings: np.ndarray, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """異常値パターンの発見"""
        try:
            # 各ノードの近傍との平均類似度を計算
            similarity_matrix = cosine_similarity(embeddings)

            # 自己類似度を除外
            np.fill_diagonal(similarity_matrix, 0)

            # 各ノードの平均類似度
            avg_similarities = np.mean(similarity_matrix, axis=1)

            # 異常値の特定（平均類似度が低いノード）
            threshold = np.percentile(avg_similarities, 10)  # 下位10%
            outlier_indices = np.where(avg_similarities <= threshold)[0]

            # 異常値の詳細分析
            outlier_analysis = {}
            for idx in outlier_indices:
                outlier_analysis[f"node_{idx}"] = {
                    "avg_similarity": float(avg_similarities[idx]),
                    "max_similarity": float(np.max(similarity_matrix[idx, :])),
                    "similarity_std": float(np.std(similarity_matrix[idx, :])),
                    "embedding_norm": float(np.linalg.norm(embeddings[idx])),
                }

            return {
                "threshold": float(threshold),
                "n_outliers": len(outlier_indices),
                "outlier_indices": outlier_indices.tolist(),
                "outlier_analysis": outlier_analysis,
                "avg_similarity_stats": {
                    "mean": float(np.mean(avg_similarities)),
                    "std": float(np.std(avg_similarities)),
                    "min": float(np.min(avg_similarities)),
                    "max": float(np.max(avg_similarities)),
                },
            }

        except Exception as e:
            logger.error(f"異常値パターン分析エラー: {e}")
            return {"error": str(e)}

    def visualize_collaboration_network(
        self, output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """協力関係の可視化

        Args:
            output_dir: 出力ディレクトリ（Noneの場合は現在のディレクトリ）

        Returns:
            生成されたファイルのパス辞書
        """
        if self.graph_data is None:
            logger.warning("グラフデータが利用できません")
            return {}

        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            output_dir = Path(output_dir) if output_dir else Path(".")
            output_dir.mkdir(parents=True, exist_ok=True)

            generated_files = {}

            # 開発者埋め込みの2D可視化
            if "dev" in self.embeddings:
                dev_embeddings = self.embeddings["dev"]

                # 2D次元削減
                if dev_embeddings.shape[1] > 2:
                    if (
                        self.interpretation_config["visualization"][
                            "dimension_reduction"
                        ]
                        == "2d"
                    ):
                        reducer = PCA(n_components=2)
                        embeddings_2d = reducer.fit_transform(dev_embeddings)
                    else:
                        # t-SNEを使用
                        reducer = TSNE(n_components=2, random_state=42)
                        embeddings_2d = reducer.fit_transform(dev_embeddings)
                else:
                    embeddings_2d = dev_embeddings

                # 開発者埋め込みプロット
                fig, ax = plt.subplots(
                    figsize=self.interpretation_config["visualization"]["plot_size"]
                )
                scatter = ax.scatter(
                    embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    c=range(len(embeddings_2d)),
                    cmap=self.interpretation_config["visualization"]["color_scheme"],
                    alpha=0.7,
                )
                ax.set_title("Developer Embeddings Visualization")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                plt.colorbar(scatter, ax=ax, label="Developer ID")

                embedding_plot_path = (
                    output_dir
                    / f'dev_embeddings_2d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                )
                plt.savefig(embedding_plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                generated_files["dev_embeddings_2d"] = str(embedding_plot_path)

            # 協力ネットワークの可視化
            collaboration_plot_path = self._create_collaboration_network_plot(
                output_dir
            )
            if collaboration_plot_path:
                generated_files["collaboration_network"] = collaboration_plot_path

            logger.info(f"可視化完了: {len(generated_files)} ファイル生成")
            return generated_files

        except ImportError:
            logger.error(
                "可視化に必要なライブラリ（matplotlib, networkx）がインストールされていません"
            )
            return {}
        except Exception as e:
            logger.error(f"可視化エラー: {e}")
            return {}

    def _create_collaboration_network_plot(self, output_dir: Path) -> Optional[str]:
        """協力ネットワークプロットの作成"""
        try:
            if not hasattr(self.graph_data, "edge_index_dict"):
                return None

            import matplotlib.pyplot as plt
            import networkx as nx

            # 開発者間協力エッジの取得
            collab_edges = None
            if ("dev", "collaborates", "dev") in self.graph_data.edge_index_dict:
                collab_edges = self.graph_data.edge_index_dict[
                    ("dev", "collaborates", "dev")
                ]

            if collab_edges is None or collab_edges.shape[1] == 0:
                logger.warning("協力関係エッジが見つかりません")
                return None

            # NetworkXグラフの作成
            G = nx.Graph()

            # ノードの追加
            n_devs = self.embeddings["dev"].shape[0] if "dev" in self.embeddings else 0
            G.add_nodes_from(range(n_devs))

            # エッジの追加
            if isinstance(collab_edges, torch.Tensor):
                collab_edges = collab_edges.detach().cpu().numpy()

            edges = [
                (int(collab_edges[0, i]), int(collab_edges[1, i]))
                for i in range(collab_edges.shape[1])
            ]
            G.add_edges_from(edges)

            # レイアウトの計算
            if len(G.nodes()) < 100:
                pos = nx.spring_layout(G, k=1, iterations=50)
            else:
                pos = nx.spring_layout(G, k=3, iterations=20)

            # プロット
            fig, ax = plt.subplots(
                figsize=self.interpretation_config["visualization"]["plot_size"]
            )

            # ノードの描画
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color="lightblue", node_size=50, alpha=0.7
            )

            # エッジの描画
            nx.draw_networkx_edges(
                G, pos, ax=ax, edge_color="gray", alpha=0.5, width=0.5
            )

            ax.set_title("Developer Collaboration Network")
            ax.axis("off")

            network_plot_path = (
                output_dir
                / f'collaboration_network_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
            plt.savefig(network_plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(network_plot_path)

        except Exception as e:
            logger.error(f"協力ネットワーク可視化エラー: {e}")
            return None

    def analyze_feature_correlations(
        self, node_type: str = "dev", basic_feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """GAT特徴量と基本特徴量の相関分析

        Args:
            node_type: 分析対象のノードタイプ
            basic_feature_names: 基本特徴量の名前リスト

        Returns:
            相関分析結果
        """
        if not self.is_fitted or node_type not in self.embeddings:
            raise ValueError(
                f"データが読み込まれていないか、ノードタイプ '{node_type}' が見つかりません"
            )

        if node_type not in self.basic_features:
            logger.warning(f"ノードタイプ '{node_type}' の基本特徴量が見つかりません")
            return {}

        gat_embeddings = self.embeddings[node_type]
        basic_features = self.basic_features[node_type]
        feature_names = basic_feature_names or [
            f"basic_feature_{i}" for i in range(basic_features.shape[1])
        ]

        results = {
            "node_type": node_type,
            "gat_dimensions": gat_embeddings.shape[1],
            "basic_features": basic_features.shape[1],
            "correlation_matrix": None,
            "top_correlations": {},
            "redundancy_analysis": {},
        }

        try:
            # 相関行列の計算
            all_features = np.hstack([gat_embeddings, basic_features])
            correlation_matrix = np.corrcoef(all_features.T)

            n_gat = gat_embeddings.shape[1]
            n_basic = basic_features.shape[1]

            # GAT特徴量と基本特徴量間の相関部分を抽出
            gat_basic_corr = correlation_matrix[:n_gat, n_gat : n_gat + n_basic]

            results["correlation_matrix"] = {
                "gat_basic_correlations": gat_basic_corr.tolist(),
                "shape": gat_basic_corr.shape,
                "max_correlation": float(np.max(np.abs(gat_basic_corr))),
                "mean_abs_correlation": float(np.mean(np.abs(gat_basic_corr))),
            }

            # 高相関ペアの特定
            threshold = self.interpretation_config["dimension_interpretation"][
                "correlation_threshold"
            ]

            for gat_dim in range(n_gat):
                high_corr_indices = np.where(
                    np.abs(gat_basic_corr[gat_dim, :]) >= threshold
                )[0]

                if len(high_corr_indices) > 0:
                    correlations = []
                    for basic_idx in high_corr_indices:
                        correlations.append(
                            {
                                "feature_name": feature_names[basic_idx],
                                "correlation": float(
                                    gat_basic_corr[gat_dim, basic_idx]
                                ),
                                "abs_correlation": float(
                                    np.abs(gat_basic_corr[gat_dim, basic_idx])
                                ),
                            }
                        )

                    # 絶対値でソート
                    correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
                    results["top_correlations"][f"gat_dim_{gat_dim}"] = correlations[
                        :10
                    ]

            # 冗長性分析
            results["redundancy_analysis"] = self._analyze_feature_redundancy(
                gat_embeddings, basic_features, feature_names, threshold
            )

            self.interpretation_results[f"correlation_analysis_{node_type}"] = results
            logger.info(f"相関分析完了: {node_type}")

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            results["error"] = str(e)

        return results

    def _analyze_feature_redundancy(
        self,
        gat_embeddings: np.ndarray,
        basic_features: np.ndarray,
        feature_names: List[str],
        threshold: float,
    ) -> Dict[str, Any]:
        """特徴量冗長性の分析"""
        try:
            # GAT特徴量間の相関
            gat_corr_matrix = np.corrcoef(gat_embeddings.T)

            # 高相関なGAT特徴量ペア
            n_gat = gat_embeddings.shape[1]
            high_gat_corr_pairs = []

            for i in range(n_gat):
                for j in range(i + 1, n_gat):
                    if np.abs(gat_corr_matrix[i, j]) >= threshold:
                        high_gat_corr_pairs.append(
                            {
                                "gat_dim_i": i,
                                "gat_dim_j": j,
                                "correlation": float(gat_corr_matrix[i, j]),
                            }
                        )

            # 基本特徴量間の相関
            basic_corr_matrix = np.corrcoef(basic_features.T)
            n_basic = basic_features.shape[1]
            high_basic_corr_pairs = []

            for i in range(n_basic):
                for j in range(i + 1, n_basic):
                    if np.abs(basic_corr_matrix[i, j]) >= threshold:
                        high_basic_corr_pairs.append(
                            {
                                "feature_i": feature_names[i],
                                "feature_j": feature_names[j],
                                "correlation": float(basic_corr_matrix[i, j]),
                            }
                        )

            return {
                "gat_redundant_pairs": high_gat_corr_pairs,
                "basic_redundant_pairs": high_basic_corr_pairs,
                "n_gat_redundant": len(high_gat_corr_pairs),
                "n_basic_redundant": len(high_basic_corr_pairs),
            }

        except Exception as e:
            logger.error(f"冗長性分析エラー: {e}")
            return {"error": str(e)}

    def get_interpretation_summary(self) -> Dict[str, Any]:
        """解釈結果の要約情報を取得

        Returns:
            解釈要約情報の辞書
        """
        if not self.interpretation_results:
            raise ValueError("解釈分析が実行されていません。")

        summary = {
            "interpretation_completed": True,
            "results_summary": {},
            "key_findings": {},
        }

        # 次元解釈結果の要約
        dimension_results = [
            k
            for k in self.interpretation_results.keys()
            if k.startswith("dimension_interpretation")
        ]
        if dimension_results:
            for result_key in dimension_results:
                result = self.interpretation_results[result_key]
                node_type = result["node_type"]

                summary["results_summary"][f"dimension_interpretation_{node_type}"] = {
                    "n_dimensions": result["n_dimensions"],
                    "n_dimension_clusters": result["dimension_clusters"].get(
                        "n_clusters", 0
                    ),
                    "has_correlation_analysis": bool(
                        result.get("correlation_analysis")
                    ),
                }

        # パターン分析結果の要約
        pattern_results = [
            k
            for k in self.interpretation_results.keys()
            if k.startswith("graph_patterns")
        ]
        if pattern_results:
            for result_key in pattern_results:
                result = self.interpretation_results[result_key]
                node_type = result["node_type"]

                summary["results_summary"][f"graph_patterns_{node_type}"] = {
                    "n_similarity_groups": result["similarity_patterns"].get(
                        "n_groups", 0
                    ),
                    "n_clusters": result["cluster_patterns"].get("n_clusters", 0),
                    "n_outliers": result["outlier_patterns"].get("n_outliers", 0),
                }

        # 相関分析結果の要約
        correlation_results = [
            k
            for k in self.interpretation_results.keys()
            if k.startswith("correlation_analysis")
        ]
        if correlation_results:
            for result_key in correlation_results:
                result = self.interpretation_results[result_key]
                node_type = result["node_type"]

                if "correlation_matrix" in result:
                    summary["results_summary"][f"correlation_analysis_{node_type}"] = {
                        "max_correlation": result["correlation_matrix"].get(
                            "max_correlation", 0
                        ),
                        "mean_abs_correlation": result["correlation_matrix"].get(
                            "mean_abs_correlation", 0
                        ),
                        "n_high_correlations": len(result.get("top_correlations", {})),
                    }

        # 主要な発見事項
        summary["key_findings"] = self._extract_key_findings()

        return summary

    def _extract_key_findings(self) -> Dict[str, Any]:
        """主要な発見事項の抽出"""
        findings = {}

        try:
            # 高相関次元の特定
            for result_key, result in self.interpretation_results.items():
                if result_key.startswith("correlation_analysis"):
                    node_type = result["node_type"]

                    if "correlation_matrix" in result:
                        max_corr = result["correlation_matrix"].get(
                            "max_correlation", 0
                        )
                        if max_corr > 0.5:
                            findings[f"high_correlation_{node_type}"] = {
                                "finding": "GAT特徴量と基本特徴量間に高い相関",
                                "max_correlation": max_corr,
                            }

            # 異常値パターンの特定
            for result_key, result in self.interpretation_results.items():
                if result_key.startswith("graph_patterns"):
                    node_type = result["node_type"]

                    n_outliers = result["outlier_patterns"].get("n_outliers", 0)
                    if n_outliers > 0:
                        findings[f"outliers_{node_type}"] = {
                            "finding": "異常値パターンを検出",
                            "n_outliers": n_outliers,
                        }

        except Exception as e:
            logger.error(f"主要発見事項抽出エラー: {e}")

        return findings

    def save(self, filepath: str) -> None:
        """解釈器を保存

        Args:
            filepath: 保存先ファイルパス
        """
        save_data = {
            "config": self.config,
            "interpretation_results": self.interpretation_results,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"GATInterpreter保存完了: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "GATInterpreter":
        """解釈器を読み込み

        Args:
            filepath: 読み込み元ファイルパス

        Returns:
            読み込まれたGATInterpreterインスタンス
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        instance = cls(config=save_data["config"])
        instance.interpretation_results = save_data["interpretation_results"]
        instance.is_fitted = save_data["is_fitted"]

        logger.info(f"GATInterpreter読み込み完了: {filepath}")
        return instance

    def generate_interpretation_report(self, output_path: Optional[str] = None) -> str:
        """解釈分析レポートを生成

        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）

        Returns:
            生成されたレポートファイルのパス
        """
        if not self.interpretation_results:
            raise ValueError("解釈分析が実行されていません。")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"gat_interpretation_report_{timestamp}.txt"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_interpretation_summary()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("GAT特徴量解釈分析レポート\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 要約
            f.write("【分析要約】\n")
            for key, value in summary["results_summary"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # 主要発見事項
            if summary["key_findings"]:
                f.write("【主要発見事項】\n")
                for key, finding in summary["key_findings"].items():
                    f.write(f"  {key}: {finding['finding']}\n")
                f.write("\n")

            # 詳細結果
            f.write("【詳細解釈結果】\n")
            for result_key, result_data in self.interpretation_results.items():
                f.write(f"\n--- {result_key} ---\n")
                self._write_interpretation_details(f, result_data)

        logger.info(f"解釈レポート生成完了: {output_path}")
        return str(output_path)

    def _write_interpretation_details(
        self, file_handle, result_data: Dict[str, Any]
    ) -> None:
        """解釈結果詳細をファイルに書き込み"""
        for key, value in result_data.items():
            if isinstance(value, dict):
                file_handle.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    if (
                        isinstance(sub_value, (dict, list))
                        and len(str(sub_value)) > 100
                    ):
                        file_handle.write(f"    {sub_key}: [詳細データ - 省略]\n")
                    else:
                        file_handle.write(f"    {sub_key}: {sub_value}\n")
            elif isinstance(value, list) and len(str(value)) > 100:
                file_handle.write(
                    f"  {key}: [リストデータ - 省略 (長さ: {len(value)})]\n"
                )
            else:
                file_handle.write(f"  {key}: {value}\n")
