"""
特徴量スケーリング器
==================

対数変換、標準化、min-max正規化、ロバストスケーリングの実装。
特徴量タイプに応じた自動スケーリング選択機能を提供します。
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, RobustScaler,
                                   StandardScaler)

logger = logging.getLogger(__name__)


class FeatureScaler:
    """特徴量スケーリング器

    対数変換、標準化、min-max正規化、ロバストスケーリングの実装。
    特徴量タイプに応じた自動スケーリング選択機能を提供。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（スケーリング手法、閾値など）
        """
        self.config = config or {}

        # スケーリング手法の設定
        self.scaling_methods = self.config.get(
            "scaling_methods",
            {
                "log_transform": {"threshold": 0.0, "offset": 1.0},
                "standard": {"with_mean": True, "with_std": True},
                "minmax": {"feature_range": (0, 1)},
                "robust": {"quantile_range": (25.0, 75.0)},
                "power": {"method": "yeo-johnson", "standardize": True},
            },
        )

        # 特徴量タイプの自動判定閾値
        self.auto_detection_thresholds = self.config.get(
            "auto_detection_thresholds",
            {
                "categorical_unique_ratio": 0.05,  # ユニーク値比率が5%以下ならカテゴリカル
                "binary_unique_count": 2,  # ユニーク値が2個ならバイナリ
                "skewness_threshold": 2.0,  # 歪度が2以上なら対数変換を検討
                "outlier_ratio_threshold": 0.1,  # 外れ値比率が10%以上ならロバストスケーリング
            },
        )

        # 時系列特徴量のパターン
        self.temporal_patterns = self.config.get(
            "temporal_patterns",
            [
                "time",
                "date",
                "hour",
                "day",
                "week",
                "month",
                "year",
                "created",
                "updated",
                "since",
                "age",
                "duration",
            ],
        )

        # 初期化
        self.scalers = {}
        self.encoders = {}
        self.feature_types = {}
        self.scaling_strategies = {}
        self.is_fitted = False

        logger.info(f"FeatureScaler初期化完了")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ) -> "FeatureScaler":
        """スケーラーを学習データに適合

        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト
            feature_types: 特徴量タイプの辞書（手動指定）

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

        self.feature_names = feature_names
        n_features = X_array.shape[1]

        if len(feature_names) != n_features:
            raise ValueError(
                f"特徴量名の数({len(feature_names)})とデータの次元数({n_features})が一致しません"
            )

        # 特徴量タイプの自動判定または手動設定
        if feature_types:
            self.feature_types = feature_types
        else:
            self.feature_types = self._detect_feature_types(X_array, feature_names)

        # 各特徴量に対するスケーリング戦略の決定
        self.scaling_strategies = self._determine_scaling_strategies(
            X_array, feature_names
        )

        # スケーラーの学習
        self._fit_scalers(X_array, feature_names)

        self.is_fitted = True
        logger.info(f"FeatureScaler学習完了: {len(feature_names)}特徴量")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """特徴量をスケーリング変換

        Args:
            X: 変換する特徴量データ

        Returns:
            スケーリング後の特徴量データ
        """
        if not self.is_fitted:
            raise ValueError(
                "スケーラーが学習されていません。先にfit()を呼び出してください。"
            )

        # データ形式の統一
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        if X_array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"入力データの次元数({X_array.shape[1]})が学習時({len(self.feature_names)})と異なります"
            )

        # 変換の実行
        X_scaled = self._apply_scaling(X_array)

        logger.debug(f"特徴量スケーリング変換完了: {X_scaled.shape}")
        return X_scaled

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """学習と変換を同時実行

        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト
            feature_types: 特徴量タイプの辞書

        Returns:
            スケーリング後の特徴量データ
        """
        return self.fit(X, feature_names, feature_types).transform(X)

    def _detect_feature_types(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, str]:
        """特徴量タイプの自動判定

        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト

        Returns:
            特徴量タイプの辞書
        """
        feature_types = {}

        for i, name in enumerate(feature_names):
            values = X[:, i]
            clean_values = values[~np.isnan(values)]

            if len(clean_values) == 0:
                feature_types[name] = "constant"
                continue

            unique_values = np.unique(clean_values)
            unique_count = len(unique_values)
            unique_ratio = unique_count / len(clean_values)

            # 特徴量名からのヒント
            name_lower = name.lower()

            # 時系列特徴量の判定
            if any(pattern in name_lower for pattern in self.temporal_patterns):
                feature_types[name] = "temporal"
            # バイナリ特徴量の判定
            elif unique_count == self.auto_detection_thresholds["binary_unique_count"]:
                if set(unique_values).issubset({0, 1}) or set(unique_values).issubset(
                    {0.0, 1.0}
                ):
                    feature_types[name] = "binary"
                else:
                    feature_types[name] = "categorical"
            # カテゴリカル特徴量の判定
            elif (
                unique_ratio
                <= self.auto_detection_thresholds["categorical_unique_ratio"]
            ):
                feature_types[name] = "categorical"
            # 数値特徴量の詳細判定
            else:
                # 歪度の計算
                if len(clean_values) > 3:
                    skewness = abs(stats.skew(clean_values))

                    # 外れ値の比率
                    q1, q3 = np.percentile(clean_values, [25, 75])
                    iqr = q3 - q1
                    outliers = clean_values[
                        (clean_values < q1 - 1.5 * iqr)
                        | (clean_values > q3 + 1.5 * iqr)
                    ]
                    outlier_ratio = len(outliers) / len(clean_values)

                    # 数値特徴量のサブタイプ判定
                    if skewness > self.auto_detection_thresholds["skewness_threshold"]:
                        feature_types[name] = "skewed_numerical"
                    elif (
                        outlier_ratio
                        > self.auto_detection_thresholds["outlier_ratio_threshold"]
                    ):
                        feature_types[name] = "outlier_prone_numerical"
                    else:
                        feature_types[name] = "numerical"
                else:
                    feature_types[name] = "numerical"

        return feature_types

    def _determine_scaling_strategies(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Dict[str, str]:
        """各特徴量のスケーリング戦略を決定

        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト

        Returns:
            スケーリング戦略の辞書
        """
        strategies = {}

        for i, name in enumerate(feature_names):
            feature_type = self.feature_types[name]
            values = X[:, i]

            if feature_type == "constant":
                strategies[name] = "none"
            elif feature_type == "binary":
                strategies[name] = "none"  # バイナリ特徴量はそのまま
            elif feature_type == "categorical":
                strategies[name] = "onehot"
            elif feature_type == "temporal":
                strategies[name] = "temporal_normalize"
            elif feature_type == "skewed_numerical":
                # 正の値のみなら対数変換、そうでなければPower変換
                clean_values = values[~np.isnan(values)]
                if np.all(clean_values > 0):
                    strategies[name] = "log_transform"
                else:
                    strategies[name] = "power_transform"
            elif feature_type == "outlier_prone_numerical":
                strategies[name] = "robust"
            else:  # numerical
                strategies[name] = "standard"

        return strategies

    def _fit_scalers(self, X: np.ndarray, feature_names: List[str]) -> None:
        """各特徴量に対するスケーラーを学習

        Args:
            X: 特徴量データ
            feature_names: 特徴量名のリスト
        """
        for i, name in enumerate(feature_names):
            strategy = self.scaling_strategies[name]
            values = X[:, i].reshape(-1, 1)

            if strategy == "none":
                self.scalers[name] = None
            elif strategy == "standard":
                scaler = StandardScaler(**self.scaling_methods["standard"])
                scaler.fit(values)
                self.scalers[name] = scaler
            elif strategy == "minmax":
                scaler = MinMaxScaler(**self.scaling_methods["minmax"])
                scaler.fit(values)
                self.scalers[name] = scaler
            elif strategy == "robust":
                scaler = RobustScaler(**self.scaling_methods["robust"])
                scaler.fit(values)
                self.scalers[name] = scaler
            elif strategy == "log_transform":
                # 対数変換のパラメータを保存
                clean_values = values[~np.isnan(values)]
                min_val = np.min(clean_values)
                offset = self.scaling_methods["log_transform"]["offset"]
                if min_val <= 0:
                    offset = abs(min_val) + offset
                self.scalers[name] = {"type": "log", "offset": offset}
            elif strategy == "power_transform":
                scaler = PowerTransformer(**self.scaling_methods["power"])
                scaler.fit(values)
                self.scalers[name] = scaler
            elif strategy == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(values)
                self.encoders[name] = encoder
            elif strategy == "temporal_normalize":
                # 時系列特徴量の正規化パラメータ
                clean_values = values[~np.isnan(values)]
                if len(clean_values) > 0:
                    self.scalers[name] = {
                        "type": "temporal",
                        "min": float(np.min(clean_values)),
                        "max": float(np.max(clean_values)),
                        "mean": float(np.mean(clean_values)),
                    }
                else:
                    self.scalers[name] = None

    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """スケーリングを適用

        Args:
            X: 入力特徴量データ

        Returns:
            スケーリング後のデータ
        """
        scaled_features = []

        for i, name in enumerate(self.feature_names):
            strategy = self.scaling_strategies[name]
            values = X[:, i]

            if strategy == "none":
                scaled_features.append(values.reshape(-1, 1))
            elif strategy in ["standard", "minmax", "robust", "power_transform"]:
                scaler = self.scalers[name]
                scaled_values = scaler.transform(values.reshape(-1, 1))
                scaled_features.append(scaled_values)
            elif strategy == "log_transform":
                params = self.scalers[name]
                offset = params["offset"]
                scaled_values = np.log1p(values + offset).reshape(-1, 1)
                scaled_features.append(scaled_values)
            elif strategy == "onehot":
                encoder = self.encoders[name]
                encoded_values = encoder.transform(values.reshape(-1, 1))
                scaled_features.append(encoded_values)
            elif strategy == "temporal_normalize":
                params = self.scalers[name]
                if params is not None:
                    # 相対時間への変換（0-1正規化）
                    min_val, max_val = params["min"], params["max"]
                    if max_val > min_val:
                        scaled_values = (values - min_val) / (max_val - min_val)
                    else:
                        scaled_values = np.zeros_like(values)
                    scaled_features.append(scaled_values.reshape(-1, 1))
                else:
                    scaled_features.append(values.reshape(-1, 1))

        # 結合
        if scaled_features:
            return np.hstack(scaled_features)
        else:
            return X

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """スケーリングを逆変換（可能な場合のみ）

        Args:
            X_scaled: スケーリング後のデータ

        Returns:
            元のスケールに戻したデータ
        """
        if not self.is_fitted:
            raise ValueError("スケーラーが学習されていません。")

        # OneHotエンコーディングがある場合は逆変換が複雑になるため、
        # 簡単な実装として数値特徴量のみ対応
        inverse_features = []
        col_idx = 0

        for i, name in enumerate(self.feature_names):
            strategy = self.scaling_strategies[name]

            if strategy == "onehot":
                encoder = self.encoders[name]
                n_categories = len(encoder.categories_[0])
                encoded_values = X_scaled[:, col_idx : col_idx + n_categories]
                # 最大値のインデックスを取得して逆変換
                original_values = encoder.inverse_transform(encoded_values)
                inverse_features.append(original_values.flatten())
                col_idx += n_categories
            else:
                scaled_values = X_scaled[:, col_idx].reshape(-1, 1)

                if strategy == "none":
                    inverse_features.append(scaled_values.flatten())
                elif strategy in ["standard", "minmax", "robust", "power_transform"]:
                    scaler = self.scalers[name]
                    original_values = scaler.inverse_transform(scaled_values)
                    inverse_features.append(original_values.flatten())
                elif strategy == "log_transform":
                    params = self.scalers[name]
                    offset = params["offset"]
                    original_values = np.expm1(scaled_values.flatten()) - offset
                    inverse_features.append(original_values)
                elif strategy == "temporal_normalize":
                    params = self.scalers[name]
                    if params is not None:
                        min_val, max_val = params["min"], params["max"]
                        original_values = (
                            scaled_values.flatten() * (max_val - min_val) + min_val
                        )
                        inverse_features.append(original_values)
                    else:
                        inverse_features.append(scaled_values.flatten())

                col_idx += 1

        return np.column_stack(inverse_features)

    def get_feature_names_out(self) -> List[str]:
        """変換後の特徴量名を取得

        Returns:
            変換後の特徴量名のリスト
        """
        if not self.is_fitted:
            raise ValueError("スケーラーが学習されていません。")

        feature_names_out = []

        for name in self.feature_names:
            strategy = self.scaling_strategies[name]

            if strategy == "onehot":
                encoder = self.encoders[name]
                categories = encoder.categories_[0]
                for category in categories:
                    feature_names_out.append(f"{name}_{category}")
            else:
                feature_names_out.append(name)

        return feature_names_out

    def get_scaling_info(self) -> Dict[str, Any]:
        """スケーリング情報を取得

        Returns:
            スケーリング情報の辞書
        """
        if not self.is_fitted:
            raise ValueError("スケーラーが学習されていません。")

        info = {
            "feature_types": self.feature_types.copy(),
            "scaling_strategies": self.scaling_strategies.copy(),
            "n_features_in": len(self.feature_names),
            "n_features_out": len(self.get_feature_names_out()),
            "feature_names_in": self.feature_names.copy(),
            "feature_names_out": self.get_feature_names_out(),
        }

        # 各戦略の統計
        strategy_counts = {}
        for strategy in self.scaling_strategies.values():
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        info["strategy_counts"] = strategy_counts

        return info

    def save(self, filepath: str) -> None:
        """スケーラーを保存

        Args:
            filepath: 保存先ファイルパス
        """
        if not self.is_fitted:
            raise ValueError("スケーラーが学習されていません。")

        save_data = {
            "config": self.config,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "scaling_strategies": self.scaling_strategies,
            "scalers": self.scalers,
            "encoders": self.encoders,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"FeatureScaler保存完了: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FeatureScaler":
        """スケーラーを読み込み

        Args:
            filepath: 読み込み元ファイルパス

        Returns:
            読み込まれたFeatureScalerインスタンス
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        instance = cls(config=save_data["config"])
        instance.feature_names = save_data["feature_names"]
        instance.feature_types = save_data["feature_types"]
        instance.scaling_strategies = save_data["scaling_strategies"]
        instance.scalers = save_data["scalers"]
        instance.encoders = save_data["encoders"]
        instance.is_fitted = save_data["is_fitted"]

        logger.info(f"FeatureScaler読み込み完了: {filepath}")
        return instance

    def generate_scaling_report(self, output_path: Optional[str] = None) -> str:
        """スケーリング分析レポートを生成

        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）

        Returns:
            生成されたレポートファイルのパス
        """
        if not self.is_fitted:
            raise ValueError("スケーラーが学習されていません。")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"feature_scaling_report_{timestamp}.txt"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scaling_info = self.get_scaling_info()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("特徴量スケーリング分析レポート\n")
            f.write("=" * 60 + "\n\n")

            # 基本情報
            f.write("【基本情報】\n")
            f.write(f"入力特徴量数: {scaling_info['n_features_in']}\n")
            f.write(f"出力特徴量数: {scaling_info['n_features_out']}\n")
            f.write(
                f"次元変化: {scaling_info['n_features_out'] - scaling_info['n_features_in']:+d}\n\n"
            )

            # スケーリング戦略の統計
            f.write("【スケーリング戦略統計】\n")
            for strategy, count in scaling_info["strategy_counts"].items():
                percentage = count / scaling_info["n_features_in"] * 100
                f.write(f"{strategy}: {count}個 ({percentage:.1f}%)\n")
            f.write("\n")

            # 特徴量タイプ別の詳細
            f.write("【特徴量別詳細】\n")
            type_groups = {}
            for name, ftype in scaling_info["feature_types"].items():
                if ftype not in type_groups:
                    type_groups[ftype] = []
                type_groups[ftype].append(name)

            for ftype, names in type_groups.items():
                f.write(f"\n{ftype} ({len(names)}個):\n")
                for name in names[:10]:  # 最初の10個のみ表示
                    strategy = scaling_info["scaling_strategies"][name]
                    f.write(f"  - {name}: {strategy}\n")
                if len(names) > 10:
                    f.write(f"  ... 他 {len(names) - 10} 個\n")

            # OneHotエンコーディングの詳細
            onehot_features = [
                name
                for name, strategy in scaling_info["scaling_strategies"].items()
                if strategy == "onehot"
            ]
            if onehot_features:
                f.write(f"\n【OneHotエンコーディング詳細】\n")
                for name in onehot_features:
                    encoder = self.encoders[name]
                    categories = encoder.categories_[0]
                    f.write(
                        f"{name}: {len(categories)}カテゴリ → {len(categories)}特徴量\n"
                    )
                    f.write(f"  カテゴリ: {list(categories)}\n")

        logger.info(f"スケーリング分析レポート生成完了: {output_path}")
        return str(output_path)
