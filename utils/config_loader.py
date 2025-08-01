"""
設定ファイルローダー
リファクタリングされた設定ファイルを読み込むためのユーティリティ
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """設定ファイルローダー"""

    def __init__(self, config_root: str = None):
        """
        Args:
            config_root: 設定ファイルのルートディレクトリ
        """
        if config_root is None:
            # デフォルトは現在のファイルの親の親ディレクトリ/configs
            current_dir = Path(__file__).parent
            self.config_root = current_dir.parent / "configs"
        else:
            self.config_root = Path(config_root)

        if not self.config_root.exists():
            raise FileNotFoundError(
                f"設定ディレクトリが見つかりません: {self.config_root}"
            )

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込み、base_configsがある場合は再帰的にマージ

        Args:
            config_path: 設定ファイルのパス（configs/からの相対パス）

        Returns:
            マージされた設定辞書
        """
        full_path = self.config_root / config_path

        if not full_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # base_configsがある場合、再帰的に読み込んでマージ
        if "base_configs" in config:
            base_config = {}
            for base_path in config["base_configs"]:
                base = self.load_config(base_path)
                base_config = self._deep_merge(base_config, base)

            # ベース設定と現在の設定をマージ
            config = self._deep_merge(base_config, config)
            # base_configsキーは不要なので削除
            del config["base_configs"]

        return config

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        辞書を深くマージ

        Args:
            base: ベース辞書
            override: 上書きする辞書

        Returns:
            マージされた辞書
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def resolve_data_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        データパス設定を解決

        Args:
            config: 設定辞書

        Returns:
            パスが解決された設定辞書
        """
        if "env" in config and "data_config" in config["env"]:
            data_config_name = config["env"]["data_config"]

            # data_paths.yamlから該当するデータ設定を取得
            data_paths_config = self.load_config("base/data_paths.yaml")

            if data_config_name in data_paths_config:
                data_config = data_paths_config[data_config_name]

                # env設定にデータパスを追加
                for key, value in data_config.items():
                    if key != "description":
                        config["env"][key] = value

        return config

    def resolve_component_configs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        コンポーネント設定を解決（IRL、RL設定など）

        Args:
            config: 設定辞書

        Returns:
            コンポーネント設定が解決された設定辞書
        """
        # IRL設定を解決
        if "irl" in config and "config" in config["irl"]:
            irl_config_name = config["irl"]["config"]
            irl_settings = self.load_config("base/irl_settings.yaml")

            if irl_config_name in irl_settings:
                irl_base_config = irl_settings[irl_config_name]
                config["irl"] = self._deep_merge(irl_base_config, config["irl"])
                del config["irl"]["config"]  # 参照キーは削除

            # オンライン学習設定も解決
            if "online_learning_config" in config["irl"]:
                online_config_name = config["irl"]["online_learning_config"]
                if online_config_name in irl_settings:
                    online_config = irl_settings[online_config_name]
                    config["irl"]["online_learning"] = online_config
                    del config["irl"]["online_learning_config"]

        # RL設定を解決
        if "rl" in config and "config" in config["rl"]:
            rl_config_name = config["rl"]["config"]
            rl_settings = self.load_config("base/rl_settings.yaml")

            if rl_config_name in rl_settings:
                rl_base_config = rl_settings[rl_config_name]
                config["rl"] = self._deep_merge(rl_base_config, config["rl"])
                del config["rl"]["config"]  # 参照キーは削除

        return config

    def load_complete_config(self, config_path: str) -> Dict[str, Any]:
        """
        完全な設定を読み込み（すべての参照を解決）

        Args:
            config_path: 設定ファイルのパス

        Returns:
            完全に解決された設定辞書
        """
        config = self.load_config(config_path)
        config = self.resolve_data_paths(config)
        config = self.resolve_component_configs(config)

        return config

    def get_available_configs(self) -> Dict[str, List[str]]:
        """
        利用可能な設定ファイルの一覧を取得

        Returns:
            カテゴリ別の設定ファイル一覧
        """
        configs = {"environments": [], "training": [], "evaluation": [], "base": []}

        for category in configs.keys():
            category_dir = self.config_root / category
            if category_dir.exists():
                for config_file in category_dir.glob("*.yaml"):
                    configs[category].append(config_file.stem)

        return configs


class SimpleConfig:
    """辞書をオブジェクトのように扱うためのクラス（後方互換性のため）"""

    def __init__(self, config_dict):
        self._dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return self._dict.get(key, default)


def load_config(config_path: str, config_root: str = None) -> SimpleConfig:
    """
    設定を読み込んでSimpleConfigオブジェクトを返す（便利関数）

    Args:
        config_path: 設定ファイルのパス
        config_root: 設定ファイルのルートディレクトリ

    Returns:
        SimpleConfigオブジェクト
    """
    loader = ConfigLoader(config_root)
    config_dict = loader.load_complete_config(config_path)
    return SimpleConfig(config_dict)


if __name__ == "__main__":
    # テスト用コード
    loader = ConfigLoader()

    # 利用可能な設定を表示
    available = loader.get_available_configs()
    print("利用可能な設定:")
    for category, configs in available.items():
        print(f"  {category}: {configs}")

    # 設定読み込みテスト
    try:
        config = loader.load_complete_config("environments/development.yaml")
        print(f"\n開発環境設定読み込み成功:")
        print(f"  開発者数: {config.get('num_developers')}")
        print(f"  デバッグ: {config.get('debug', {}).get('enabled')}")
        print(f"  データパス: {config.get('env', {}).get('backlog_path')}")
    except Exception as e:
        print(f"設定読み込みエラー: {e}")
