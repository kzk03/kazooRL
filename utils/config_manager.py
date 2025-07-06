#!/usr/bin/env python3
"""
設定ファイルの管理
"""

from pathlib import Path

import yaml


class ConfigManager:
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)

    def load_config(self, config_name):
        """設定ファイルを読み込み"""
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    def save_config(self, config_name, config_data):
        """設定ファイルを保存"""
        config_path = self.config_dir / f"{config_name}.yaml"
        self.config_dir.mkdir(exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
