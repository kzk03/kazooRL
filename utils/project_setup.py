#!/usr/bin/env python3
"""
プロジェクトのセットアップとディレクトリ初期化
"""

import os
from pathlib import Path

def setup_project_directories():
    """プロジェクトの必要なディレクトリを作成"""
    dirs = [
        "data",
        "logs", 
        "models",
        "outputs",
        "results",
        "configs"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"ディレクトリ作成: {dir_name}")

def verify_dependencies():
    """依存関係の確認"""
    try:
        import torch
        import numpy as np
        import yaml
        import json
        print("✓ 必要な依存関係が全て利用可能です")
        return True
    except ImportError as e:
        print(f"✗ 依存関係エラー: {e}")
        return False

if __name__ == "__main__":
    setup_project_directories()
    verify_dependencies()
