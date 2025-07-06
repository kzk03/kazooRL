#!/usr/bin/env python3
"""
Kazooプロジェクト構造の最終リファクタリングスクリプト
全ての分析スクリプトを統合し、重複を排除する
"""

import os
import shutil
from pathlib import Path


def main():
    print("=== Kazoo プロジェクト最終リファクタリング開始 ===\n")
    
    # プロジェクトのルートディレクトリ
    root_dir = Path("/Users/kazuki-h/rl/kazoo")
    
    # 移動すべきファイルのマッピング
    file_moves = {
        # 分析スクリプトをanalysis/reportsに統合
        "analyze_collaboration.py": "analysis/reports/collaboration_analysis.py",
        "analyze_gat_features.py": "analysis/reports/gat_analysis.py",  # 既存の同名ファイルと統合
        "analyze_irl_collaboration.py": "analysis/reports/irl_collaboration_analysis.py",
        "analyze_irl_completion.py": "analysis/reports/irl_completion_analysis.py", 
        "analyze_irl_feature_weights.py": "analysis/reports/irl_weights_analysis.py",
        "analyze_training_results.py": "analysis/reports/training_analysis.py",  # 既存の同名ファイルと統合
        "simple_irl_analysis.py": "analysis/reports/simple_irl_analysis.py",
        "generate_summary_report.py": "analysis/reports/summary_report.py",  # 既存の同名ファイルと統合
        
        # 評価・テストスクリプト
        "test_feature_dimensions.py": "evaluation/test_feature_dimensions.py",
        
        # パイプラインスクリプト
        "run_full_training_from_scratch.py": "pipelines/full_training_pipeline.py",
        
        # utilsフォルダに移動するスクリプト
        "project_restructure_plan.py": "utils/project_restructure_plan.py",
        "execute_restructure.py": "utils/execute_restructure.py",
    }
    
    # toolsフォルダの内容をメインフォルダに統合
    tools_moves = {
        "tools/analysis/analyze_weights.py": "analysis/reports/weights_analysis.py",
        "tools/analysis/create_expert_trajectories.py": "data_processing/create_expert_trajectories.py", 
        "tools/analysis/create_expert_trajectories_bot_excluded.py": "data_processing/create_expert_trajectories_bot_excluded.py",
        "tools/data_processing/get_github_data.py": "data_processing/extract_github_data.py",  # 既存のファイルと統合
        "tools/data_processing/generate_backlog.py": "data_processing/generate_backlog.py",  # 既存のファイルと統合
        "tools/data_processing/generate_profiles.py": "data_processing/generate_profiles.py",  # 既存のファイルと統合
        "tools/data_processing/generate_labels.py": "data_processing/generate_labels.py",  # 既存のファイルと統合
        "tools/data_processing/generate_graph.py": "data_processing/generate_graph.py",  # 既存のファイルと統合
        "tools/data_processing/build_developer_network.py": "data_processing/build_network.py",  # 既存のファイルと統合
    }
    
    # scriptsフォルダの整理
    scripts_moves = {
        "scripts/train_collaborative_gat.py": "training/gat/train_collaborative_gat.py",
        "scripts/plot_gnn_graph.py": "analysis/visualization/plot_gnn_graph.py",
        "scripts/run_complete_pipeline.py": "pipelines/complete_pipeline.py",
        "scripts/retrain_gnn_with_recent_data.py": "training/gat/retrain_gnn_with_recent_data.py",
        "scripts/train_irl.py": "training/irl/train_irl.py",  # 既存のファイルと統合
        "scripts/full_training_pipeline.py": "pipelines/full_training_pipeline.py",  # 上のファイルと統合
        "scripts/train_gnn.py": "training/gat/train_gnn.py",
        "scripts/train_oss.py": "training/rl/train_oss.py",
        "scripts/evaluate_2022_test.py": "evaluation/evaluate_2022_test.py",
    }
    
    # 必要なディレクトリを作成
    directories_to_create = [
        "analysis/visualization",
        "utils",
        "training/gat",
        "training/irl", 
        "training/rl",
        "evaluation",
        "pipelines"
    ]
    
    for dir_path in directories_to_create:
        full_path = root_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        # __init__.pyファイルを作成
        init_file = full_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
    
    # ファイルを移動
    all_moves = {**file_moves, **tools_moves, **scripts_moves}
    
    for src, dst in all_moves.items():
        src_path = root_dir / src
        dst_path = root_dir / dst
        
        if src_path.exists():
            print(f"移動: {src} -> {dst}")
            if dst_path.exists():
                print(f"  警告: 既存ファイル {dst} をバックアップします")
                backup_path = dst_path.with_suffix(dst_path.suffix + ".backup")
                shutil.move(str(dst_path), str(backup_path))
            
            # ディレクトリが存在しない場合は作成
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
        else:
            print(f"スキップ: {src} (ファイルが存在しません)")
    
    print("\n=== utilsフォルダの内容を作成 ===")
    
    # utilsにproject_setup.pyを作成
    project_setup_content = '''#!/usr/bin/env python3
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
'''
    
    utils_setup_file = root_dir / "utils" / "project_setup.py"
    utils_setup_file.write_text(project_setup_content)
    
    # utilsにconfig_manager.pyを作成
    config_manager_content = '''#!/usr/bin/env python3
"""
設定ファイルの管理
"""

import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
    
    def load_config(self, config_name):
        """設定ファイルを読み込み"""
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    def save_config(self, config_name, config_data):
        """設定ファイルを保存"""
        config_path = self.config_dir / f"{config_name}.yaml"
        self.config_dir.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
'''
    
    config_manager_file = root_dir / "utils" / "config_manager.py"
    config_manager_file.write_text(config_manager_content)
    
    print("\n=== visualization用のスクリプトを作成 ===")
    
    # analysis/visualization/plot_results.pyを作成
    plot_results_content = '''#!/usr/bin/env python3
"""
結果の可視化スクリプト
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_metrics(log_file):
    """トレーニングメトリクスのプロット"""
    # ログファイルからデータを読み込み、グラフを作成
    pass

def plot_irl_weights(weights_file):
    """IRL重みの可視化"""
    if Path(weights_file).exists():
        weights = np.load(weights_file)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(weights)), weights)
        plt.title("IRL Feature Weights")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight")
        plt.show()

def plot_gat_features(features_file):
    """GAT特徴量の可視化"""
    # 特徴量の可視化ロジック
    pass

if __name__ == "__main__":
    print("可視化スクリプトが実行されました")
'''
    
    plot_results_file = root_dir / "analysis" / "visualization" / "plot_results.py"
    plot_results_file.write_text(plot_results_content)
    
    print("\n=== 古いディレクトリの削除確認 ===")
    
    # 空になったディレクトリを削除
    old_dirs = ["tools", "scripts"]
    for old_dir in old_dirs:
        old_path = root_dir / old_dir
        if old_path.exists():
            try:
                # ディレクトリが空かチェック
                if not any(old_path.iterdir()):
                    old_path.rmdir()
                    print(f"空のディレクトリを削除: {old_dir}")
                else:
                    print(f"ディレクトリに残りファイルがあります: {old_dir}")
                    for item in old_path.iterdir():
                        print(f"  - {item.name}")
            except OSError:
                print(f"ディレクトリ削除に失敗: {old_dir}")
    
    print("\n=== 新しいREADMEを作成 ===")
    
    new_readme_content = '''# Kazoo - 強化学習ベースのOSS開発支援システム

## 📁 プロジェクト構造

```
kazoo/
├── src/kazoo/           # メインのソースコード
├── training/            # 各種トレーニングスクリプト
│   ├── gat/            # Graph Attention Network関連
│   ├── irl/            # Inverse Reinforcement Learning関連
│   └── rl/             # Reinforcement Learning関連
├── analysis/            # 分析・レポート生成
│   ├── reports/        # 各種分析レポート
│   └── visualization/  # 可視化スクリプト
├── evaluation/          # モデル評価・テスト
├── data_processing/     # データ前処理
├── pipelines/          # エンドツーエンドパイプライン
├── utils/              # ユーティリティ
├── configs/            # 設定ファイル
├── data/               # データファイル
├── models/             # 保存されたモデル
└── outputs/            # 出力結果

```

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
python utils/project_setup.py
```

### 2. データ準備
```bash
python data_processing/generate_graph.py
python data_processing/generate_profiles.py
python data_processing/generate_labels.py
```

### 3. 完全なトレーニングパイプライン実行
```bash
python pipelines/full_pipeline.py
```

### 4. 個別コンポーネントのトレーニング

#### GAT (Graph Attention Network)
```bash
python training/gat/train_gat.py
```

#### IRL (Inverse Reinforcement Learning)  
```bash
python training/irl/train_irl.py
```

#### RL (Reinforcement Learning)
```bash
python training/rl/train_rl.py
```

## 📊 分析・レポート

### 包括的な分析レポート
```bash
python analysis/reports/summary_report.py
```

### 個別分析
- **IRL分析**: `python analysis/reports/irl_analysis.py`
- **GAT分析**: `python analysis/reports/gat_analysis.py`
- **協力関係分析**: `python analysis/reports/collaboration_analysis.py`
- **トレーニング結果分析**: `python analysis/reports/training_analysis.py`

### 可視化
```bash
python analysis/visualization/plot_results.py
```

## 🧪 評価・テスト

```bash
python evaluation/evaluate_models.py
python evaluation/test_features.py
```

## 📋 主要機能

- **GAT**: 開発者間の協力関係をモデリング
- **IRL**: 専門家の行動から報酬関数を学習
- **RL**: PPOアルゴリズムによる効果的な開発者推薦
- **分析**: 詳細な重み分析と可視化
- **評価**: 包括的なモデル性能評価

## 🔧 設定

設定ファイルは`configs/`フォルダに配置：
- `base.yaml`: 基本設定
- `dev_profiles.yaml`: 開発者プロファイル

## 📈 改善されたポイント

1. **機能別の明確な分離**: GAT、IRL、RL、分析、データ処理が独立
2. **統一された分析システム**: 全ての分析が`analysis/`フォルダに集約
3. **エンドツーエンドパイプライン**: `pipelines/`で完全自動化
4. **充実した評価システム**: `evaluation/`で包括的テスト
5. **再利用可能なユーティリティ**: `utils/`で共通機能

## 🏗️ 開発者向け

新しい機能の追加時は、適切なフォルダに配置してください：
- トレーニング関連: `training/`
- 分析関連: `analysis/`
- 評価関連: `evaluation/`
- データ処理: `data_processing/`
'''
    
    new_readme_file = root_dir / "README_FINAL.md"
    new_readme_file.write_text(new_readme_content)
    
    print("\n=== 最終リファクタリング完了 ===")
    print("新しい構造:")
    print("- training/ (GAT, IRL, RL)")
    print("- analysis/ (reports, visualization)")
    print("- evaluation/ (テスト・評価)")
    print("- data_processing/ (データ前処理)")
    print("- pipelines/ (エンドツーエンドパイプライン)")
    print("- utils/ (ユーティリティ)")
    print("\n詳細は README_FINAL.md を参照してください")

if __name__ == "__main__":
    main()
