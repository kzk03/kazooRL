#!/usr/bin/env python3
"""
Kazoo プロジェクト構造実行スクリプト
新しい階層構造にファイルを整理する
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


def create_new_directory_structure():
    """新しいディレクトリ構造を作成"""
    print("📁 新しいディレクトリ構造を作成中...")

    directories = [
        "training/gat",
        "training/irl",
        "training/rl",
        "pipelines",
        "analysis/reports",
        "analysis/visualization",
        "evaluation",
        "data_processing",
        "utils",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")

    # __init__.pyファイルを作成
    init_dirs = ["training", "analysis"]
    for dir_name in init_dirs:
        init_file = Path(dir_name) / "__init__.py"
        init_file.write_text("# -*- coding: utf-8 -*-\n")


def move_and_rename_files():
    """ファイルの移動とリネームを実行"""
    print("\n📦 ファイル移動・リネーム中...")

    # 移動・リネーム計画
    moves = [
        # GAT関連
        ("scripts/train_collaborative_gat.py", "training/gat/train_gat.py"),
        ("scripts/train_gnn.py", "training/gat/train_gat_standalone.py"),
        # IRL関連
        ("scripts/train_irl.py", "training/irl/train_irl.py"),
        # RL関連
        ("scripts/train_oss.py", "training/rl/train_rl.py"),
        # パイプライン
        ("run_full_training_from_scratch.py", "pipelines/full_pipeline.py"),
        # 分析
        ("analyze_training_results.py", "analysis/reports/training_analysis.py"),
        ("generate_summary_report.py", "analysis/reports/summary_report.py"),
        ("analyze_gat_features.py", "analysis/reports/gat_analysis.py"),
        # 評価
        ("scripts/evaluate_2022_test.py", "evaluation/evaluate_models.py"),
        ("test_feature_dimensions.py", "evaluation/test_features.py"),
        # データ処理
        (
            "tools/data_processing/generate_graph.py",
            "data_processing/generate_graph.py",
        ),
        (
            "tools/data_processing/generate_profiles.py",
            "data_processing/generate_profiles.py",
        ),
        (
            "tools/data_processing/generate_backlog.py",
            "data_processing/generate_backlog.py",
        ),
        (
            "tools/data_processing/build_developer_network.py",
            "data_processing/build_network.py",
        ),
        (
            "tools/data_processing/get_github_data.py",
            "data_processing/extract_github_data.py",
        ),
        (
            "tools/data_processing/generate_labels.py",
            "data_processing/generate_labels.py",
        ),
    ]

    for old_path, new_path in moves:
        old_file = Path(old_path)
        new_file = Path(new_path)

        if old_file.exists():
            shutil.copy2(old_file, new_file)
            print(f"✅ {old_path} -> {new_path}")
        else:
            print(f"❌ {old_path} - ファイルが見つかりません")


def create_unified_irl_analysis():
    """統合IRL分析スクリプトを作成"""
    print("\n🔄 統合IRL分析スクリプト作成中...")

    unified_script = '''#!/usr/bin/env python3
"""
統合IRL分析スクリプト
複数のIRL分析機能を統合し、分かりやすいレポートを生成
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('src')

class IRLAnalyzer:
    """IRL結果の統合分析クラス"""
    
    def __init__(self, weights_path="data/learned_weights_training.npy"):
        self.weights_path = Path(weights_path)
        self.weights = None
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self):
        """特徴量名の定義"""
        base_features = [
            "ログイン名の長さ", "名前の有無", "名前の長さ", "会社情報の有無", "会社名の長さ",
            "場所情報の有無", "場所情報の長さ", "プロフィール文の有無", "プロフィール文の長さ",
            "公開リポジトリ数", "公開リポジトリ数(対数)", "フォロワー数", "フォロワー数(対数)",
            "フォロー数", "フォロー数(対数)", "アカウント年数(日)", "アカウント年数(年)",
            "フォロワー/フォロー比", "年間リポジトリ作成数", "人気度スコア", "活動度スコア",
            "影響力スコア", "経験値スコア", "社交性スコア", "プロフィール完成度"
        ]
        
        gat_features = [f"GAT特徴量{i}" for i in range(37)]
        return base_features + gat_features
    
    def load_weights(self):
        """重みを読み込み"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"重みファイルが見つかりません: {self.weights_path}")
        
        self.weights = np.load(self.weights_path)
        print(f"✅ IRL重み読み込み成功: {self.weights.shape}")
        return self.weights
    
    def analyze_weights(self):
        """重みの基本分析"""
        if self.weights is None:
            self.load_weights()
        
        analysis = {
            "総特徴量数": len(self.weights),
            "基本特徴量数": 25,
            "GAT特徴量数": len(self.weights) - 25,
            "重要特徴量数": np.sum(np.abs(self.weights) > 0.5),
            "正の重み数": np.sum(self.weights > 0),
            "負の重み数": np.sum(self.weights < 0),
            "平均重み": self.weights.mean(),
            "標準偏差": self.weights.std(),
            "最大重み": self.weights.max(),
            "最小重み": self.weights.min()
        }
        
        return analysis
    
    def get_important_features(self, top_n=10):
        """重要な特徴量を取得"""
        if self.weights is None:
            self.load_weights()
        
        # 絶対値で重要度をソート
        importance_indices = np.argsort(np.abs(self.weights))[::-1]
        
        important_features = []
        for i in range(min(top_n, len(importance_indices))):
            idx = importance_indices[i]
            weight = self.weights[idx]
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"特徴量{idx}"
            important_features.append((name, weight, idx))
        
        return important_features
    
    def generate_simple_report(self):
        """分かりやすいレポートを生成"""
        if self.weights is None:
            self.load_weights()
        
        print("🎯 IRL学習結果 - 分かりやすい解釈")
        print("=" * 60)
        
        # 基本統計
        analysis = self.analyze_weights()
        
        print("📊 学習結果サマリー:")
        print(f"  分析した特徴量数: {analysis['総特徴量数']}")
        print(f"  重要な特徴量数: {analysis['重要特徴量数']}")
        print(f"  正の影響: {analysis['正の重み数']}個")
        print(f"  負の影響: {analysis['負の重み数']}個")
        
        # 協力関係 vs 基本情報
        base_weights = self.weights[:25]
        gat_weights = self.weights[25:] if len(self.weights) > 25 else np.array([])
        
        base_importance = np.mean(np.abs(base_weights))
        gat_importance = np.mean(np.abs(gat_weights)) if len(gat_weights) > 0 else 0
        
        print(f"\\n🤝 協力関係 vs 基本情報:")
        print(f"  基本情報の重要度: {base_importance:.3f}")
        print(f"  協力関係の重要度: {gat_importance:.3f}")
        
        if gat_importance > base_importance:
            ratio = gat_importance / base_importance
            print(f"  → 協力関係が {ratio:.1f}倍重要！")
        
        # 重要な特徴量
        important_features = self.get_important_features(10)
        
        print(f"\\n✅ 最重要特徴量 Top 10:")
        for rank, (name, weight, idx) in enumerate(important_features, 1):
            status = "優先" if weight > 0 else "回避"
            print(f"  {rank:2d}. {name[:20]:20s} ({status}: {weight:6.3f})")
        
        return analysis, important_features
    
    def create_visualization(self):
        """可視化グラフを作成"""
        if self.weights is None:
            self.load_weights()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 重要度ランキング
        important_features = self.get_important_features(15)
        names = [f[0][:15] + "..." if len(f[0]) > 15 else f[0] for f in important_features]
        weights = [f[1] for f in important_features]
        colors = ['blue' if w > 0 else 'red' for w in weights]
        
        ax1.barh(range(len(weights)), weights, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(weights)))
        ax1.set_yticklabels(names, fontsize=10)
        ax1.set_xlabel('Weight Value')
        ax1.set_title('Top 15 Most Important Features')
        ax1.grid(True, alpha=0.3)
        
        # 2. 基本 vs GAT比較
        base_weights = self.weights[:25]
        gat_weights = self.weights[25:] if len(self.weights) > 25 else np.array([])
        
        categories = ['Basic Features']
        importances = [np.mean(np.abs(base_weights))]
        
        if len(gat_weights) > 0:
            categories.append('GAT Features')
            importances.append(np.mean(np.abs(gat_weights)))
        
        ax2.bar(categories, importances, color=['skyblue', 'lightcoral'][:len(categories)], alpha=0.8)
        ax2.set_ylabel('Average Importance')
        ax2.set_title('Feature Category Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. 重み分布
        ax3.hist(self.weights, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', label='Zero')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Weight Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 累積重要度
        sorted_abs_weights = np.sort(np.abs(self.weights))[::-1]
        cumsum_weights = np.cumsum(sorted_abs_weights)
        cumsum_normalized = cumsum_weights / cumsum_weights[-1] * 100
        
        ax4.plot(range(1, len(self.weights)+1), cumsum_normalized, 'b-', linewidth=2)
        ax4.axhline(80, color='red', linestyle='--', alpha=0.7, label='80%')
        ax4.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95%')
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('Cumulative Importance (%)')
        ax4.set_title('Cumulative Feature Importance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = Path("outputs") / f"irl_unified_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 分析グラフ保存: {output_path}")
        plt.close()
        
        return output_path

def main():
    """メイン実行"""
    print("🔍 統合IRL分析")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)
    
    try:
        analyzer = IRLAnalyzer()
        analysis, important_features = analyzer.generate_simple_report()
        output_path = analyzer.create_visualization()
        
        print(f"\\n🎉 分析完了!")
        print(f"📊 可視化: {output_path}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''

    unified_file = Path("analysis/reports/irl_analysis.py")
    unified_file.write_text(unified_script)
    print(f"✅ 統合IRL分析スクリプト作成: {unified_file}")


def create_unified_pipeline():
    """統合パイプラインを作成"""
    print("\n🔄 統合パイプライン作成中...")

    pipeline_script = '''#!/usr/bin/env python3
"""
Kazoo 統合学習パイプライン
GAT → IRL → RL の完全な学習フローを実行
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

class KazooPipeline:
    """Kazoo学習パイプラインクラス"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.project_root = Path(__file__).parent.parent
        
    def run_command(self, cmd, description, working_dir=None):
        """コマンドを実行"""
        print(f"\\n🚀 {description}")
        print(f"実行コマンド: {cmd}")
        print("=" * 60)
        
        if working_dir:
            original_dir = Path.cwd()
            os.chdir(working_dir)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ {description} 完了")
                return True
            else:
                print(f"❌ {description} 失敗 (exit code: {result.returncode})")
                return False
        finally:
            if working_dir:
                os.chdir(original_dir)
    
    def check_prerequisites(self):
        """前提条件をチェック"""
        print("📋 前提条件チェック中...")
        
        required_files = [
            "data/backlog_training.json",
            "data/expert_trajectories.pkl",
            "data/labels.pt",
            "configs/base_training.yaml"
        ]
        
        all_exist = True
        for file_path in required_files:
            if (self.project_root / file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - 見つかりません")
                all_exist = False
        
        return all_exist
    
    def run_gat_training(self):
        """GAT訓練を実行"""
        return self.run_command(
            "python training/gat/train_gat.py",
            "GAT (Graph Attention Network) 訓練",
            self.project_root
        )
    
    def run_irl_training(self):
        """IRL訓練を実行"""
        return self.run_command(
            "python training/irl/train_irl.py", 
            "IRL (Inverse Reinforcement Learning) 訓練",
            self.project_root
        )
    
    def run_rl_training(self):
        """RL訓練を実行"""
        return self.run_command(
            "python training/rl/train_rl.py",
            "RL (Reinforcement Learning) 訓練", 
            self.project_root
        )
    
    def run_analysis(self):
        """結果分析を実行"""
        return self.run_command(
            "python analysis/reports/irl_analysis.py",
            "結果分析・レポート生成",
            self.project_root
        )
    
    def run_evaluation(self):
        """評価を実行"""
        return self.run_command(
            "python evaluation/evaluate_models.py",
            "モデル評価",
            self.project_root
        )
    
    def run_full_pipeline(self):
        """完全パイプラインを実行"""
        print(f"🎯 Kazoo統合学習パイプライン開始")
        print(f"📅 開始時刻: {self.start_time}")
        print(f"📁 作業ディレクトリ: {self.project_root}")
        print("=" * 80)
        
        # 前提条件チェック
        if not self.check_prerequisites():
            print("❌ 前提条件が満たされていません")
            return False
        
        steps = [
            ("GAT訓練", self.run_gat_training),
            ("IRL訓練", self.run_irl_training), 
            ("RL訓練", self.run_rl_training),
            ("結果分析", self.run_analysis),
            ("モデル評価", self.run_evaluation)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            if not step_func():
                failed_steps.append(step_name)
                print(f"⚠️ {step_name}で失敗しましたが、継続します")
        
        # 完了レポート
        end_time = datetime.now()
        elapsed = end_time - self.start_time
        
        print("\\n" + "=" * 80)
        print(f"🎉 Kazoo統合学習パイプライン完了!")
        print(f"⏱️ 実行時間: {elapsed}")
        print(f"📅 完了時刻: {end_time}")
        
        if failed_steps:
            print(f"\\n⚠️ 失敗したステップ: {', '.join(failed_steps)}")
        else:
            print(f"\\n✅ 全ステップが正常に完了しました!")
        
        # 生成されたファイルの確認
        self.check_generated_files()
        
        return len(failed_steps) == 0
    
    def check_generated_files(self):
        """生成されたファイルを確認"""
        print(f"\\n📂 生成ファイル確認:")
        
        expected_files = [
            "data/gnn_model_collaborative.pt",
            "data/graph_collaborative.pt",
            "data/learned_weights_training.npy",
            "models/ppo_agent.pt"
        ]
        
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"✅ {file_path} ({size:,} bytes)")
            else:
                print(f"❌ {file_path} - 生成されませんでした")

def main():
    """メイン実行"""
    import os
    
    pipeline = KazooPipeline()
    success = pipeline.run_full_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
'''

    pipeline_file = Path("pipelines/full_pipeline.py")
    pipeline_file.write_text(pipeline_script)
    print(f"✅ 統合パイプライン作成: {pipeline_file}")


def update_import_paths():
    """インポートパスを更新"""
    print("\n🔧 インポートパス更新中...")

    # 新しいファイルでsrcへのパスを修正
    files_to_update = [
        "training/gat/train_gat.py",
        "training/irl/train_irl.py",
        "training/rl/train_rl.py",
        "analysis/reports/irl_analysis.py",
        "evaluation/evaluate_models.py",
    ]

    for file_path in files_to_update:
        file_obj = Path(file_path)
        if file_obj.exists():
            content = file_obj.read_text()

            # srcパスの修正
            if "sys.path.append('src')" in content:
                # ファイルの階層に応じてパスを調整
                depth = len(file_obj.parts) - 1
                new_path = "../" * depth + "src"
                content = content.replace(
                    "sys.path.append('src')", f"sys.path.append('{new_path}')"
                )
                file_obj.write_text(content)
                print(f"✅ {file_path} - インポートパス更新")


def create_readme():
    """新しい構造のREADMEを作成"""
    print("\n📝 README作成中...")

    readme_content = """# Kazoo プロジェクト - 新構造

## 📁 ディレクトリ構造

```
kazoo/
├── training/           # 学習関連スクリプト
│   ├── gat/           # GAT (Graph Attention Network) 訓練
│   ├── irl/           # IRL (Inverse Reinforcement Learning) 訓練  
│   └── rl/            # RL (Reinforcement Learning) 訓練
├── pipelines/         # 統合パイプライン
├── analysis/          # 分析・レポート
│   ├── reports/       # 分析レポート生成
│   └── visualization/ # 可視化
├── evaluation/        # 評価・テスト
├── data_processing/   # データ処理
├── utils/             # ユーティリティ
├── src/              # ライブラリコード
├── configs/          # 設定ファイル
├── data/             # データファイル
├── models/           # 学習済みモデル
└── outputs/          # 出力ファイル
```

## 🚀 使用方法

### 完全パイプライン実行
```bash
python pipelines/full_pipeline.py
```

### 個別ステップ実行

#### GAT訓練
```bash
python training/gat/train_gat.py
```

#### IRL訓練  
```bash
python training/irl/train_irl.py
```

#### RL訓練
```bash
python training/rl/train_rl.py
```

### 分析・レポート生成
```bash
python analysis/reports/irl_analysis.py
```

### 評価
```bash
python evaluation/evaluate_models.py
```

## 📊 主要な改善点

- **機能別整理**: GAT、IRL、RLの訓練スクリプトを分離
- **統合分析**: 複数の分析機能を統合し、分かりやすいレポートを生成
- **パイプライン化**: 完全な学習フローを自動実行
- **エラーハンドリング**: 各ステップでのエラー処理を強化

## 🔧 設定

主要な設定は `configs/base_training.yaml` で管理されています。

## 📈 出力

- 学習済みモデル: `models/`
- 分析結果: `outputs/`
- ログ: `logs/`
"""

    readme_file = Path("README_NEW_STRUCTURE.md")
    readme_file.write_text(readme_content)
    print(f"✅ README作成: {readme_file}")


def main():
    """メイン実行"""
    print("🔄 Kazoo プロジェクト構造整理実行")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)

    try:
        create_new_directory_structure()
        move_and_rename_files()
        create_unified_irl_analysis()
        create_unified_pipeline()
        update_import_paths()
        create_readme()

        print("\\n🎉 プロジェクト構造整理完了!")
        print("\\n📋 確認事項:")
        print("1. 新しい構造でのファイル配置確認")
        print("2. インポートパスの動作確認")
        print("3. パイプライン動作テスト")
        print("\\n💡 古いファイルは残してあります（バックアップとして）")

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
