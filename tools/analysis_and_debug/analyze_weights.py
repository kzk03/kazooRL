import os

# 自分のプロジェクトのFeatureExtractorを正しくインポートする
# このスクリプトはプロジェクトルートから実行することを想定
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(os.getcwd())
from src.kazoo.features.feature_extractor import FeatureExtractor


def analyze_reward_weights(config_path, weights_path):
    """
    学習済みの重みを読み込み、どの特徴が重要かを可視化する。
    """
    print("--- Analyzing Learned Reward Weights ---")

    # 1. 設定ファイルを読み込む
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    cfg = OmegaConf.load(config_path)

    # 2. 特徴量抽出器をインスタンス化して、特徴量の名前リストを取得
    try:
        feature_extractor = FeatureExtractor(cfg)
        feature_names = feature_extractor.feature_names
    except Exception as e:
        print(f"Error initializing FeatureExtractor: {e}")
        print(
            "Please check if your config file has the correct structure for FeatureExtractor."
        )
        return

    # 3. 学習済みの重みを読み込む
    if not os.path.exists(weights_path):
        print(f"Error: Weight file not found at {weights_path}")
        print("Please make sure 'train_irl.py' has been run successfully.")
        return
    weights = np.load(weights_path)

    if len(weights) != len(feature_names):
        print(
            f"Error: Mismatch between weights ({len(weights)}) and feature names ({len(feature_names)})"
        )
        print("Please ensure you are using the same config for training and analysis.")
        return

    # 4. Pandas DataFrameを使って結果を分かりやすく表示
    df = pd.DataFrame({"Feature": feature_names, "Weight": weights})

    # 重みの絶対値でソートして、影響の大きい順に表示
    df["Abs_Weight"] = df["Weight"].abs()
    df_sorted = df.sort_values(by="Abs_Weight", ascending=False).drop(
        columns=["Abs_Weight"]
    )

    print("\n--- Learned Reward Weights (Sorted by Impact) ---")
    # to_string()を使うことで、全ての行が表示される
    print(df_sorted.to_string())
    print("\n---------------------------------------------")
    print("\n[Interpretation]")
    print(
        " - Positive weights (大きい正の数): Encourage actions. Developers PREFER tasks with these features."
    )
    print(
        " - Negative weights (大きい負の数): Discourage actions. Developers AVOID tasks with these features."
    )
    print(" - Weights near zero (0に近い数): Have little to no effect on the decision.")


if __name__ == "__main__":
    # 設定ファイルのパスと、train_irl.pyが出力する重みファイルのパスを指定
    # このスクリプトはプロジェクトのルートディレクトリから実行することを想定
    CONFIG_PATH = "configs/base.yaml"
    WEIGHTS_PATH = "data/learned_weights.npy"

    analyze_reward_weights(CONFIG_PATH, WEIGHTS_PATH)
