#!/usr/bin/env python3
"""
統合強化学習システム実行スクリプト
簡単にシステムを実行するためのラッパー
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="統合強化学習システム実行")
    parser.add_argument(
        "--config", 
        default="unified_rl", 
        help="設定ファイル名 (configs/以下の.yamlファイル)"
    )
    parser.add_argument(
        "--method", 
        choices=["original", "stable_baselines", "unified"],
        default="unified",
        help="訓練方法の選択"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=None,
        help="訓練ステップ数"
    )
    parser.add_argument(
        "--eval-only", 
        action="store_true",
        help="評価のみ実行（訓練はスキップ）"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="クイックモード（少ないステップ数で高速実行）"
    )
    
    args = parser.parse_args()
    
    # 実行コマンド構築
    script_path = Path(__file__).parent / "train_unified_rl.py"
    
    cmd = [
        sys.executable, 
        str(script_path),
        f"--config-name={args.config}"
    ]
    
    # パラメータオーバーライド
    overrides = []
    
    if args.method:
        overrides.append(f"training_method={args.method}")
    
    if args.timesteps:
        overrides.append(f"rl.total_timesteps={args.timesteps}")
    
    if args.quick:
        overrides.extend([
            "rl.total_timesteps=5000",
            "rl.eval_freq=500",
            "env.max_steps=20",
            "optimization.max_developers=10",
            "optimization.max_tasks=50"
        ])
    
    if args.eval_only:
        overrides.append("training_method=evaluation_only")
    
    # オーバーライドをコマンドに追加
    for override in overrides:
        cmd.append(override)
    
    print("🚀 統合強化学習システム実行")
    print(f"   設定: {args.config}")
    print(f"   方法: {args.method}")
    print(f"   コマンド: {' '.join(cmd)}")
    print("=" * 60)
    
    # 実行
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ 実行完了")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ 実行エラー: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("⏹️ ユーザーによる中断")
        return 1

if __name__ == "__main__":
    exit(main())
