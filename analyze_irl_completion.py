#!/usr/bin/env python3
"""
逆強化学習完了後の分析スクリプト
GAT、グラフ、IRL の結果を総合的に分析
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_gat_results():
    """GAT訓練結果の分析"""
    print("🧠 GAT (Graph Attention Network) 結果分析")
    print("=" * 50)
    
    # GATモデルの確認
    gat_model_path = Path("data/gnn_model_collaborative.pt")
    if gat_model_path.exists():
        try:
            gat_model = torch.load(gat_model_path, map_location='cpu')
            print(f"✅ GAT モデル読み込み成功: {gat_model_path}")
            
            # モデル構造の確認
            if hasattr(gat_model, 'state_dict'):
                state_dict = gat_model.state_dict()
                print(f"📊 モデルパラメータ数: {len(state_dict)} layers")
                
                total_params = 0
                for name, param in state_dict.items():
                    param_count = param.numel()
                    total_params += param_count
                    print(f"  - {name}: {param.shape} ({param_count:,} params)")
                
                print(f"🔢 総パラメータ数: {total_params:,}")
            else:
                print(f"📋 モデル型: {type(gat_model)}")
                
        except Exception as e:
            print(f"❌ GAT モデル読み込みエラー: {e}")
    else:
        print(f"❌ GAT モデルファイルが見つかりません: {gat_model_path}")
    
    # 協力グラフの確認
    graph_path = Path("data/graph_collaborative.pt")
    if graph_path.exists():
        try:
            graph_data = torch.load(graph_path, map_location='cpu')
            print(f"✅ 協力グラフ読み込み成功: {graph_path}")
            print(f"📊 グラフ構造: {type(graph_data)}")
            
            if hasattr(graph_data, 'x'):
                print(f"  - ノード特徴量: {graph_data.x.shape}")
            if hasattr(graph_data, 'edge_index'):
                print(f"  - エッジ数: {graph_data.edge_index.shape[1]}")
            if hasattr(graph_data, 'edge_attr'):
                print(f"  - エッジ属性: {graph_data.edge_attr.shape}")
                
        except Exception as e:
            print(f"❌ 協力グラフ読み込みエラー: {e}")
    else:
        print(f"❌ 協力グラフファイルが見つかりません: {graph_path}")

def analyze_irl_results():
    """IRL訓練結果の分析"""
    print("\n🎯 IRL (Inverse Reinforcement Learning) 結果分析")
    print("=" * 50)
    
    # IRL重みの確認
    weights_path = Path("data/learned_weights_training.npy")
    if weights_path.exists():
        try:
            weights = np.load(weights_path)
            print(f"✅ IRL重み読み込み成功: {weights_path}")
            print(f"📊 重み形状: {weights.shape}")
            print(f"📈 重み統計:")
            print(f"  - 平均: {weights.mean():.6f}")
            print(f"  - 標準偏差: {weights.std():.6f}")
            print(f"  - 最小値: {weights.min():.6f}")
            print(f"  - 最大値: {weights.max():.6f}")
            
            # 重みの分布を可視化
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.hist(weights, bins=50, alpha=0.7, edgecolor='black')
            plt.title('IRL重みの分布')
            plt.xlabel('重み値')
            plt.ylabel('頻度')
            
            plt.subplot(2, 2, 2)
            plt.plot(weights)
            plt.title('IRL重みの順序プロット')
            plt.xlabel('特徴量インデックス')
            plt.ylabel('重み値')
            
            plt.subplot(2, 2, 3)
            top_indices = np.argsort(np.abs(weights))[-10:]
            plt.barh(range(len(top_indices)), weights[top_indices])
            plt.title('絶対値上位10の重み')
            plt.xlabel('重み値')
            plt.ylabel('特徴量インデックス')
            plt.gca().set_yticklabels([f'特徴{i}' for i in top_indices])
            
            plt.subplot(2, 2, 4)
            plt.boxplot(weights)
            plt.title('IRL重みのボックスプロット')
            plt.ylabel('重み値')
            
            plt.tight_layout()
            output_path = Path("outputs") / f"irl_weights_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📊 重み分析グラフ保存: {output_path}")
            plt.close()
            
            return weights
            
        except Exception as e:
            print(f"❌ IRL重み読み込みエラー: {e}")
            return None
    else:
        print(f"❌ IRL重みファイルが見つかりません: {weights_path}")
        return None

def analyze_feature_dimensions():
    """特徴量次元の分析"""
    print("\n🔢 特徴量次元分析")
    print("=" * 50)
    
    try:
        # 特徴抽出器のテスト
        import sys
        sys.path.append('src')
        from kazoo.features.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # サンプルプロファイルで特徴量次元を確認
        sample_profile = {
            'login': 'test_user',
            'name': 'Test User',
            'company': 'Test Company',
            'location': 'Tokyo',
            'bio': 'Test bio',
            'public_repos': 10,
            'followers': 100,
            'following': 50,
            'created_at': '2020-01-01T00:00:00Z'
        }
        
        features = extractor.extract_features(sample_profile)
        print(f"✅ 特徴量抽出成功")
        print(f"📊 総特徴量次元: {len(features)}")
        
        # 基本特徴量とGAT特徴量の内訳
        base_features = extractor.extract_base_features(sample_profile)
        print(f"  - 基本特徴量: {len(base_features)} 次元")
        
        if hasattr(extractor, 'gnn_extractor') and extractor.gnn_extractor is not None:
            # GAT特徴量の次元を推定
            gat_dim = len(features) - len(base_features)
            print(f"  - GAT特徴量: {gat_dim} 次元")
        else:
            print(f"  - GAT特徴量: 0 次元（GAT未使用）")
            
    except Exception as e:
        print(f"❌ 特徴量次元分析エラー: {e}")

def check_training_files():
    """訓練用ファイルの確認"""
    print("\n📁 訓練用ファイル確認")
    print("=" * 50)
    
    training_files = [
        ("data/graph_training.pt", "訓練用グラフ"),
        ("data/gnn_model_collaborative.pt", "GAT協力モデル"),
        ("data/graph_collaborative.pt", "協力グラフ"),
        ("data/learned_weights_training.npy", "IRL学習重み"),
        ("data/backlog_training.json", "訓練用バックログ"),
        ("data/expert_trajectories.pkl", "エキスパート軌跡"),
        ("data/labels.pt", "ラベルデータ")
    ]
    
    all_exist = True
    for file_path, description in training_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {description}: {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {description}: {file_path} - 見つかりません")
            all_exist = False
    
    return all_exist

def main():
    """メイン分析実行"""
    print("🔍 Kazoo パイプライン中間分析 (IRL完了後)")
    print(f"📅 実行日時: {datetime.now()}")
    print("=" * 60)
    
    # ファイル確認
    files_ok = check_training_files()
    if not files_ok:
        print("\n⚠️ 一部のファイルが見つかりません。パイプラインが正常に完了していない可能性があります。")
    
    # GAT結果分析
    analyze_gat_results()
    
    # IRL結果分析
    weights = analyze_irl_results()
    
    # 特徴量次元分析
    analyze_feature_dimensions()
    
    print("\n" + "=" * 60)
    print("📋 分析サマリー:")
    print("✅ GAT (Graph Attention Network) 訓練完了")
    print("✅ 協力ネットワークグラフ生成完了")
    print("✅ IRL (Inverse Reinforcement Learning) 重み学習完了")
    
    if weights is not None:
        significant_features = np.sum(np.abs(weights) > 0.01)
        print(f"📊 有意な特徴量数: {significant_features}/{len(weights)}")
        print(f"🎯 次のステップ: 強化学習（RL）実行")
    
    print("\n🔄 強化学習実行前の推奨事項:")
    print("1. 特徴量次元が一貫していることを確認")
    print("2. IRL重みが適切に学習されていることを確認") 
    print("3. 強化学習の設定（エピソード数、学習率等）を確認")

if __name__ == "__main__":
    main()
