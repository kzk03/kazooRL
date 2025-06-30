#!/usr/bin/env python3
"""
開発者間の協力ネットワークを生成し、GATで活用できる形式で保存するスクリプト
"""

import json
import yaml
import torch
import numpy as np
from collections import defaultdict
import networkx as nx
from pathlib import Path

def build_developer_collaboration_network(dev_profiles_path, output_path):
    """
    開発者プロファイルから協力ネットワークを構築し、
    GATで使用できる形式で保存する
    """
    print("Building developer collaboration network...")
    
    # 開発者プロファイルを読み込み
    with open(dev_profiles_path, 'r', encoding='utf-8') as f:
        profiles = yaml.safe_load(f)
    
    # 開発者リストとIDマッピングを作成
    developers = list(profiles.keys())
    dev_to_id = {dev: i for i, dev in enumerate(developers)}
    id_to_dev = {i: dev for dev, i in dev_to_id.items()}
    
    print(f"Found {len(developers)} developers")
    
    # 協力関係のエッジを収集
    collaboration_edges = []
    collaboration_weights = defaultdict(int)
    
    for dev_name, profile in profiles.items():
        dev_id = dev_to_id[dev_name]
        collaborators = profile.get('collaborators', [])
        
        for collaborator in collaborators:
            if collaborator in dev_to_id:
                collaborator_id = dev_to_id[collaborator]
                edge = (min(dev_id, collaborator_id), max(dev_id, collaborator_id))
                collaboration_weights[edge] += 1
    
    # エッジリストと重みリストを作成
    edges = list(collaboration_weights.keys())
    weights = [collaboration_weights[edge] for edge in edges]
    
    print(f"Created {len(edges)} collaboration edges")
    
    # NetworkXグラフを作成（可視化・分析用）
    G = nx.Graph()
    G.add_nodes_from(range(len(developers)))
    for (src, dst), weight in collaboration_weights.items():
        G.add_edge(src, dst, weight=weight)
    
    # 基本統計を計算
    avg_degree = sum(dict(G.degree()).values()) / len(developers)
    density = nx.density(G)
    
    print(f"Network statistics:")
    print(f"  - Average degree: {avg_degree:.2f}")
    print(f"  - Density: {density:.4f}")
    print(f"  - Connected components: {nx.number_connected_components(G)}")
    
    # PyTorch Geometric形式でエッジインデックスを作成
    if edges:
        edge_index = torch.tensor(edges).t().contiguous()
        edge_weights = torch.tensor(weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weights = torch.empty(0, dtype=torch.float)
    
    # 開発者の特徴量も抽出
    dev_features = []
    for dev_name in developers:
        profile = profiles[dev_name]
        features = [
            profile.get('total_merged_prs', 0),
            profile.get('total_lines_changed', 0),
            profile.get('collaboration_network_size', 0),
            profile.get('comment_interactions', 0),
            profile.get('cross_issue_activity', 0),
            len(profile.get('touched_files', [])),
            len(profile.get('label_affinity', {})),
            len(profile.get('collaborators', []))
        ]
        dev_features.append(features)
    
    dev_features = torch.tensor(dev_features, dtype=torch.float)
    
    # データを保存
    network_data = {
        'dev_collaboration_edge_index': edge_index,
        'dev_collaboration_edge_weights': edge_weights,
        'dev_features_enhanced': dev_features,
        'dev_to_id': dev_to_id,
        'id_to_dev': id_to_dev,
        'num_developers': len(developers),
        'network_stats': {
            'avg_degree': avg_degree,
            'density': density,
            'num_edges': len(edges),
            'num_components': nx.number_connected_components(G)
        }
    }
    
    torch.save(network_data, output_path)
    print(f"✅ Developer collaboration network saved to: {output_path}")
    
    return network_data

def visualize_network(network_data, output_image_path):
    """
    協力ネットワークを可視化（オプション）
    """
    try:
        import matplotlib.pyplot as plt
        
        # NetworkXグラフを再構築
        edge_index = network_data['dev_collaboration_edge_index']
        edge_weights = network_data['dev_collaboration_edge_weights']
        id_to_dev = network_data['id_to_dev']
        
        G = nx.Graph()
        G.add_nodes_from(range(network_data['num_developers']))
        
        if edge_index.size(1) > 0:
            edges = edge_index.t().numpy()
            weights = edge_weights.numpy()
            
            for (src, dst), weight in zip(edges, weights):
                G.add_edge(src, dst, weight=weight)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # ノードサイズを次数で調整
        node_sizes = [G.degree(node) * 50 + 50 for node in G.nodes()]
        
        # エッジの太さを重みで調整
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        nx.draw(G, pos, 
                node_size=node_sizes,
                width=edge_widths,
                node_color='lightblue',
                edge_color='gray',
                alpha=0.7)
        
        plt.title("Developer Collaboration Network")
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Network visualization saved to: {output_image_path}")
        
    except ImportError:
        print("matplotlib not available, skipping visualization")

if __name__ == "__main__":
    DEV_PROFILES_PATH = "configs/dev_profiles.yaml"
    OUTPUT_PATH = "data/developer_collaboration_network.pt"
    VIZ_PATH = "outputs/developer_network.png"
    
    # ネットワークを構築
    network_data = build_developer_collaboration_network(DEV_PROFILES_PATH, OUTPUT_PATH)
    
    # 可視化（オプション）
    visualize_network(network_data, VIZ_PATH)
