# scripts/train_gnn.py

from pathlib import Path

import torch

from kazoo.GAT.GAT_model import GATModel  # 修正したモデルをインポート


def main():
    # グラフデータをロード
    graph_path = Path("data/graph.pt")

    if not graph_path.exists():
        print(f"エラー: グラフデータが見つかりません: {graph_path}")
        return

    data = torch.load(graph_path, weights_only=False)
    print(data)  # グラフの概要を出力
    for key, value in data.x_dict.items():
        print(f"Node type '{key}': {value.shape}")  # 各ノードタイプの形状を出力
    print("✅ グラフ読み込み成功")

    # --- モデルの初期化 ---
    # generate_graph.pyで定義した実際の次元数に必ず合わせる
    in_channels_dict = {"dev": 8, "task": 9}

    if ("task", "written_by", "dev") not in data.edge_index_dict:
        writes_edge_index = data[("dev", "writes", "task")].edge_index
        data[("task", "written_by", "dev")].edge_index = writes_edge_index.flip([0])

    print("GNNモデルを初期化します...")
    model = GATModel(in_channels_dict=in_channels_dict, out_channels=32)
    print(model)

    # --- 最適化の準備 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- 学習ループ（動作確認のため10回だけ実行） ---
    print("\n--- 動作確認のためのフォワードパスを開始 ---")
    for epoch in range(10):
        optimizer.zero_grad()

        # モデルのフォワードパスを実行
        # この処理がエラーなく通ることを確認するのが目的
        embeddings = model(data.x_dict, data.edge_index_dict)

        # ここでは単純なダミーの損失を計算
        loss = sum(embedding.mean() for embedding in embeddings.values())

        # 実際のタスクでは、この後損失を計算して backword() -> step() を行う
        # loss.backward()
        # optimizer.step()

        print(f"Epoch {epoch:02d}: フォワードパス成功。")
        print(f"  - devノードの出力shape: {embeddings['dev'].shape}")
        print(f"  - taskノードの出力shape: {embeddings['task'].shape}")

    print("\n✅ エラーなくモデルのフォワードパスが完了しました。")


if __name__ == "__main__":
    main()
