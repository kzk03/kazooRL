import numpy as np
import pickle
from src.kazoo.features.feature_extractor import get_features, FEATURE_DIM, FEATURE_NAMES

def calculate_feature_expectations(trajectories):
    """軌跡データから平均的な特徴量ベクトル（特徴量期待値）を計算する"""
    total_features = np.zeros(FEATURE_DIM)
    num_steps = 0
    for trajectory in trajectories:
        for step in trajectory:
            features = get_features(step['state'], step['action'], step['action_details'], step['actor'])
            total_features += features
            num_steps += 1
    return total_features / num_steps if num_steps > 0 else total_features

def run_rl_and_get_trajectories(reward_weights):
    """【最重要】現在の報酬重みの下でRLを実行し、エージェントが生成した軌跡を返す"""
    print(f"  - Running RL sub-problem with weights: {np.round(reward_weights, 2)}")
    # ここに、`kazooRL`のPPO学習機能を呼び出すロジックを実装します。
    # 1. `OSSSimpleEnv`を、現在の`reward_weights`で報酬を計算するように初期化
    # 2. `indep_ppo`でエージェントを一定期間学習
    # 3. 学習したポリシーでシミュレーションを実行し、その軌跡を収集して返す
    # この関数は、本研究の技術的な核心部分です。
    
    # 現状は、ダミーの軌跡を返す仮実装です。
    from src.kazoo.consts.actions import Action
    dummy_trajectory = [{"state": {}, "action": np.random.choice(list(Action)), "action_details": {}, "actor": "dummy_agent"}]
    return [dummy_trajectory]

def main():
    print("Loading expert trajectories...")
    with open('data/expert_trajectories.pkl', 'rb') as f:
        expert_trajectories = pickle.load(f)
    
    expert_feature_expectations = calculate_feature_expectations(expert_trajectories)
    print(f"Expert Feature Expectations: {np.round(expert_feature_expectations, 2)}")

    reward_weights = np.zeros(FEATURE_DIM)
    learning_rate = 0.1
    
    print("\nStarting IRL training loop...")
    for i in range(20): # IRLのイテレーション回数
        print(f"\n--- IRL Iteration {i+1}/20 ---")
        agent_trajectories = run_rl_and_get_trajectories(reward_weights)
        agent_feature_expectations = calculate_feature_expectations(agent_trajectories)
        print(f"  - Agent Feature Expectations: {np.round(agent_feature_expectations, 2)}")
        
        gradient = expert_feature_expectations - agent_feature_expectations
        reward_weights += learning_rate * gradient
        print(f"  - Updated Reward Weights: {np.round(reward_weights, 2)}")

    output_path = "data/learned_reward_weights.npy"
    np.save(output_path, reward_weights)
    print(f"\nIRL training complete. Learned weights saved to {output_path}")

    print("\n--- Motivation Analysis (Learned Weights) ---")
    for name, weight in zip(FEATURE_NAMES, reward_weights):
        print(f"{name:>20}: {weight:.4f}")

if __name__ == '__main__':
    main()