import numpy as np
import torch

from kazoo.learners.ppo_agent import PPOAgent


class RolloutStorage:
    """経験データを一時的に保存するためのヘルパークラス"""
    def __init__(self, num_steps, obs_space, act_space, device):
        self.device = device
        self.num_steps = num_steps
        
        # データ保存用のTensorを適切なサイズで初期化
        self.obs = torch.zeros((num_steps,) + obs_space.shape).to(device)
        self.actions = torch.zeros((num_steps,) + act_space.shape).to(device)
        self.log_probs = torch.zeros(num_steps).to(device)
        self.rewards = torch.zeros(num_steps).to(device)
        self.dones = torch.zeros(num_steps).to(device)
        self.values = torch.zeros(num_steps).to(device)
        # 以前の修正で追加したentropiesも忘れずに含めます
        self.entropies = torch.zeros(num_steps).to(device)
        
        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value, entropy):
        """ステップごとのデータを追加する"""
        self.obs[self.step] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[self.step] = torch.as_tensor(action)
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = torch.as_tensor(reward)
        self.dones[self.step] = torch.as_tensor(done)
        self.values[self.step] = value
        self.entropies[self.step] = entropy
        self.step = (self.step + 1) % self.num_steps

    def get(self):
        """学習に使うためのデータを返す"""
        return self



class IndependentPPOController:
    """複数のPPOAgentを管理し、マルチエージェント学習を実行する司令塔"""
    def __init__(self, env, config):
        self.env = env
        self.agent_ids = env.agent_ids
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        # --- 各エージェントのエージェントとストレージを作成 ---
        self.agents = {}
        self.storages = {}
        for agent_id in self.agent_ids:
            obs_space = self.env.observation_space[agent_id]
            act_space = self.env.action_space[agent_id]
            
            # エージェントを作成
            self.agents[agent_id] = PPOAgent(
                obs_space=obs_space,
                act_space=act_space,
                lr=config.lr,
                gamma=config.gamma,
                # ...その他のPPOハイパーパラメータをconfigから渡す...
                epochs=config.epochs,
                device=self.device,
            )
            # データ保存用のストレージを作成
            self.storages[agent_id] = RolloutStorage(
                config.rollout_len, obs_space, act_space, self.device
            )

# src/kazoo/learners/independent_ppo_controller.py の learn メソッド

    def learn(self, total_timesteps):
        print("Starting Multi-Agent PPO Training...")
        
        # 環境をリセットし、初期観測を取得
        obs, info = self.env.reset()

        # ★★★ ここからが修正箇所 ★★★
        
        global_step = 0
        num_updates = 0
        
        # 全体の学習ステップ数が目標に達するまでループ
        while global_step < total_timesteps:
            num_updates += 1
            print(f"\n===== Update Cycle: {num_updates} | Global Step: {global_step}/{total_timesteps} =====")

            # --- データ収集 (Rollout) ---
            # rollout_lenステップ分、データを収集する
            for i in range(self.config.rollout_len):
                
                actions_dict = {}
                log_probs_dict = {}
                values_dict = {}
                entropies_dict = {}
                
                # 1. 全てのエージェントから行動を取得
                for agent_id, agent_obs in obs.items():
                    action, log_prob, entropy, value = self.agents[agent_id].get_action_and_value(agent_obs)
                    actions_dict[agent_id] = action.item()
                    log_probs_dict[agent_id] = log_prob.detach()
                    values_dict[agent_id] = value.detach()
                    entropies_dict[agent_id] = entropy.detach()

                # 2. 環境を1ステップ進める
                next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions_dict)
                dones = {agent_id: terminateds.get(agent_id, False) or truncateds.get(agent_id, False) for agent_id in self.agent_ids}

                # 3. 各エージェントのデータをストレージに保存
                for agent_id in self.agent_ids:
                    self.storages[agent_id].add(
                        obs[agent_id],
                        actions_dict[agent_id],
                        log_probs_dict[agent_id],
                        rewards[agent_id],
                        dones[agent_id],
                        values_dict[agent_id],
                        entropies_dict[agent_id]
                    )
                
                # 観測を更新
                obs = next_obs
                global_step += 1 # 全体のステップ数をインクリメント

            # --- 学習 (Update) ---
            # 4. 全てのエージェントの更新メソッドを呼び出す
            # with torch.no_grad(): # GAE計算中は勾配不要
            #     # ... GAE計算 ...
            
            for agent_id in self.agent_ids:
                rollout_data = self.storages[agent_id].get()
                self.agents[agent_id].update(rollout_data)

        print("Training finished.")