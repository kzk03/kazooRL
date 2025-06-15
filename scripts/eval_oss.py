from pathlib import Path

import torch
import yaml

from kazoo.envs.oss_gym_wrapper import OSSGymWrapper
from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig


def main():
    ROOT = Path(__file__).resolve().parent.parent
    CONFIGS = ROOT / "configs"
    DATA = ROOT / "data"
    MODEL_PATH = ROOT / "models" / "ppo_agent.pt"

    with (CONFIGS / "base.yaml").open() as f:
        cfg = yaml.safe_load(f)

    # dev_profiles.yaml から人数を動的取得
    with (CONFIGS / "dev_profiles.yaml").open() as f:
        profiles = yaml.safe_load(f)
    cfg["n_agents"] = len(profiles)

    # 環境生成
    raw_env = OSSSimpleEnv(
        task_file=str(DATA / "github_data.json"),
        profile_file=str(CONFIGS / "dev_profiles.yaml"),
        n_agents=cfg["n_agents"],
    )
    env = OSSGymWrapper(raw_env)

    # モデル構築
    agent_id = env.agents[0]
    obs_space = env.observation_spaces[agent_id]
    act_space = env.action_spaces[agent_id]

    agent = IndependentPPO(
        obs_space=obs_space,
        act_space=act_space,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_eps=cfg["clip_eps"],
        vf_coef=cfg["vf_coef"],
        ent_coef=cfg["ent_coef"],
        rollout_len=cfg["rollout_len"],
        mini_batch=cfg["mini_batch"],
        epochs=cfg["epochs"],
        device=cfg.get("device", "cpu"),
    )

    agent_model = agent

    # 重み読み込み
    checkpoint = torch.load(MODEL_PATH, map_location=agent.device)
    agent.net.load_state_dict(checkpoint["net"])
    agent.policy_head.load_state_dict(checkpoint["pi"])
    agent.value_head.load_state_dict(checkpoint["vf"])
    print("✅ モデル読み込み完了")

    obs = env.reset()
    done = {agent: False for agent in env.agents}
    total_reward = {agent: 0.0 for agent in env.agents}
    step = 0

    while not all(done.values()):
        actions = {}
        for agent in env.agents:
            if not done[agent]:
                action, logp, value = agent_model.act(obs[agent])
                actions[agent] = action

        next_obs, rewards, terminations, infos = env.step(actions)

        row = f"[step {step}]"
        for agent in env.agents:
            if not done[agent]:
                a = actions[agent]
                r = rewards[agent]
                row += f" {agent}: a={a} r={r:.2f} |"
                total_reward[agent] += r
                done[agent] = terminations[agent]
            else:
                row += f" {agent}: ---        |"
        print(row.rstrip(" |"))

        obs = next_obs
        step += 1

    print("🎯 Evaluation complete!")
    for agent in env.agents:
        print(f"  → {agent}: total reward = {total_reward[agent]:.2f}")


if __name__ == "__main__":
    main()
