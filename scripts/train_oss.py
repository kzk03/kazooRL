from pathlib import Path
from types import SimpleNamespace

import yaml

from kazoo.envs.oss_gym_wrapper import OSSGymWrapper
from kazoo.envs.oss_simple import make_oss_env
from kazoo.learners.indep_ppo import IndependentPPO, PPOConfig


def main():
    ROOT = Path(__file__).resolve().parent.parent
    CONFIGS = ROOT / "configs"
    DATA = ROOT / "data"

    # Load base config
    with (CONFIGS / "base.yaml").open() as f:
        cfg = yaml.safe_load(f)

    # dev_profiles.yaml から人数を動的取得
    with (CONFIGS / "dev_profiles.yaml").open() as f:
        profiles = yaml.safe_load(f)
    cfg["n_agents"] = len(profiles)
    

    # Create env and wrap
    raw_env = make_oss_env(
        task_file=str(DATA / "github_data.json"),
        profile_file=str(CONFIGS / "dev_profiles.yaml"),
        n_agents=cfg["n_agents"],
    )
    env = OSSGymWrapper(raw_env)

    # Select agent
    agent_id = env.agents[0]
    act_space = env.action_spaces[agent_id]
    obs_space = env.observation_spaces[agent_id]

    print(f"DEBUG: agent = {agent_id}")
    print(f"DEBUG: action_space = {act_space}")
    print(f"DEBUG: action_space type = {type(act_space)}")

    # Set PPO config
    ppo_cfg = PPOConfig(
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

    agent.train(env, total_steps=cfg["total_steps"])


if __name__ == "__main__":
    main()
