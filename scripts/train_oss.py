# scripts/train_oss.py
import hydra
from omegaconf import DictConfig, OmegaConf
import json

from kazoo.envs.oss_simple import OSSSimpleEnv
# ★★★ インポートするクラスを新しい司令塔クラスに変更 ★★★
from kazoo.learners.independent_ppo_controller import IndependentPPOController


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # ... (環境の準備は同じ) ...
    with open("data/backlog.json", "r") as f:
        backlog = json.load(f)
    with open("configs/dev_profiles.yaml", "r") as f:
        dev_profiles = OmegaConf.load(f)
    env = OSSSimpleEnv(config=cfg, backlog=backlog, dev_profiles=dev_profiles)

    # ★★★ 学習器をIndependentPPOControllerに変更 ★★★
    learner = IndependentPPOController(env=env, config=cfg)
    
    learner.learn(total_timesteps=cfg.total_steps)

    print("Training finished.")


if __name__ == "__main__":
    main()