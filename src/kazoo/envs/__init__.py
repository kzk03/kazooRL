# src/kazoo/envs/__init__.py
from gymnasium import register
from .oss_gym_wrapper import OSSGymWrapper

register(
    id="OSSSimpleGym-v0",
    entry_point="kazoo.envs.oss_gym_wrapper:OSSGymWrapper"
)