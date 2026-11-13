"""A set of common constants."""
VISUAL_OBSERVATIONS_PREFIX = "rgb"
ACTION_KEY = "demo_action"

# Full MuJoCo physics state captured per step so a demo can be replayed by
# restoring state (robot + object freejoints) instead of re-applying actions
# open-loop. Persisted as info_qpos / info_qvel via SAFETENSORS_INFO_PREFIX.
STATE_QPOS_KEY = "qpos"
STATE_QVEL_KEY = "qvel"

TRACKED_PACKAGES = ["mujoco", "roboeval"]

SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_OBSERVATION_PREFIX = "obs_"
SAFETENSORS_INFO_PREFIX = "info_"

GYM_OBSERVATION_KEY = "observation"
GYM_REWARD_KEY = "reward"
GYM_TERMINATIION_KEY = "termination"
GYM_TRUNCATION_KEY = "truncation"
GYM_INFO_KEY = "info"
