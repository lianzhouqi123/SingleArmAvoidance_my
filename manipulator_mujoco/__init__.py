# Import the registration function from Gymnasium
from gymnasium.envs.registration import register
register(
    id="manipulator_mujoco/rm63Env-v0",
    entry_point="manipulator_mujoco.envs:rm63Env",
)

register(
    id="Rm63Env-s1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s1",
)

register(
    id="Rm63Env-s1-1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s1_1",
)

register(
    id="Rm63Env-s2",
    entry_point="manipulator_mujoco.envs:Rm63Env_s2",
)

register(
    id="Rm63Env-s2-1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s2_1",
)

register(
    id="Rm63Env-s3",
    entry_point="manipulator_mujoco.envs:Rm63Env_s3",
)

register(
    id="Rm63Env-s3-1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s3_1",
)

register(
    id="Rm63Env-s4",
    entry_point="manipulator_mujoco.envs:Rm63Env_s4",
)

register(
    id="Rm63Env-s4-1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s4_1",
)

register(
    id="Rm63Env-s5-1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s5_1",
)

register(
    id="Rm63Env-s4-1-1",
    entry_point="manipulator_mujoco.envs:Rm63Env_s4_1_1",
)

register(
    id="Rm63Env-s4-2",
    entry_point="manipulator_mujoco.envs:Rm63Env_s4_2",
)

register(
    id="Rm63Env-s5-2",
    entry_point="manipulator_mujoco.envs:Rm63Env_s5_2",
)
