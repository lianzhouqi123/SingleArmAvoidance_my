import time
import os
import numpy as np
from dm_control import mjcf
from dm_control import mujoco
import mujoco.viewer
# import gym
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, AG95
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.controllers import Rm63_Controller
import math as m
from manipulator_mujoco.utils.fkine import arm_obstacle_distance_detection

"""
env_situation4: 随机障碍（半径一定），随机目标
"""


class Rm63Env_s4_1_1(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(self, render_mode=None,
                 reach_level=0.030,                                     # 目标达到阈值判定
                 reach_reward_1=1200,                                   # 目标达到奖励1
                 reach_reward_2=100,                                    # 目标到达奖励2
                 reach_distance_parameter=5000,                         # 目标距离奖励因子
                 target_original_pos=[-0.7, -0.2, 0.1],                 # 随机目标位置采样原点
                 pose_range=[0.4, 0.4, 0.4],                            # 随机目标位置采样范围
                 obstacle_original_pos=[-0.7, -0.2, 0.1],               # 障碍物位置采样原点
                 obstacle_pos_range=[0.4, 0.4, 0.4],                    # 障碍物位置采采样范围
                 obstacle_radiance_fixed=0.05,                          # 障碍物预设半径
                 seg_num=[10, 10, 10, 10, 10, 10, 10, 10, 10],          # 机械臂包络判定球
                 obstacle_parameter_a=-0.04958,                         # 障碍物避障奖励因子a
                 obstacle_parameter_b=-0.50420,                         # 障碍物避障奖励因子b
                 ):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=-0.01, high=0.01, shape=(6,), dtype=np.float64
        )

        # 全局参数传递
        self.reach_level = reach_level
        self.reach_reward_1 = reach_reward_1
        self.reach_reward_2 = reach_reward_2
        self.reach_distance_parameter = reach_distance_parameter
        self.pose_range = pose_range
        self.obstacle_pos_range = obstacle_pos_range
        self.seg_num = seg_num
        self.obstacle_parameter_a = obstacle_parameter_a
        self.obstacle_parameter_b = obstacle_parameter_b



        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode
        ############################
        # create MJCF model
        ############################
        self._mjcf_model = mjcf.RootElement()
        # checkerboard floor
        self._arena = StandardArena()

        # set up rm63 arm
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/rm63/urdf/rm_63_6f.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site',
            actuator_names='actuator',
        )

        # attach arm to arena
        self._arena.attach(self._arm.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0])

        # set up the original position of obstacle (also the first position of obstacle)
        self.obstacle_pos = obstacle_original_pos
        # set up the obstacle radiance, unit: m
        self.obstacle_radiance = obstacle_radiance_fixed
        # initialize the obstacle
        self.obstacle(ori_pos=self.obstacle_pos, obstacle_shape="sphere", quat=[1, 0, 0, 0],
                      size=str(self.obstacle_radiance), rgba=[1, 0, 0, 1], show_range=True)


        # set up the "original" position of target
        self.target_original_pos = target_original_pos

        # generate physics model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        # 排除obstacle生成过程中的碰撞
        # _base_y = self._physics.named.data.geom["_base_y"]
        # _base_x = self._physics.named.data.geom["_base_x"]
        # self._physics.bind(ignore_collisions=[(_base_x, _base_y)])



        # set up RM63 controller
        self._Rm63_controller = Rm63_Controller(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=6500,
            damping_ratio=0.32,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_obs(self) -> np.ndarray:
        # 关节位置（1*6）
        position = self._physics.data.qpos[:6]

        # 关节转速（1*6）
        velocity = self._physics.data.qvel[:6]

        # 目标位置（1*3）
        target_pose = self.target_sample(mode="range_sample_mode", pos_range=self.pose_range)  # （1*7）
        target_xyz = target_pose[0:3]                             # （1*3）

        # 夹爪中心位置（1*3）
        ee_pose_xyz = self._physics.bind(self._arm.eef_site).xpos  # (1*3)

        # 末端距离差值（1*1）
        distance = [self._Rm63_controller.distance(target_pose)]

        # 障碍球位置与距离（1*5）
        obstacle_xyz = self.obstacle_step_position    # (1*3)

        obstacle_radiance = self.obstacle_radiance  # (1*1)

        flag, obstacle_distance_seg_num = arm_obstacle_distance_detection(position,
                                                                          obstacle_xyz,
                                                                          self.obstacle_radiance,
                                                                          self.seg_num)
        obstacle_distance = np.min(obstacle_distance_seg_num)  # (1*1)

        obstacle_safe_distance = [obstacle_distance - obstacle_radiance]
        # qpos前6项为关节角度
        observation = np.concatenate((position,                     # (1*6)
                                      velocity,                     # (1*6)
                                      distance,                     # (1*1)
                                      ee_pose_xyz,                  # (1*3)
                                      target_xyz,                   # (1*3)
                                      obstacle_xyz,                 # (1*3)
                                      obstacle_safe_distance,       # (1*1)
                                      [obstacle_distance]           # (1*1)
                                     ))
        return observation

    def _get_reward(self, observation, _observation):
        """
        奖励
        """

        # 速度Rv
        _velocity = _observation[6:12]
        Rv = -np.linalg.norm(_velocity)*0.1

        # 末端距离Rd
        distance = observation[12]
        _distance = _observation[12]
        Rs = 0
        if _distance < self.reach_level:
            Rd = self.reach_reward_1
            terminated = True
        else:
            Rd = (distance - _distance) * self.reach_distance_parameter - _distance
            terminated = False
            pass

        # 碰撞检测Rc
        contact = self._physics.data.ncon
        contact = contact - 2
        if contact == 0:
            Rc = 0
        else:
            Rc = -200

        # 避障
        safe_distance = _observation[22]
        if safe_distance < 0.05:
            Ro = -2000
            truncated = True
        elif safe_distance>=0.05 and safe_distance<0.1:
            Ro = self.obstacle_parameter_b/(safe_distance+self.obstacle_parameter_a)
            truncated = False
        else:
            Ro = 0
            truncated = False

        # 单个时间步内的奖励值
        reward = Rd + Rv + Rs + Rc + Ro
        return reward, terminated, truncated


    def _get_info(self) -> dict:
        # just注释
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            self._physics.bind(self._arm.joints).qpos[:] = [
                0.0,
                m.pi / 6,
                -m.pi * 0.6,
                0.0,
                -1.5707,
                0.0
            ]
        # init the position of the obstacle
        self.obstacle_control(mode="static", pos_range=[0.4, 0.4, 0.4])
        # init the position of the target
        self.target_sample(mode="range_sample_mode", pos_range=self.pose_range)

        observation = self._get_obs()
        info = self._get_info()

        if self._render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # 施加动作前
        observation = self._get_obs()
        # action = np.array(action).squeeze(axis=0)
        # make obstacle gravity bias in z-axis
        self._physics.data.qfrc_applied[-1] = self._physics.data.qfrc_bias[-1]
        # 单步动作
        self._Rm63_controller.run(action)
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()

        _observation = self._get_obs()
        reward, terminated, truncated = self._get_reward(observation, _observation)
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )

        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer

            self._viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()

    # ------------------------------------------------Main Env Program Ends----------------------------------------------- #
    """
    Utilized functions in env only.
    For rapidly development, I put them here.
    """

    def obstacle(self, obstacle_shape: str, ori_pos: list, quat: list, size: str, rgba: list, show_range: bool = False):
        """
        Purpose: generate an obstacle with 3DOF in space
        obstacle_shape: shape of obstacle, "sphere" or "box"...
        pos: the original point of obstacle, instead of the central point
        quat: the quaternion of obstacle
        size: sphere-radius, box-xyz half distance
        show_range: show the range of obstacle in simulation space
        """

        # generate x and y axis
        _base_y = Primitive(name="_base_y", type="sphere", size="0.001", rgba=[1, 0, 0, 0], conaffinity=0,
                            contype=0, option="geom")
        _base_x = Primitive(name="_base_x", type="sphere", size="0.001", rgba=[0, 1, 0, 0], conaffinity=0,
                            contype=0, option="geom")
        # generate obstacle
        _obstacle = Primitive(name="obstacle", type=obstacle_shape, size=size, rgba=rgba, option="geom")
        # original point of obstacle
        pos = ori_pos

        # attach _base_y to arena
        _box_frame, freejoint = self._arena.attach_axis(_base_y.mjcf_model, pos=[-3, 0 + pos[1], 0], quat=[1, 0, 0, 0],
                                                        joint_axis=[-1, 0, 0])
        # attach _base_x to _base_y and add x-slide joint
        frame_base_x = _base_y.mjcf_model.attach(_base_x.mjcf_model)
        frame_base_x.pos = [3 + pos[0], 3, 0]
        frame_base_x.add("joint", type="slide", axis=[0, -1, 0])
        # attach obstacle to _base_x and add y-slide joint
        frame_obstacle = _base_x.mjcf_model.attach(_obstacle.mjcf_model)
        frame_obstacle.pos = [0, -3, 0 + pos[2]]
        frame_obstacle.quat = quat
        frame_obstacle.add("joint", type="slide", axis=[0, 0, 1])

        if show_range == True:
            # render the range of the obstacle in simulation space
            obstacle_space_render = Primitive(name="_base_y", type="box", size="0.2 0.2 0.2", rgba=[1, 0, 1, 0.1],
                                              option="site")
            # obstacle_range_render pos is the central point method to set up
            self._arena.attach(obstacle_space_render.mjcf_model, pos=[pos[0] + 0.2, pos[1] + 0.2, pos[2] + 0.2],
                               quat=[1, 0, 0, 0])

    def obstacle_control(self, mode: str, pos_range: list = [0, 0, 0], vel_range: list = [0, 0, 0]):
        """
        Purpose: control the obstacle in static randomly mode or dynamic mode,
        obstacle central position: have been modified in env init
        pos_range: !!![y, x, z] position range of random sample!!!
        vel_range: !!![y, x, z] velocity range of random sample!!!
        """

        if mode == "static":
            # gravity bias in z-axis
            # axis have been modified in env init
            # self.obstacle_step_position = np.array([-np.random.random() * pos_range[1],
            #                                         -np.random.random() * pos_range[0],
            #                                         +np.random.random() * pos_range[2]])
            self.obstacle_step_position = np.array([-0.7, -0.2, 0.1])
            self._physics.data.qpos[6:] = self.obstacle_step_position


        elif mode == "dynamic":
            # need be modified
            self._physics.data.qvel[6:] = np.array([vel_range[0], vel_range[1], vel_range[2]])

        return self.obstacle_step_position

    def target_sample(self, pos_range: list, mode: str, ori_pos: list = [0, 0, 0]):
        """
        Purpose: sample the target position in space, with specified range
        In this stage, we set up the target and the obstacle in the same box space
        "range_sample_mode" : sample the target position in specified range
        "step_refresh_mode" : refresh the target position once during one episode(need to be modified)
        pos_range: set up the sample range of target position
        ori_pos: the original position of target
        !!target_original_pos: have been set up in env init part!!
        """

        if mode == "range_sample_mode":
            self.target_pose = np.array([self.target_original_pos[0] + pos_range[0] * np.random.random(),
                                         self.target_original_pos[1] + pos_range[1] * np.random.random(),
                                         self.target_original_pos[2] + pos_range[2] * np.random.random(),
                                         1, 0, 0, 0])
            distance_t_o = np.linalg.norm(self.obstacle_step_position - ori_pos[:3])
            # if target pose show up inner of the obstacle, resample the target pose
            while distance_t_o <= self.obstacle_radiance + 0.15:
                self.target_pose = np.array([self.target_original_pos[0] + pos_range[0] * np.random.random(),
                                             self.target_original_pos[1] + pos_range[1] * np.random.random(),
                                             self.target_original_pos[2] + pos_range[2] * np.random.random(),
                                             1, 0, 0, 0])


        elif mode == "step_refresh_mode":
            self.target_pose = np.array([ori_pos[0],
                                         ori_pos[1],
                                         ori_pos[2],
                                         1, 0, 0, 0])

        return self.target_pose