import time
import os
import numpy as np
import torch
import math as m
from dm_control import mjcf
from dm_control import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm, AG95
from manipulator_mujoco.obstacles import Inv_pendulum
from manipulator_mujoco.controllers import Rm63_Controller
from manipulator_mujoco.rl_program.ounoise import OUNoise
from manipulator_mujoco.utils.fkine import arm_obstacle_distance_detection
from manipulator_mujoco.utils.transform_utils import *


class rm63Env(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-0.01, high=0.01, shape=(6,), dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        self._mjcf_model = mjcf.RootElement()
        # checkerboard floor 定义环境
        self._arena = StandardArena()

        # rm63 arm 定义机械臂
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/rm63/urdf/rm_63_6f.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site',
            actuator_names='actuator',
        )

        # 定义障碍物
        self._obstacle = Inv_pendulum(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/obstacles/inv_pendulum.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site',
            actuator_names='actuator',
        )

        # attach arm to arena 把机械臂放入环境
        self._arena.attach(
            self._arm.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0]
        )
        # 把障碍物放入环境
        self._arena.attach(
            self._obstacle.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0]
        )

        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)
        self.contacts = self._physics.data.contact[:]

        # set up OSC controller
        self._Rm63_controller = Rm63_Controller(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=4000,
            damping_ratio=0.707,
        )

        self._Obstacle_controller = Rm63_Controller(
            physics=self._physics,
            joints=self._obstacle.joints,
            eef_site=self._obstacle.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=4000,
            damping_ratio=0.8,
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep  # 仿真时间间隔
        self._viewer = None
        self._step_start = None

        # 障碍物运动初始化
        self.t_init = 0.0  # 初始时间,通过随机此变量,随机初始位置(轨迹为正弦)
        self.step_max = 1000  # 障碍物最大循环步数
        self.t_max = self.step_max * self._timestep  # 最长时间
        self.time_sequence = np.arange(0, self.t_max, self._timestep)  # 用于障碍物运动的时间序列
        self.obstacle_xpos_old = np.zeros(3)  # 初始化障碍物末端上一时刻位置

        self.step_count = 0  # 步数

        # OU噪声
        self.ounoise = OUNoise(self.action_space.shape[0])  # 定义
        self.ounoise_noise_scale = 0.01  # 初始OU噪声（随循环次数调整为final）
        self.ounoise_final_noise_scale = 0.0001  # 最终OU噪声
        self.exploration_end = 1000  # 调整次数，超过次数的循环均为最终OU噪声

    def _get_obs(self) -> torch.tensor:
        # 障碍物速度
        self.obstacle_xvel = (self._physics.bind(self._obstacle.eef_site).xpos - self.obstacle_xpos_old)/self._timestep
        self.arm_quat = mat2quat(self._physics.bind(self._arm.eef_site).xmat.reshape([3, 3]))  # 机械臂末端姿态四元数 4
        # observation = np.concatenate([self._physics.data.qpos[:6],  # 机械臂关节角，6
        #                               self._physics.data.qvel[:6],  # 机械臂关节角速度，6
        #                               self._physics.bind(self._arm.eef_site).xpos,  # 机械臂末端位置 3
        #                               self.arm_quat,  # 机械臂末端姿态四元数 4
        #                               self.target_pose,  # 目标位置及姿态四元数 7
        #                               self._physics.bind(self._obstacle.eef_site).xpos,  # 障碍物位置，3
        #                               self.obstacle_xvel,  # 障碍物速度，3
        #                               ])
        observation = np.concatenate([self._physics.data.qpos[:6],  # 机械臂关节角，6
                                      self._physics.data.qvel[:6],  # 机械臂关节角速度，6
                                      self._physics.bind(self._arm.eef_site).xpos,  # 机械臂末端位置 3
                                      self.target_pose[0:3],  # 目标位置 3
                                      ])
        # 更新上一时刻位置，用于下一时刻计算速度
        self.obstacle_xpos_old = np.copy(self._physics.bind(self._obstacle.eef_site).xpos)
        observation = torch.tensor(observation, dtype=torch.float32)
        return observation

    @property
    def get_obs(self):
        return self._get_obs()

    def _get_info(self) -> dict:
        # TODO come up with an info dict that makes sense for your RL task
        # just注释
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            arm_pos_init = [
                0.0 + m.pi * (0.1 * np.random.rand() - 0.05),
                m.pi * 0.2 + m.pi * (0.1 * np.random.rand() - 0.05),
                -m.pi * 0.4 + m.pi * (0.1 * np.random.rand() - 0.05),
                0.0 + m.pi * (0.1 * np.random.rand() - 0.05),
                -m.pi / 2 + m.pi * (0.1 * np.random.rand() - 0.05),
                0.0 + m.pi * (0.1 * np.random.rand() - 0.05)
            ]
            # arm_pos_init = [0.0, m.pi * 0.2, -m.pi * 0.4, 0.0, -m.pi / 2, 0.0]
            # self._physics.bind(self._arm.joints).qpos[:] = arm_pos_init

            self._physics.bind(self._arm.joints).qpos[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # self._physics.bind(self._arm.joints).qvel[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # self._physics.bind(self._arm.joints).qacc[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # 随机化target位置
            quat_targ_theta = np.random.rand() * m.pi/6 * 0
            quat_targ_u = np.array([1, 1, 1])  # np.random.rand(3)  # 目标四元数转轴
            quat_targ_u = quat_targ_u/np.linalg.norm(quat_targ_u) * np.sin(quat_targ_theta/2)
            # self.target_pose = np.array([-0.6 + 0.1 * np.random.rand() * 0,
            #                              0.0 + 0.4 * np.random.rand() * 0,
            #                              0.2 + 0.2 * np.random.rand() * 0, np.cos(quat_targ_theta),
            #                              quat_targ_u[0], quat_targ_u[1], quat_targ_u[2]])
            self.target_pose = np.array([-0.5, 0.0, 0.3, 0, 1, 0, 0])

            self.t_init = np.random.rand()   # 随机障碍物初始时间
            self.step_count = 0

            # 障碍物位置初始化
            self._physics.bind(self._obstacle.joints).qpos = [
                np.sin(2 * m.pi * self.t_init),  # [-1, 1)
                0.0
            ]

            # 障碍物轨迹
            self.obstacle_track = np.sin(2 * m.pi / 0.2 * (self.t_init + self.time_sequence))
            # 障碍物末端上一时刻位置
            self.obstacle_xpos_old = np.copy(self._physics.bind(self._obstacle.eef_site).xpos)

            self.step_wait(self._physics.bind(self._arm.joints).qpos[:],
                           self._physics.bind(self._obstacle.joints).qpos, 2.000)

            # 可视化，调试用
        # self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action) -> tuple:
        action = torch.clip(action, -1 / 57.3 * 25, 1 / 57.3 * 25).detach().cpu().numpy()
        # 动机械臂
        rm63_state = self._physics.data.qpos[0:6].reshape([1, 6])
        rm63_target = rm63_state + action.reshape([1, 6])  # 计算目标运动角度，用来稳定
        self._Rm63_controller.run(action.reshape([1, 6]))  # 输入控制器

        # 动障碍物
        obstacle_state = self.get_obs_obstacle()
        obstacle_target = np.zeros([1, 2])  # 初始化障碍物目标位置
        obstacle_target[0, 0] = self.obstacle_track[self.step_count]  # 导入障碍物目标位置
        obstacle_action = obstacle_target - obstacle_state  # 计算位置变化量

        self._Obstacle_controller.run(obstacle_action.reshape([1, 2]))  # 输入控制器
        self._physics.step()  # 填入迭代步长

        # 稳定运动 之前测试关节调节时间大约为120ms(3deg阶跃)
        # self.step_wait(rm63_target, obstacle_target, 0.200)

        obstacle_step_position, obstacle_radius = self.get_obstacle_pos_rad(obstacle_target)
        flag, o_a_distance = arm_obstacle_distance_detection(self._physics.data.qpos[:6],
                                                             obstacle_step_position,
                                                             obstacle_radius,
                                                             seg_num=[10, 10, 40, 10, 30, 10, 10])

        # render frame 在mujoco中可视化显示
        if self._render_mode == "human":
            self._render_frame()

        observation = self._get_obs()

        # reward
        delta, p, d_ref, c1, c2, c3 = 0.3, 8, 0.2, 1000, 100, 60
        # R_T 终点
        d_t_tran = self._Rm63_controller.distance(self.target_pose)  # 位置
        d_t_pos = self._Rm63_controller.get_end_dis(self.target_pose)  # 姿态
        # d_T = d_t_tran + 0.1 * d_t_pos
        d_T = d_t_tran
        if d_T < delta:
            R_T = -d_T ** 2 / 2
        else:
            R_T = -delta * (d_T - delta / 2)
        # R_A 动作
        R_A = -np.linalg.norm(action) ** 2
        # R_O 障碍
        d_O = np.min(o_a_distance)
        R_O = -(d_ref / (d_O + d_ref)) ** p

        # reward = c1 * R_T + c2 * R_A + c3 * R_O
        reward = c1 * R_T

        done = False  # 是否结束循环
        terminated = False  # 是否到终点
        # if d_t_tran < 0.01 and d_t_pos < 0.05:  # 到终点
        if d_t_tran < 0.05:  # 到终点
            done = True
            terminated = True
            reward += 50000
        if self.contacts.dist.shape[0] != 0:  # 碰撞
            done = True
            terminated = False
            reward -= 10000
        if self.step_count >= self.step_max - 1:  # 循环次数过多
            done = True
            terminated = False
            reward -= 10000

        self.step_count += 1
        info = self._get_info()

        return observation, reward, done, terminated, info

    # 等待运动稳定
    def step_wait(self, rm63_target, obstacle_target, t=0.060):
        n = int(t / self._timestep)
        # print("n = ", n)
        for ii in range(n):
            # 取当前状态
            rm63_state = self._physics.data.qpos[0:6].reshape([1, 6])
            obstacle_state = self._physics.data.qpos[6:8].reshape([1, 2])
            # 计算运动量action
            rm63_action = rm63_target.reshape([1, 6]) - rm63_state
            obstacle_action = obstacle_target.reshape([1, 2]) - obstacle_state
            # 放入控制器
            self._Rm63_controller.run(rm63_action)
            self._Obstacle_controller.run(obstacle_action)
            # print(self._physics.data.qvel[:])
            # if np.any(np.abs(torque_rm63) == 150.0) or np.any(np.abs(torque_obstacle) == 150.0):
            #     abc = 1
            #     break
            self._physics.step()
        # print(np.all(np.abs(self._physics.data.qvel[:]) < 1e-3))
        abc = 1

    # 获取障碍物绝对位置和半径，从xml中人工搬运至此
    def get_obstacle_pos_rad(self, obstacle_track_angle):
        obstacle_base_pos = np.array([-0.3, 0, 0])
        angle = obstacle_track_angle.reshape([2])[0]
        obstacle_pos = obstacle_base_pos + np.array([0, -0.3*np.sin(angle),
                                                     0.3*np.cos(angle)])
        obstacle_rad = 0.05
        return obstacle_pos, obstacle_rad

    def move123(self, angle, obs_flag=False):
        if ~obs_flag:
            self._Rm63_controller.run(angle.reshape([1, 6]))
            # step physics
            self._physics.step()  # 填入迭代步长
        observation = self._get_obs()

        return observation

    def get_rm63_vel_acc(self):
        velocity = self._physics.data.qvel[0:6]
        acceleration = self._physics.data.qacc[0:6]
        return velocity, acceleration

    def get_obs_obstacle(self):
        observation = self._physics.data.qpos[6:8]

        return observation

    def move456(self, angle, obs_flag=False):
        if ~obs_flag:
            self._Obstacle_controller.run(angle.reshape([1, 2]))
            # step physics
            self._physics.step()  # 填入迭代步长
        observation = self._physics.data.qpos[6:8]

        return observation

    def show(self):
        self._render_frame()

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

            # TODO come up with a better frame rate keeping strategy
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
