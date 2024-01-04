import time
import os
import numpy as np
import torch
from dm_control import mjcf
from dm_control import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.props import Primitive
from manipulator_mujoco.controllers import Rm63_Controller
from manipulator_mujoco.utils.transform_utils import *
from manipulator_mujoco.utils.fkine import arm_obstacle_distance_detection
import math as m

"""
env_situation1: 固定目标，固定初始,碰撞不停,固定障碍
"""


class Rm63Env_s3(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(
            self,
            reach_level=0.01,
            reach_reward=50,
            reach_distance_parameter=3000,
            target_original_pos=[-0.7, -0.2, 0.1],
            pose_range=[0.4, 0.4, 0.4],
            render_mode=None,
    ):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
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

        self.reach_level = reach_level
        self.reach_reward = reach_reward
        self.reach_distance_parameter = reach_distance_parameter

        # attach arm to arena
        self._arena.attach(self._arm.mjcf_model, pos=[0, 0, 0], quat=[1, 0, 0, 0])

        # obstacle
        self.obstacle_ori_pos = np.array([-0.3, 0.0, 0.4])
        self.obstacle_quat = [1, 0, 0, 0]
        self.obstacle_size = 0.05
        self.obstacle(obstacle_shape="sphere", ori_pos=self.obstacle_ori_pos, quat=self.obstacle_quat,
                      size=str(self.obstacle_size), rgba=[0, 1, 0, 0], show_range=False)

        # set up the original position of target
        self._target = Target(self._arena.mjcf_model)
        self.target_original_pos = target_original_pos
        self.pose_range = pose_range

        # generate physics model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

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

        self.step_count = 0  # 步数
        self.step_max = 600  # 最大循环步数
        self.flag_cont = False
        self.dis_o_old = 1  # 上次和障碍物最小距离

    def _get_obs(self) -> torch.tensor:
        """
        Observation
        """
        # 关节位置（1*6）
        position = self._physics.data.qpos[:6]
        # 关节转速（1*6）
        velocity = self._physics.data.qvel[:6]
        # 目标位置（1*3）
        target_pose = self.target_pose  # （1*7）
        target_xyz = target_pose[0:3]  # （1*3）
        # 末端位置（1*3）
        ee_pose_xyz = self._physics.bind(self._arm.eef_site).xpos  # (1*3)
        # 机械臂末端姿态四元数 （1*4）
        ee_pose_quat = mat2quat(self._physics.bind(self._arm.eef_site).xmat.reshape([3, 3])).reshape([4])  # 4
        # 末端距离差值（1*1）
        distance = [self._Rm63_controller.distance(target_pose)]
        # 障碍球位置大小（1*4）
        obstacle = np.hstack([self.obstacle_step_position, np.array([self.obstacle_size])]).reshape([4])

        observation = np.concatenate(
            (
                position,  # （1*6）机械臂关节角，6
                velocity,  # （1*6）机械臂关节角速度，6
                ee_pose_xyz,  # （1*3）机械臂末端位置 3
                # ee_pose_quat,  # （1*4）机械臂末端姿态四元数 4
                target_xyz,  # （1*3）目标位置 3
                obstacle,  # （1*4）障碍球位置大小 4
            )
        )
        observation = torch.tensor(observation, dtype=torch.float32)
        return observation

        # def _get_reward(self, action):
        #     """
        #     奖励
        #     """
        #     # 与障碍物距离
        #     flag, o_a_distance = arm_obstacle_distance_detection(self._physics.data.qpos[:6],
        #                                                          self.obstacle_step_position,
        #                                                          self.obstacle_size,
        #                                                          seg_num=[10, 10, 40, 10, 30, 10, 10, 10, 10])
        #     # reward
        #     delta, p, d_ref, c1, c2, c3 = 0.3, 2, 0.2, 1000, 100, 1060
        #     # R_T 终点
        #     d_t_tran = self._Rm63_controller.distance(self.target_pose)  # 位置
        #     d_t_pos = self._Rm63_controller.get_end_dis(self.target_pose)  # 姿态
        #     # d_T = d_t_tran + 0.1 * d_t_pos
        #     d_T = d_t_tran
        #     if d_T < delta:
        #         R_T = -d_T ** 2 / 2
        #     else:
        #         R_T = -delta * (d_T - delta / 2)
        #     # R_A 动作
        #     R_A = -np.linalg.norm(action) ** 2
        #     # R_O 障碍
        #     d_O = np.min(o_a_distance)
        #     R_O = -(d_ref / (d_O + d_ref)) ** p
        #
        #     # reward = c1 * R_T + c2 * R_A + c3 * R_O
        #     reward = c1 * R_T
        #
        #     done = False  # 是否结束循环
        #     terminated = False  # 是否到终点
        #     self.flag_cont = False
        #     contact = self._physics.data.ncon
        #     if contact != 0:  # 碰撞
        #         done = False
        #         terminated = False
        #         self.flag_cont = True
        #         reward -= 200
        #     if d_t_tran < 0.01:  # 到终点
        #         done = True
        #         terminated = True
        #         reward += 100000
        #     elif self.step_count >= self.step_max - 1:  # 循环次数过多
        #         done = True
        #         terminated = False
        #
        #     return reward, done, terminated

    def _get_reward(self, dis_o_old):
        """
        奖励
        """
        done = False  # 是否结束循环
        terminated = False  # 是否到终点
        self.flag_cont = False
        contact = self._physics.data.ncon
        # 与障碍物距离
        flag, o_a_distance = arm_obstacle_distance_detection(self._physics.data.qpos[:6],
                                                             self.obstacle_step_position,
                                                             self.obstacle_size,
                                                             seg_num=[10, 10, 40, 10, 30, 10, 10, 10, 10])
        # R_O 障碍
        dis_o = np.min(o_a_distance)
        # reward
        rg, rc, wo_far, wo_near, wt, d_goal, d_danger = 20000, -250, 80, 100, -100, 0.01, 0.2
        # R_T 终点
        d_t_tran = self._Rm63_controller.distance(self.target_pose)  # 位置
        d_t_pos = self._Rm63_controller.get_end_dis(self.target_pose)  # 姿态
        d_target = d_t_tran
        if dis_o - dis_o_old > 0:  # 往远处走收益更小，防止反复横跳
            r_avoid = wo_far * (dis_o - dis_o_old)
        else:
            r_avoid = wo_near * (dis_o - dis_o_old)
        r_target = wt * d_target
        if d_target < d_goal:  # 到终点
            reward = rg
            done = True
            terminated = True
        elif contact != 0:  # 碰撞
            reward = rc
            # done = True
            done = False
            terminated = False
            self.flag_cont = True
        elif dis_o < d_danger:  # 进入障碍物的危险距离
            reward = r_avoid
        else:  # 未进入障碍物危险距离
            reward = r_target

        if self.step_count >= self.step_max - 1 and done == False:  # 循环次数过多
            done = True
            terminated = False

        return reward, done, terminated, dis_o

    def _get_info(self) -> dict:
        # just注释
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        # reset physics
        with self._physics.reset_context():
            # put arm in a reasonable starting position
            # arm_pos_init = [
            #     0.0 + m.pi * 0.1 * (np.random.rand() - 0.5),
            #     m.pi * 0.2 + m.pi * 0.1 * (np.random.rand() - 0.5),
            #     -m.pi * 0.4 + m.pi * 0.1 * (np.random.rand() - 0.5),
            #     0.0 + m.pi * 0.1 * (np.random.rand() - 0.5),
            #     -m.pi / 2 + m.pi * 0.1 * (np.random.rand() - 0.5),
            #     0.0 + m.pi * 0.1 * (np.random.rand() - 0.5)
            # ]
            arm_pos_init = [0.0, m.pi * 0.2, -m.pi * 0.4, 0.0, -m.pi / 2, 0.0]
            self._physics.bind(self._arm.joints).qpos[:] = arm_pos_init

            # self._physics.bind(self._arm.joints).qpos[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # self._physics.bind(self._arm.joints).qvel[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # self._physics.bind(self._arm.joints).qacc[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # init the position of the target
        self.target_sample(mode="fixed_sample_mode", pos_range=self.pose_range)

        # obstacle
        obstacle_pos_range = np.array([0, 0, 0])
        self.obstacle_control(mode="static", ori_pos=self.obstacle_ori_pos, pos_range=obstacle_pos_range)

        if self._render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()

        self.step_count = 0
        flag, o_a_distance = arm_obstacle_distance_detection(self._physics.data.qpos[:6],
                                                             self.obstacle_step_position,
                                                             self.obstacle_size,
                                                             seg_num=[10, 10, 40, 10, 30, 10, 10, 10, 10])
        # R_O 障碍
        dis_o = np.min(o_a_distance)
        self.dis_o_old = dis_o

        return observation, info

    def step(self, action) -> tuple:
        action = action.detach().cpu().numpy()

        # 施加动作前
        observation = self._get_obs()

        # 将单个步长输出限制后输出至控制器
        self._Rm63_controller.run(action)
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()

        reward, done, terminated, self.dis_o_old = self._get_reward(self.dis_o_old)
        info = self._get_info()

        self.step_count += 1

        return observation, reward, done, terminated, info

    @ property
    def contact_detect(self):
        contact = self._physics.data.ncon
        return contact

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

    def obstacle(self, obstacle_shape: str, ori_pos: np.ndarray, quat: list, size: str,
                 rgba: list, show_range: bool = False):
        """
        Purpose: generate an obstacle with 3DOF in space
        obstacle_shape: shape of obstacle, "sphere" or "box"...
        pos: the original point of obstacle, instead of the central point
        quat: the quaternion of obstacle
        show_range: show the range of obstacle in simulation space
        """

        # generate x and y axis
        _base_y = Primitive(name="_base_y", type="sphere", size="0.001", rgba=[1, 0, 0, 0], conaffinity=0,
                            contype=0)
        _base_x = Primitive(name="_base_x", type="sphere", size="0.001", rgba=[0, 1, 0, 0], conaffinity=0,
                            contype=0)
        # generate obstacle
        _obstacle = Primitive(name="obstacle", type=obstacle_shape, size=size, rgba=rgba)
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

        self.obstacle_mocap = self._arena.mjcf_model.worldbody.add("body", name="mocap_obstacle", mocap=True)
        self.obstacle_mocap.add(
            "geom",
            type=obstacle_shape,
            size=[float(size)] * 3,
            rgba=[0, 1, 1, 0.2],
            conaffinity=0,
            contype=0,
            pos=[0.0, 0.0, 0.0],  # -0.5, -0.2 + 0.4 * np.random.random(), 0.1
            quat=[1, 0, 0, 0],
        )

        if show_range == True:
            # render the range of the obstacle in simulation space
            obstacle_space_render = Primitive(name="_base_y", type="box", size="0.2 0.2 0.2", rgba=[1, 0, 1, 0.1])
            # obstacle_range_render pos is the central point method to set up
            self._arena.attach(obstacle_space_render.mjcf_model, pos=[pos[0] + 0.2, pos[1] + 0.2, pos[2] + 0.2],
                               quat=[1, 0, 0, 0])

    def obstacle_control(self, mode: str, ori_pos: np.ndarray, pos_range: np.ndarray = np.array([0, 0, 0])
                         , vel_range: np.ndarray = np.array([0, 0, 0])):
        """
        Purpose: control the obstacle in static randomly mode or dynamic mode,
        obstacle central position: have been modified in env init
        pos_range: !!![y, x, z] position range of random sample!!!
        vel_range: !!![y, x, z] velocity range of random sample!!!
        """

        if mode == "static":
            # gravity bias in z-axis
            # axis have been modified in env init
            self.obstacle_step_position = np.array([ori_pos[1] - (np.random.random() - 0.5) * pos_range[1],
                                                    ori_pos[0] - (np.random.random() - 0.5) * pos_range[0],
                                                    ori_pos[2] + (np.random.random() - 0.5) * pos_range[2]])
            self._physics.data.qpos[6:9] = self.obstacle_step_position
            self._physics.bind(self.obstacle_mocap).mocap_pos[:] = np.copy(self.obstacle_step_position[[1, 0, 2]])


        elif mode == "dynamic":
            # need be modified
            self._physics.data.qvel[6:9] = np.array([vel_range[0], vel_range[1], vel_range[2]])

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
            while distance_t_o <= self.obstacle_radiance:
                self.target_pose = np.array([self.target_original_pos[0] + pos_range[0] * np.random.random(),
                                             self.target_original_pos[1] + pos_range[1] * np.random.random(),
                                             self.target_original_pos[2] + pos_range[2] * np.random.random(),
                                             1, 0, 0, 0])
                self._target.set_mocap_pose(self._physics,
                                            position=self.target_pose[0:3],
                                            quaternion=self.target_pose[3:]
                                            )
        elif mode == "step_refresh_mode":
            self.target_pose = np.array([ori_pos[0],
                                         ori_pos[1],
                                         ori_pos[2],
                                         1, 0, 0, 0])
            self._target.set_mocap_pose(self._physics,
                                        position=self.target_pose[0:3],
                                        quaternion=self.target_pose[3:]
                                        )
        elif mode == "fixed_sample_mode":
            self.target_pose = np.array([self.target_original_pos[0] + pos_range[0] * 0.5,
                                         self.target_original_pos[1] + pos_range[1] * 0.5,
                                         self.target_original_pos[2] + pos_range[2] * 0.5,
                                         1, 0, 0, 0])
            self._target.set_mocap_pose(self._physics,
                                        position=self.target_pose[0:3],
                                        quaternion=self.target_pose[3:]
                                        )

        return self.target_pose
