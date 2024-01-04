import math

from scipy.linalg import cho_factor, cho_solve

from manipulator_mujoco.controllers import JointEffortController

import numpy as np

from manipulator_mujoco.utils.controller_utils import (
    task_space_inertia_matrix,
    pose_error,
)

from manipulator_mujoco.utils.mujoco_utils import (
    get_site_jac, 
    get_fullM
)

from manipulator_mujoco.utils.transform_utils import (
    mat2quat,
    quat2mat
)
from manipulator_mujoco.utils.fkine import compute_pos

class OperationalSpaceController(JointEffortController):
    def __init__(
        self,
            physics: object,
            joints: object,
            eef_site: object,
            min_effort: np.ndarray,
            max_effort: np.ndarray,
            kp: float,
            ko: float,
            kv: float,
            vmax_xyz: float,
            vmax_abg: float,
    ) -> None:
        
        super().__init__(physics, joints, min_effort, max_effort)

        self._eef_site = eef_site
        self._kp = kp
        self._ko = ko
        self._kv = kv
        self._vmax_xyz = vmax_xyz
        self._vmax_abg = vmax_abg

        self._eef_id = self._physics.bind(eef_site).element_id
        self._jnt_dof_ids = self._physics.bind(joints).dofadr
        self._dof = len(self._jnt_dof_ids)

        self._task_space_gains = np.array([self._kp] * 3 + [self._ko] * 3)
        self._lamb = self._task_space_gains / self._kv
        self._sat_gain_xyz = vmax_xyz / self._kp * self._kv
        self._sat_gain_abg = vmax_abg / self._ko * self._kv
        self._scale_xyz = vmax_xyz / self._kp * self._kv
        self._scale_abg = vmax_abg / self._ko * self._kv

    def run(self, target):
        # 当前
        pose_err = target
        # joint_angle = self._physics.bind(self._joints).qpos#并为调用查询当前关节角度
        # step_angle = joint_angle + target
        # target_mat = compute_pos(step_angle).squeeze()
        # target_pos = np.array([target_mat[0, 3], target_mat[1, 3], target_mat[2, 3]])
        # target_quat = mat2quat(np.array([[target_mat[0, 0], target_mat[0, 1], target_mat[0, 2]],
        #                                 [target_mat[1, 0], target_mat[1, 1], target_mat[1, 2]],
        #                                  [target_mat[2, 0], target_mat[2, 1], target_mat[2, 2]]]))
        # #
        # target_pose = np.concatenate([target_pos, target_quat])

        # Get the Jacobian matrix for the end-effector.
        J = get_site_jac(
            self._physics.model.ptr,
            self._physics.data.ptr,
            self._eef_id,
        )
        J = J[:, self._jnt_dof_ids]

        # Get the mass matrix and its inverse for the controlled degrees of freedom (DOF) of the robot.
        M_full = get_fullM(
            self._physics.model.ptr,
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        Mx, M_inv = task_space_inertia_matrix(M, J)

        # Get the joint velocities for the controlled DOF.
        dq = self._physics.bind(self._joints).qvel

        # Get the end-effector position, orientation matrix, and twist (spatial velocity).
        # ee_pos = self._physics.bind(self._eef_site).xpos
        # ee_quat = mat2quat(self._physics.bind(self._eef_site).xmat.reshape(3, 3))
        # ee_pose = np.concatenate([ee_pos, ee_quat])

        # Calculate the pose error (difference between the target and current pose).
        # pose_err = pose_error(target_pose, ee_pose)

        # Initialize the task space control signal (desired end-effector motion).
        u_task = np.zeros(6)

        # Calculate the task space control signal.
        u_task += self._scale_signal_vel_limited(pose_err)

        # joint space control signal
        u = np.zeros(self._dof)

        # Add the task space control signal to the joint space control signal
        u += np.dot(J.T, np.dot(Mx, u_task))

        # Add damping to joint space control signal
        u += -self._kv * np.dot(M, dq)

        # Add gravity compensation to the target effort
        u += self._physics.bind(self._joints).qfrc_bias

        # send the target effort to the joint effort conatroller

        super().run(u)

    def distance(self, target):
        '''
        衡量末端位置与目标位置的笛卡尔空间差异
        input: target
        return: distance between target and end
        '''
        target_pose = target
        ee_pos = self._physics.bind(self._eef_site).xpos
        ee_quat = mat2quat(self._physics.bind(self._eef_site).xmat.reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        # Calculate the pose error (difference between the target and current pose).
        pose_err = pose_error(target_pose, ee_pose)
        #只取前三个计算笛卡尔空间位置差异
        pose_err = pose_err[:3]
        return np.linalg.norm(pose_err)

    def GetEndDIS(self, target):
        '''
        target is a vector of 7D [x, y, z, qx, qy, qz, w]
        '''
        # 取出目标位姿[qx, qy, qz, w]
        target_pos = target[3:]
        # 目标位姿转为旋转矩阵3X3
        target_pos = quat2mat(target_pos)
        ee_mat = self._physics.bind(self._eef_site).xmat.reshape(3, 3)
        ee_mat = ee_mat.T# 末端R矩阵转置
        R3 = np.dot(ee_mat, target_pos)
        quatR3 = mat2quat(R3)
        dis_theta = 2 * math.acos(quatR3[3])
        return dis_theta

    def _scale_signal_vel_limited(self, u_task: np.ndarray) -> np.ndarray:
        """
        Scale the control signal such that the arm isn't driven to move faster in position or orientation than the specified vmax values.

        Parameters:
            u_task (numpy.ndarray): The task space control signal.

        Returns:
            numpy.ndarray: The scaled task space control signal.
        """
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        scale = np.ones(6)
        if norm_xyz > self._sat_gain_xyz:
            scale[:3] *= self._scale_xyz / norm_xyz
        if norm_abg > self._sat_gain_abg:
            scale[3:] *= self._scale_abg / norm_abg

        return self._kv * scale * self._lamb * u_task

    def compute_accel(self, qpos_err, qvel_err):
        M_full = get_fullM(
            self._physics.model.ptr,
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        C = -self._physics.data.qfrc_bias.copy()
        C -= self._kp * qpos_err + self._kv * qvel_err
        q_accel = cho_solve(cho_factor(M, overwrite_a=True, check_finite=False), C, overwrite_b=True,
                            check_finite=False)
        return q_accel.squeeze()

    def compute_torque(self, pos_err, target_vel):
        ''' usage:
                torque = self.compute_torque(target, np.zeros(self._dof))
                self._physics.bind(self._joints).qfrc_applied = torque '''
        dt = self._physics.model.opt.timestep
        qvel = self._physics.bind(self._joints).qvel
        qvel_err = target_vel - qvel
        q_accel = self.compute_accel(pos_err, qvel_err)
        qvel_err += q_accel * dt
        torque = -self._kp * pos_err - self._kv * qvel_err
        return torque
