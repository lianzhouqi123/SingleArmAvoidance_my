import numpy as np
import math
from manipulator_mujoco.controllers import JointEffortController
from manipulator_mujoco.utils.mujoco_utils import (
    get_fullM,
)
from manipulator_mujoco.utils.transform_utils import (
    mat2quat,
    quat2mat
)
from manipulator_mujoco.utils.controller_utils import (
    pose_error,
)


class Rm63_Controller(JointEffortController):
    def __init__(
            self,
            physics: object,
            joints: object,
            eef_site: object,
            min_effort: np.ndarray,
            max_effort: np.ndarray,
            kp: float,
            damping_ratio,
    ) -> None:
        super().__init__(physics, joints, min_effort, max_effort)

        self._physics = physics
        self._joints = joints
        self._eef_site = eef_site
        self._min_effort = min_effort
        self._max_effort = max_effort
        self._kp = kp
        self._damping_ratio = damping_ratio
        self._kd = 2 * np.sqrt(self._kp) * self._damping_ratio
        self._jnt_dof_ids = self._physics.bind(joints).dofadr

    def run(self, target) -> None:
        M_full = get_fullM(
            self._physics.model.ptr,
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids, :][:, self._jnt_dof_ids]
        dq = self._physics.bind(self._joints).qvel
        ddq = self._physics.bind(self._joints).qacc
        target = np.array(target).squeeze(axis=0)
        torque = self._kp * np.dot(M, target) - self._kd * np.dot(M, dq)
        torque += self._physics.bind(self._joints).qfrc_bias
        torque = np.clip(torque, self._min_effort, self._max_effort)
        self._physics.bind(self._joints).qfrc_applied = torque


    def distance(self, target):
        """
        衡量末端位置与目标位置的笛卡尔空间差异
        input: target
        return: distance between target and end
        """
        target_pose = target
        ee_pos = self._physics.bind(self._eef_site).xpos
        ee_quat = mat2quat(self._physics.bind(self._eef_site).xmat.reshape(3, 3))
        ee_pose = np.concatenate([ee_pos, ee_quat])
        # Calculate the pose error (difference between the target and current pose).
        pose_err = pose_error(target_pose, ee_pose)
        # 只取前三个计算笛卡尔空间位置差异
        pose_err = pose_err[:3]
        return np.linalg.norm(pose_err)

    def get_end_dis(self, target):
        """
        target is a vector of 7D [x, y, z, qx, qy, qz, w]
        """
        # 取出目标位姿[qx, qy, qz, w]
        target_pos = target[3:]
        # 目标位姿转为旋转矩阵3X3
        target_pos = quat2mat(target_pos)
        ee_mat = self._physics.bind(self._eef_site).xmat.reshape(3, 3)
        ee_mat = ee_mat.T  # 末端R矩阵转置
        r3 = np.dot(ee_mat, target_pos)
        quatR3 = mat2quat(r3)
        dis_theta = 2 * math.acos(quatR3[0])
        return dis_theta
