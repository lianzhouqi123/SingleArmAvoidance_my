import numpy as np
import math as m
from manipulator_mujoco.utils.rot import rotx, roty, rotz


def compute_pos(joint):
    R1 = np.dot(rotz(180), rotz(joint[0]))
    R2 = np.dot(np.dot(rotz(90), roty(-90)), rotz(joint[1]))
    R3 = rotz(joint[2])
    R4 = np.dot(np.dot(roty(90), rotz(90)), rotz(joint[3]))
    R5 = np.dot(rotx(-90), rotz(joint[4]))
    R6 = np.dot(rotx(90), rotz(joint[5]))
    R7 = rotz(30.69)

    p1 = np.array([0, 0, 0.108])
    p2 = np.array([0.086, 0, 0.064])
    p3 = np.array([0.380, 0, 0])
    p4 = np.array([0.083, 0.086, 0])
    p5 = np.array([0, 0, 0.322])
    p6 = np.array([0, -0.1435, 0])
    p7 = np.array([0, 0, 0.210])

    T1 = np.hstack((R1, p1.reshape(-1, 1)))
    T2 = np.hstack((R2, p2.reshape(-1, 1)))
    T3 = np.hstack((R3, p3.reshape(-1, 1)))
    T4 = np.hstack((R4, p4.reshape(-1, 1)))
    T5 = np.hstack((R5, p5.reshape(-1, 1)))
    T6 = np.hstack((R6, p6.reshape(-1, 1)))
    T7 = np.hstack((R7, p7.reshape(-1, 1)))

    T1 = np.vstack((T1, np.array([0, 0, 0, 1])))
    T2 = np.vstack((T2, np.array([0, 0, 0, 1])))
    T3 = np.vstack((T3, np.array([0, 0, 0, 1])))
    T4 = np.vstack((T4, np.array([0, 0, 0, 1])))
    T5 = np.vstack((T5, np.array([0, 0, 0, 1])))
    T6 = np.vstack((T6, np.array([0, 0, 0, 1])))
    T7 = np.vstack((T7, np.array([0, 0, 0, 1])))

    D1 = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5), T6), T7)

    return D1


def compute_each_joint_sapcepos(joint):
    joint = np.degrees(joint)
    R1 = np.dot(rotz(180), rotz(joint[0]))
    R2 = np.dot(np.dot(rotz(90), roty(-90)), rotz(joint[1]))
    R3 = rotz(joint[2])
    R4 = np.dot(np.dot(roty(90), rotz(90)), rotz(joint[3]))
    R5 = np.dot(rotx(-90), rotz(joint[4]))
    R6 = np.dot(rotx(90), rotz(joint[5]))
    R7 = rotz(30.69)
    R8 = np.eye(3)
    R9 = np.eye(3)

    p1 = np.array([0, 0, 0.108])
    p2 = np.array([0.086, 0, 0.064])
    p3 = np.array([0.380, 0, 0])
    p4 = np.array([0.083, 0.086, 0])
    p5 = np.array([0, 0, 0.322])
    p6 = np.array([0, -0.1435, 0])
    p7 = np.array([0, 0, 0.120])
    p8 = np.array([0.160, 0, 0.090])
    p9 = np.array([-0.160, 0, 0.090])

    T1 = np.hstack((R1, p1.reshape(-1, 1)))
    T2 = np.hstack((R2, p2.reshape(-1, 1)))
    T3 = np.hstack((R3, p3.reshape(-1, 1)))
    T4 = np.hstack((R4, p4.reshape(-1, 1)))
    T5 = np.hstack((R5, p5.reshape(-1, 1)))
    T6 = np.hstack((R6, p6.reshape(-1, 1)))
    T7 = np.hstack((R7, p7.reshape(-1, 1)))
    T8 = np.hstack((R8, p8.reshape(-1, 1)))
    T9 = np.hstack((R9, p9.reshape(-1, 1)))

    T1 = np.vstack((T1, np.array([0, 0, 0, 1])))
    T2 = np.vstack((T2, np.array([0, 0, 0, 1])))
    T3 = np.vstack((T3, np.array([0, 0, 0, 1])))
    T4 = np.vstack((T4, np.array([0, 0, 0, 1])))
    T5 = np.vstack((T5, np.array([0, 0, 0, 1])))
    T6 = np.vstack((T6, np.array([0, 0, 0, 1])))
    T7 = np.vstack((T7, np.array([0, 0, 0, 1])))
    T8 = np.vstack((T8, np.array([0, 0, 0, 1])))
    T9 = np.vstack((T9, np.array([0, 0, 0, 1])))

    # compute the position of each joint
    D1 = T1
    D2 = np.dot(T1, T2)
    D3 = np.dot(D2, T3)
    D4 = np.dot(D3, T4)
    D5 = np.dot(D4, T5)
    D6 = np.dot(D5, T6)
    D7 = np.dot(D6, T7)
    D8 = np.dot(D6, T8)
    D9 = np.dot(D6, T9)

    p_q1 = D1[0:3, 3].reshape(1, -1)
    p_q2 = D2[0:3, 3].reshape(1, -1)
    p_q3 = D3[0:3, 3].reshape(1, -1)
    p_q4 = D4[0:3, 3].reshape(1, -1)
    p_q5 = D5[0:3, 3].reshape(1, -1)
    p_q6 = D6[0:3, 3].reshape(1, -1)
    p_q7 = D7[0:3, 3].reshape(1, -1)
    p_q8 = D8[0:3, 3].reshape(1, -1)
    p_q9 = D9[0:3, 3].reshape(1, -1)
    return p_q1, p_q2, p_q3, p_q4, p_q5, p_q6, p_q7, p_q8, p_q9


def arm_obstacle_distance_detection(joint, obstacle_pos, obstacle_radius, seg_num: list):
    """
    Purpose: compute the distance between the arm and the obstacle
    input: joint, obstacle_pos, obstacle_radius, seg_num
    return: flag, distance
    joint: each joint angle of the arm
    obstacle_pos: the position of the obstacle
    obstacle_radius: the radius of the obstacle
    seg_num: the number of the segment points of "each" link
    flag: if distance is less than the obstacle_radius, flag is True, else False
    distance: the distance among the obstacle and detection points of the arm
    """
    p_base = np.array([0, 0, 0])
    p_q1, p_q2, p_q3, p_q4, p_q5, p_q6, p_q7, p_q8, p_q9 = compute_each_joint_sapcepos(joint)
    link1_point = spaceline_segment(p_base, p_q1, seg_num[0])
    link2_point = spaceline_segment(p_q1, p_q2, seg_num[1])
    link3_point = spaceline_segment(p_q2, p_q3, seg_num[2])
    link4_point = spaceline_segment(p_q3, p_q4, seg_num[3])
    link5_point = spaceline_segment(p_q4, p_q5, seg_num[4])
    link6_point = spaceline_segment(p_q5, p_q6, seg_num[5])
    link7_point = spaceline_segment(p_q6, p_q7, seg_num[6])
    link8_point = spaceline_segment(p_q7, p_q8, seg_num[7])
    link9_point = spaceline_segment(p_q7, p_q9, seg_num[8])
    link_point = np.vstack((p_base, link1_point, p_q1,
                            link2_point, p_q2,
                            link3_point, p_q3,
                            link4_point, p_q4,
                            link5_point, p_q5,
                            link6_point, p_q6,
                            link7_point, p_q7,
                            link8_point, p_q8,
                            link9_point, p_q9))
    distance = np.zeros((1, link_point.shape[0]))
    for i in range(link_point.shape[0]):
        distance[0, i] = np.linalg.norm(link_point[i, :] - obstacle_pos)

    # test the distance between the obstacle and the arm
    if np.any(distance <= obstacle_radius):
        flag = True
    else:
        flag = False

    return flag, distance


def spaceline_segment(base_point: np.array, destination_point: np.array, seg_num: int) -> np.array:
    """
    Purpose: compute the segment point of the space line
    input: base_point, destination_point, seg_num
    return: segment_point (n, 3), n is the number of segment points in column(up to down)
                        [[x1, y1, z1],
                         [x2, y2, z2],
                         ...,
                         [xn, yn, zn]]
    base_point: the start point of the space line, unit: m
    destination_point: the end point of the space line, unit: m
    """
    destination_point = np.array(destination_point).reshape(1, -1)
    base_point = np.array(base_point).reshape(1, -1)
    x_part = np.zeros((1, seg_num))
    y_part = np.zeros((1, seg_num))
    z_part = np.zeros((1, seg_num))
    x_i = (destination_point[0, 0] - base_point[0, 0]) / (seg_num + 1)
    y_i = (destination_point[0, 1] - base_point[0, 1]) / (seg_num + 1)
    z_i = (destination_point[0, 2] - base_point[0, 2]) / (seg_num + 1)
    for i in range(seg_num):
        x_part[0, i] = base_point[0, 0] + i * x_i
        y_part[0, i] = base_point[0, 1] + i * y_i
        z_part[0, i] = base_point[0, 2] + i * z_i

    x_part = x_part.reshape(-1, 1)
    y_part = y_part.reshape(-1, 1)
    z_part = z_part.reshape(-1, 1)
    segment_point = np.hstack((x_part, y_part, z_part))

    return segment_point