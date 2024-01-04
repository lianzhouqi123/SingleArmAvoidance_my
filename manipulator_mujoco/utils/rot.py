import math as m
import numpy as np


def rotx(angle):
    angle = m.radians(angle)
    rotmat = np.array([[1, 0, 0],
                       [0, m.cos(angle), -m.sin(angle)],
                       [0, m.sin(angle), m.cos(angle)]])
    return rotmat


def roty(angle):
    angle = m.radians(angle)
    rotmat = np.array([[m.cos(angle), 0, m.sin(angle)],
                       [0, 1, 0],
                       [-m.sin(angle), 0, m.cos(angle)]])
    return rotmat


def rotz(angle):
    angle = m.radians(angle)
    rotmat = np.array([[m.cos(angle), -m.sin(angle), 0],
                       [m.sin(angle), m.cos(angle), 0],
                       [0, 0, 1]])
    return rotmat