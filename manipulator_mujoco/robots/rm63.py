import os
from manipulator_mujoco.robots.arm import Arm

_RM63_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/rm63/urdf/rm_63_6f.xml',
)

_JOINTS = (
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
)

_EEF_SITE = 'eef_site'

_ATTACHMENT_SITE = 'attachment_site'

_ACTUATORS = (
    'joint1_T',
    'joint2_T',
    'joint3_T',
    'joint4_T',
    'joint5_T',
    'joint6_T',
)

class rm63(Arm):
    def __init__(self, name: str = None):
        super().__init__(_RM63_XML, _EEF_SITE, _ATTACHMENT_SITE, _ACTUATORS, _JOINTS, name)