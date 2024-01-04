import os
from manipulator_mujoco.obstacles.inv_pendulum import Inv_pendulum

_INV_PENDULUM_XML = os.path.join(
    os.path.dirname(__file__),
    '../assets/obstacles/inv_pendulum.xml'
)

_JOINTS = (
    'obstacle_joint1',
    'obstacle_joint2',
)

_ACTUATORS = (
    'obstacle_joint1_T',
    'obstacle_joint2_T',
)

_EEF_SITE = 'eef_site'

_ATTACHMENT_SITE = 'attachment_site'

class Obstacle(Inv_pendulum):
    def __init__(self, name: str = None):
        super().__init__(_INV_PENDULUM_XML, _EEF_SITE, _ATTACHMENT_SITE, _ACTUATORS, _JOINTS, name)
