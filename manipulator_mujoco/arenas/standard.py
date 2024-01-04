from dm_control import mjcf
import numpy as np

class StandardArena(object):#场景初始化
    def __init__(self) -> None:
        """
        Initializes the StandardArena object by creating a new MJCF model and adding a checkerboard floor and lights.
        """
        self._mjcf_model = mjcf.RootElement()
        self._mjcf_model.option.timestep = 0.01
        self._mjcf_model.option.flag.warmstart = "enable"

        chequered = self._mjcf_model.asset.add(
            "texture",
            type="2d",
            builtin="checker",#'none', 'gradient', 'checker', 'flat'
            width=300,
            height=300,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.3, 0.4, 0.5],
        )
        self.grid = self._mjcf_model.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=[5, 5],
            reflectance=0.001,
        )
        self._mjcf_model.worldbody.add("geom", type="plane", size=[2, 2, 0.1], material=self.grid)
        #add a box obstacle
        # self._mjcf_model.worldbody.add("geom", type="sphere", size=[0.20/2, 0.28/2, 0.175/2], pos=[-0.300, 0, 0.175/2], material=self.grid)

        for x in [-2, 2]:
            self._mjcf_model.worldbody.add("light", pos=[x, -1, 3], dir=[-x, 1, -2])

    def attach(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model at a specified position and orientation.

        Args:
            child: The child element to attach.
            pos: The position of the child element.
            quat: The orientation of the child element.

        Returns:
            The frame of the attached child element.
        """
        frame = self._mjcf_model.attach(child)
        frame.pos = pos
        frame.quat = quat
        return frame

    def attach_axis(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0], joint_axis: list = [1, 0, 0]) -> [
        mjcf.Element, mjcf.Element]:
        """
        Attaches a child element to the MJCF model with a slide joint.

        Args:
            child: The child element to attach.

        Returns:
            The frame of the attached child element.
        """
        frame = self.attach(child)
        freejoint = frame.add('joint', type='slide', axis=joint_axis)  # type optimal: slide, hinge, ball
        frame.pos = pos
        frame.quat = quat
        return frame, freejoint

    def attach_free(self, child,  pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]) -> mjcf.Element:
        """
        Attaches a child element to the MJCF model with a free joint.

        Args:
            child: The child element to attach.

        Returns:
            The frame of the attached child element.
        """
        frame = self.attach(child)
        frame.add('freejoint')
        frame.pos = pos
        frame.quat = quat
        return frame
    
    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """
        Returns the MJCF model for the StandardArena object.

        Returns:
            The MJCF model.
        """
        return self._mjcf_model