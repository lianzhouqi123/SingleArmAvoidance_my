from dm_control import mjcf
import numpy as np


class Primitive(object):  # option: geom or site
    """
    A base class representing a primitive object in a simulation environment.
    """

    def __init__(self, option: str = "geom", **kwargs):
        """
        Initialize the Primitive object.

        Args:
            **kwargs: Additional keyword arguments for configuring the primitive.
        """
        self._mjcf_model = mjcf.RootElement()
        if option == "geom":
            # Add a geometric element to the worldbody
            self._geom = self._mjcf_model.worldbody.add("geom", **kwargs)
        elif option == "site":
            # Add a site element to the worldbody
            self._site = self._mjcf_model.worldbody.add("site", **kwargs)

    @property
    def geom(self):
        """Returns the primitive's geom, e.g., to change color or friction."""
        return self._geom

    def site(self):
        """Returns the primitive's site."""
        return self._site

    @property
    def mjcf_model(self):
        """Returns the primitive's mjcf model."""
        return self._mjcf_model

    def add_actuator(self, joint):
        self._mjcf_model.add("actuator", name="actuator", joint=joint)
