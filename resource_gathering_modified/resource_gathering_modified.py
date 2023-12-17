from mo_gymnasium.envs.resource_gathering.resource_gathering import ResourceGathering
from typing import Optional
import numpy as np


class ResourceGatheringModified(ResourceGathering):
    def __init__(self, render_mode: Optional[str] = None):
        super(ResourceGatheringModified, self).__init__(render_mode=render_mode)

        # The map of resource gathering env
        self.map = np.array(
            [
                [" ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " "],
                [" ", " ", "H", " ", " "],
            ]
        )
