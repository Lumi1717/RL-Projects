import gym
from gym import spaces
import pygame
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rbg_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size # size of the gride
        self.window_size = 516 # PyGame window

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(4) # the number of actions the agent came move
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0,1]),
            2: np.array([-1,0]),
            3: np.array([0,-1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._action_to_direction, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_locaiton = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_locaiton
        while np.array_equal(self._target_location, self._agent_locaiton):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()
        
        return observation, info
    
    def step(self, action):
        
