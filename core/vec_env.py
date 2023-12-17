from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np


class VecEnvCustom(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.buf_rews = np.zeros((self.num_envs, len(self.targets)), dtype=np.float32)

    @property
    def targets(self):
        return self.get_attr('targets')[0]

    @property
    def reward_length(self):
        return self.get_attr('reward_length')[0]

    @property
    def clusters(self):
        return self.get_attr('clusters')[0]

    @property
    def assignment(self):
        return self.get_attr('assignment')[0]

    @property
    def assignment_mask(self):
        return self.get_attr('assignment_mask')[0]

    @property
    def name(self):
        return self.get_attr('unwrapped')[0].spec.id

    @property
    def low(self):
        return self.get_attr('low')[0]

    @property
    def high(self):
        return self.get_attr('high')[0]

    def get_accuracy(self, assignment_mask):
        return self.env_method('get_accuracy', indices=[0], assignment_mask=assignment_mask)[0]
