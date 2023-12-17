import gymnasium as gym
import numpy as np
from torch import cuda
from core.env_wrappers import ResourceN, MujocoNSpeedWrapper
from policy import PolicyEMRL, PolicyDiffRL, PolicyClusterRL
from core.vec_env import VecEnvCustom
from config import Config


def run_gym(env_name, algorithm, cfg, seed=None, n_agents=100, n_policies=5, base_reward=False, device='cpu'):
    """
    :param env_name: "HalfCheetah-v4", "Walker2d-v4", "Ant-v4", "Hopper-v4", "resource-gathering-v1"
    :param algorithm: 'em', 'diff', 'cluster', 'oracle', 'random_even_clusters'
    :param seed: int or None
    :param n_agents: int
    :param n_policies: int, <= n_agents
    :param base_reward: bool
    :param device: 'cpu', 'cuda'
    """
    print(env_name, algorithm)
    print('seed', seed)
    print(f'{n_agents} agents, {n_policies} policies')
    task = 'resource' if env_name.startswith('resource') else 'speed'

    np.random.seed(seed)
    log_dir = f'runs/{env_name}/{task}/{algorithm}/{seed}/'
    assignment = None
    max_iters = 2000
    low, high = cfg.target_intervals[env_name]
    kwargs = {}
    if env_name.startswith('resource'):
        import mo_gymnasium
        import resource_gathering_modified
        make = mo_gymnasium.make
        wrapper = ResourceN
        if_render = True
    else:
        make = gym.make
        kwargs = {'ctrl_cost_weight': 0}
        if env_name.startswith('Ant'):
            max_iters = 2000
            kwargs['contact_cost_weight'] = 0
            kwargs['healthy_z_range'] = (0.4, 1)
        wrapper = MujocoNSpeedWrapper
        if_render = False
    env_fn = lambda: wrapper(make(env_name, **kwargs), n_agents, low, high, n_policies, base_reward, seed)
    cfg.env_fn = env_fn
    envs = [env_fn] * n_policies
    env_render = wrapper(make(env_name, render_mode="rgb_array"), n_agents, low, high, n_policies, base_reward, seed)
    n_agents = env_render.n_agents
    envs = VecEnvCustom(envs)
    reward_length = (envs.high - envs.low) / n_policies / 2
    # print(f'Targets:\n{envs.targets.round(2)}')

    print(f'device {device}')
    print(f'Expected centroids: {np.linspace(low+reward_length, high-reward_length, n_policies, endpoint=True)}')

    if algorithm == 'oracle':
        if task == 'speed':
            thresholds = np.linspace(low + reward_length*2, high, n_policies, endpoint=True)
            assignment = [[] for _ in range(n_policies)]
            assignment[0] = np.arange(n_agents)[envs.targets <= thresholds[0]].tolist()
            for i in range(1, n_policies):
                assignment[i] = np.arange(n_agents)[(thresholds[i-1] < envs.targets) &
                                                    (envs.targets <= thresholds[i])].tolist()
        else:
            assignment = envs.assignment
    elif algorithm == 'random_even_clusters':
        assignment = np.arange(n_agents)
        np.random.shuffle(assignment)
        assignment = assignment.reshape((n_policies, -1)).tolist()

    if algorithm == 'diff' and not base_reward:
        algorithm = PolicyDiffRL
    elif algorithm == 'cluster' and not base_reward:
        algorithm = PolicyClusterRL
    else:
        algorithm = PolicyEMRL

    policy = algorithm(envs, n_agents, n_policies, max_iters=max_iters, log_freq=50, cfg=cfg, device=device,
                       fixed_assignment=assignment, log_dir=log_dir, if_render=if_render, env_render=env_render, task=task)
    policy.train()


if __name__ == '__main__':
    env_name = 'HalfCheetah-v4'
    algorithm = 'diff'
    seed = 0
    n_agents = 100
    n_policies = 5
    base_reward = False
    device = 'cpu'  # 'cuda' if cuda.is_available() else 'cpu'
    cfg = Config()
    run_gym(env_name, algorithm, cfg, seed, n_agents, n_policies, base_reward, device)
