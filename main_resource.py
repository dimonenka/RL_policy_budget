from core.main import run_gym
from config import Config


def run_resource(env_name, algorithm, cfg, seed=None, n_agents=100, n_policies=5, base_reward=False, device='cpu'):
    cfg.gamma = 1
    cfg.entropy_start = 5e-2
    cfg.entropy_end = 1e-3
    cfg.lr_start = 5e-4
    cfg.lr_end = 1e-4
    cfg.rms = False
    cfg.adam_eps = 1e-8
    run_gym(env_name, algorithm, cfg, seed, n_agents, n_policies, base_reward, device)


if __name__ == '__main__':
    env_name = "resource-gathering-v1"
    algorithm = 'em'
    seed = 42
    n_agents, n_policies = 25, 5
    base_reward = False
    device = 'cpu'
    cfg = Config()
    run_resource(env_name, algorithm, cfg, seed, n_agents, n_policies, base_reward, device)
