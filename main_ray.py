from core.main import run_gym
from main_resource import run_resource
import ray
from config import Config


@ray.remote(num_cpus=1, num_gpus=0)
def run_ray(env_name, algorithm, seed=None, n_agents=100, n_policies=5):
    cfg = Config()
    if env_name.startswith('resource'):
        run_resource(env_name, algorithm, cfg, seed, n_agents, n_policies, base_reward=False, device='cpu')
    else:
        run_gym(env_name, algorithm, cfg, seed, n_agents, n_policies, base_reward=False, device='cpu')


if __name__ == '__main__':
    env_names = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4', 'resource-gathering-v1']
    algorithms = ['em', 'diff', 'cluster', 'random_even_clusters']
    seeds = range(10)
    n_agents, n_policies = 100, 5

    ray.init()
    for env_name in env_names:
        result_ids = []
        for algorithm in algorithms:
            for seed in seeds:
                result_ids.append(run_ray.remote(env_name, algorithm, seed, n_agents, n_policies))
        ray.get(result_ids)
