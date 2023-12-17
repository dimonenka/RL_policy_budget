# Personalized Reinforcement Learning with a Budget of Policies
**Dmitry Ivanov, Omer Ben-Porat**

*Personalization in machine learning (ML) tailors models' decisions to the individual characteristics of users. 
While this approach has seen success in areas like recommender systems, its expansion into high-stakes fields such as 
healthcare and autonomous driving is hindered by the extensive regulatory approval processes involved. To address this 
challenge, we propose a novel framework termed represented Markov Decision Processes (r-MDPs) that is designed 
to balance the need for personalization with the regulatory constraints. In an r-MDP, we cater to a diverse user 
population, each with unique preferences, through interaction with a small set of representative policies. 
Our objective is twofold: efficiently match each user to an appropriate representative policy and simultaneously 
optimize these policies to maximize overall social welfare. We develop two deep reinforcement learning algorithms 
that efficiently solve r-MDPs. These algorithms draw inspiration from the principles of classic K-means clustering
and are underpinned by robust theoretical foundations. Our empirical investigations, conducted across a variety of
simulated environments, showcase the algorithms' ability to facilitate meaningful personalization even under 
constrained policy budgets. Furthermore, they demonstrate scalability, efficiently adapting to larger policy budgets.*

### TLDR

To run one experiment, run `core/main.py` for MuJoCo 
or `main_resource.py` for Resource Gathering.
Change the specifics of the experiment under
`if __name__ == '__main__':` in those same files.
The logs are saved into `runs` folder and require TensorBoard
to inspect.

### The file structure

1) main_ray.py runs many (by default, 40) experiments in parallel. 
Specifically, it runs each of four algorithms for 10 seeds,
and repeats it for each environment.
These experiments are for specific `n` and `k`.

The algorithms are:
- `em`, which stands for our expectation-maximization algorithm,
- `diff`, which stands for our end-to-end algorithm,
- `random`, which trains policies given random even agent assignments,
- `cluster`, which is the clustering baseline.

The environments are: 
- `HalfCheetah-v4`, 
- `Walker2d-v4`, 
- `Ant-v4`, 
- `Hopper-v4`,
- `resource-gathering-v1` 
(which is our modification defined in `resource_gathering_modified/`).

2) `core/main.py` runs one experiment for specific `n`, `k`,
task, MuJoCo environment, algorithm, and seed. It contains a function
`run_gym` that is called in `main_ray.py` to run each experiment.
`main_resource.py` is the same but for Resource Gathering.

3) `policies/policy_gym.py` contain three classes, `PolicyEMRL`, 
`PolicyDiffRL`, and `PolicyClusterRL` that respectively implement 
our EM-like algorithm, our end-to-end algorithm, 
and the clustering baseline algorithm. 
These handle both policy and assignment updates,
as well as logging via TensorBoard.

4) The hyperparameters are mostly set in `config.py`.
Some are hardcoded in `policies/policy_gym.py` and in 
5) `core/main.py`, `main_resource.py`.
To vary which experiments to run, change directly
`main_ray.py` or `core/main.py` or `main_resource.py`.

6) `utils/` folder contains classes for all neural networks, as well as
a class for observation normalization.

7) `gym_main/env_wrappers.py` redefines MDPs as r-MDPs,
i.e., creates populations of agents.

8) `plot/` folder handles processing of TensorBoard logs
into tables and plots that we used for the paper. 
This relies on logs for particular `n` and `k` being stored in
f`runs/{n}_agents_{k}_policies*/`


## Citation

Will be added once AAAI 2024 proceedings are published.
