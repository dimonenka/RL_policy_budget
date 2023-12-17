import numpy as np
from gymnasium.spaces import Discrete, Box
from plot.render import save_frames_as_gif, clusters_histogram, draw_paths
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.networks import ActorDiscrete, ActorContinuous, Critic, AssignmentNetwork
from utils.running_mean_std import RunningMeanStd
from core.vec_env import VecEnvCustom


class PolicyEMRL:
    max_kl = 0.05
    max_grad_norm = 0.5

    def __init__(self, envs, n_agents, n_policies, max_iters, log_freq, cfg, device='cpu',
                 fixed_assignment=None, log_dir=None, if_render=False, env_render=None, task='speed'):
        self.set_name()
        self.task = task
        if isinstance(envs.action_space, Discrete):
            self.discrete = True
            self.n_actions = envs.action_space.n
        elif isinstance(envs.action_space, Box):
            self.discrete = False
            self.n_actions = envs.action_space.shape[0]
        else:
            raise NotImplementedError('Action space is expected to be Discrete or Box')
        self.cfg = cfg
        self.inp_size = envs.observation_space.shape[0]
        self.hid_actor = cfg.hid_actor
        self.hid_critic = cfg.hid_critic
        self.n_hid_layers = cfg.n_hid_layers
        self.lr_start = cfg.lr_start
        self.lr = cfg.lr_start
        self.lr_end = cfg.lr_end
        self.lr_mult = (cfg.lr_end / cfg.lr_start) ** (1.1 / max_iters)
        self.gamma = cfg.gamma
        self.device = device
        self.locs = np.zeros(n_policies)
        self.return_type = cfg.return_type
        self.mina, self.maxa = None, None
        if not self.discrete:
            self.mina = np.array(envs.action_space.low)
            self.maxa = np.array(envs.action_space.high)
        self.warm_start_iters = cfg.warm_start_iters
        self.mixing_coef = cfg.mixing_coef

        self.env = envs
        self.n_policies = n_policies
        self.n_agents = n_agents
        self.fixed_assignment = fixed_assignment
        self.SW = 0
        self.entropy_start = cfg.entropy_start
        self.entropy = self.entropy_start
        self.entropy_end = cfg.entropy_end
        self.entropy_mult = (self.entropy_end / self.entropy_start) ** (1.1 / max_iters) if self.entropy_start > 0 else 0
        self.batch_size = cfg.batch_size
        self.n_batches = cfg.n_batches
        self.iteration = 0
        self.max_iters = max_iters
        self.log_freq = log_freq

        assert 1 <= self.n_policies <= self.n_agents, f'set 1 <= n_policies <= {self.n_agents}'
        self.init_policies()

        self.last_rewards = np.zeros((self.n_policies, self.n_agents), dtype=float)
        self.last_rewards_buffer = self.last_rewards.copy()
        self.q_table = self.last_rewards.copy()
        self.last_states = self.env.reset()

        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

        self.ppo_update = cfg.ppo_update
        self.ppo_epochs = cfg.ppo_epochs
        self.ppo_eps = cfg.ppo_eps
        self.approx_kl = 0
        self.gae_lam = cfg.gae_lam

        self.if_render = if_render and (env_render is not None)
        self.env_render = env_render

        self.rms = cfg.rms
        if self.rms:
            self.state_rms = RunningMeanStd(shape=(self.n_policies, self.inp_size))
        self.n_flips = 0
        self.loss_actor_log, self.loss_critic_log = 0, 0

    def set_name(self):
        self.name = 'EM-PPO'

    def init_policies(self):
        device = self.device
        if self.discrete:
            self.policy = ActorDiscrete(self.inp_size, self.hid_actor, self.n_actions, self.n_policies,
                                        self.n_hid_layers, device)
        else:
            self.policy = ActorContinuous(self.inp_size, self.hid_actor, self.n_actions, self.n_policies,
                                          self.mina, self.maxa, self.n_hid_layers, device)
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), self.lr_start, eps=self.cfg.adam_eps)
        self.critic = Critic(self.inp_size, self.hid_critic, self.n_agents, self.n_policies, self.n_hid_layers, device)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), self.lr_start, eps=self.cfg.adam_eps)

        self.init_assignment() if self.fixed_assignment is None else self.init_fixed_assignment()

    def init_assignment(self):
        self.assignment = [list(range(k, self.n_agents, self.n_policies)) for k in range(self.n_policies)]
        self.old_assignment = self.assignment.copy()

    def init_fixed_assignment(self):
        self.assignment = self.fixed_assignment
        self.old_assignment = self.assignment.copy()

    def reset_assignment(self):
        return [[] for _ in range(self.n_policies)]

    def process_state(self, state):
        if self.rms:
            state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-5), -5, 5)
        return torch.FloatTensor(state).unsqueeze(0)

    def warm_start(self):
        for _ in range(self.warm_start_iters):
            self.M_step()
            self.iteration += 1
        self.E_step()
        print('Warm-started policy:')
        self.log()

    def get_assignment_mask(self):
        assignment_mask = torch.zeros((self.n_policies, self.n_agents)).long()
        for policy_idx in range(self.n_policies):
            agents = self.assignment[policy_idx]
            assignment_mask[policy_idx, agents] = 1
        return assignment_mask

    def M_step(self):
        loss_actor_log, loss_critic_log = 0, 0
        n_batches, batch_size = self.n_batches, self.batch_size
        if self.ppo_update:
            n_batches = 1
            batch_size = self.n_batches * self.batch_size
        for _ in range(n_batches):
            assignment_mask = self.get_assignment_mask()
            state_batch, action_batch, reward_batch, done_batch, log_prob_batch, entropy_batch, loc_batch = [], [], [], [], [], [], []

            for _ in range(batch_size):
                states = self.last_states
                states = self.process_state(states)

                actions, log_probs, entropies = self.policy(states.to(self.policy.device))
                actions, log_probs, entropies = actions.cpu(), log_probs.cpu(), entropies.cpu()

                self.env.step_async(actions.squeeze(0).detach().numpy())

                state_batch.append(states)
                action_batch.append(actions)
                log_prob_batch.append(log_probs)
                entropy_batch.append(entropies)

                self.last_states, rewards, dones, infos = self.env.step_wait()

                reward_batch.append(rewards)
                done_batch.append(dones)
                loc_batch.append([info['loc'] for info in infos])

                self.last_rewards_buffer += rewards
                if any(dones):
                    for i, d in enumerate(dones):
                        if d:
                            self.last_rewards[i] = self.last_rewards_buffer[i].copy()
                            self.last_rewards_buffer[i] *= 0
                    SW = (self.last_rewards * assignment_mask.detach().numpy()).sum().item()
                    self.SW = 0.99 * self.SW + 0.01 * SW

            states = torch.cat(state_batch)  # (batch_size, n_policies, state_size)
            actions = torch.cat(action_batch).detach()  # (batch_size, n_policies) if self.discrete else (batch_size, n_policies, n_actions)
            rewards = torch.FloatTensor(np.array(reward_batch))  # (batch_size, n_policies, n_agents)
            dones = torch.LongTensor(np.array(done_batch)).unsqueeze(-1)  # (batch_size, n_policies, 1)
            log_probs = torch.cat(log_prob_batch)  # (batch_size, n_policies)
            entropies = torch.cat(entropy_batch)  # (batch_size,)

            values = self.get_values(states)  # (batch_size, n_policies, n_agents)

            returns = self.get_returns(rewards, dones, values.detach())  # (batch_size-1, n_policies, n_agents)
            log_probs, values = log_probs[:returns.shape[0]], values[:returns.shape[0]]
            states, actions = states[:returns.shape[0]], actions[:returns.shape[0]]

            TD = self.get_TD(returns, values)  # (batch_size-1, n_policies, n_agents)
            advantages = self.get_advantages(TD, assignment_mask)  # (batch_size-1, n_policies)

            if not self.ppo_update:
                loss_actor = self.loss_actor(log_probs, advantages)
                loss_actor += self.loss_entropy(entropies)
                loss_critic = self.loss_critic(TD)

                self.update_actor(loss_actor)
                self.update_critic(loss_critic)
                loss_actor_log += loss_actor.item() / self.n_batches
                loss_critic_log += loss_critic.item() / self.n_batches
            else:
                b_inds = np.arange(batch_size-1)
                for epoch in range(self.ppo_epochs):
                    np.random.shuffle(b_inds)
                    for b in range(self.n_batches):
                        mb_inds = b_inds[b::self.n_batches]
                        state_batch = states[mb_inds]
                        action_batch = actions[mb_inds]
                        return_batch = returns[mb_inds]
                        old_log_prob_batch = log_probs[mb_inds].detach()
                        old_value_batch = values[mb_inds].detach()
                        old_TD_batch = TD[mb_inds]

                        assignment_mask = self.get_assignment_mask()
                        advantage_batch = self.get_advantages(old_TD_batch, assignment_mask)

                        new_value_batch = self.get_values(state_batch)
                        TD_batch = self.get_TD(return_batch, new_value_batch)

                        _, new_log_prob_batch, entropy_batch = self.policy(state_batch, action_batch)

                        loss_actor = self.loss_actor_ppo(old_log_prob_batch, new_log_prob_batch, advantage_batch)
                        loss_actor += self.loss_entropy(entropy_batch)
                        loss_critic = self.loss_critic(TD_batch)
                        # loss_critic = self.loss_critic_ppo(old_value_batch, new_value_batch, return_batch)

                        self.update_actor(loss_actor)
                        self.update_critic(loss_critic)
                        loss_actor_log += loss_actor.item() / self.n_batches / self.ppo_epochs
                        loss_critic_log += loss_critic.item() / self.n_batches / self.ppo_epochs
                    if self.approx_kl > self.max_kl: break

            locs = np.array(loc_batch).mean(0)
            self.locs = 0.99 * self.locs + 0.01 * locs
            if self.rms:
                self.state_rms.update(states.numpy())

        self.loss_actor_log = loss_actor_log
        self.loss_critic_log = loss_critic_log

        self.schedule()
        return self.SW

    def E_step(self):
        if self.fixed_assignment is None:
            # assign each agent to the best policy, s.t. each policy has at least one agent
            new_assignment = self.reset_assignment()
            assigned_agents = set()
            utilities = np.zeros((self.n_policies, self.n_agents), dtype=float)
            for policy_idx in range(self.n_policies):
                utilities[policy_idx] = self.evaluate_policy(policy_idx, list(range(self.n_agents)), False)
            for current_policy, agents in enumerate(self.assignment):
                for agent in agents:
                    if len(set(agents) - assigned_agents) > 1 or current_policy >= self.n_policies:
                        best_policy = utilities[:, agent].argmax()
                    else:
                        best_policy = current_policy
                    new_assignment[best_policy].append(agent)
                    assigned_agents.add(agent)
            self.assignment = new_assignment

    def train(self):
        if self.cfg.train:
            self.warm_start()
            print(f'{self.name} in progress...')
            while self.iteration <= self.max_iters:
                self.M_step()
                self.E_step()
                if (self.iteration + 1) % self.log_freq == 0:
                    self.log()
                self.iteration += 1
            torch.save(self.policy.state_dict(), self.log_dir + "policy.pth")
        else:
            self.policy.load_state_dict(torch.load(self.log_dir + "policy.pth"))
        self.render()

    def get_values(self, states):
        policy_idx = torch.arange(self.n_policies).to(self.critic.device).view(1, -1)  # (1, n_policies)
        policy_idx = policy_idx.repeat((states.shape[0], 1))  # (batch_size, n_policies)
        values = self.critic(states.to(self.critic.device), policy_idx).to('cpu')  # (batch_size, n_policies, n_agents)
        return values

    def get_returns(self, rewards, dones, values):
        # throws away the last time_step
        if self.return_type == 'mc':
            returns = []
            R = values[-1]
            for r, d in zip(rewards[:-1].flip(0), dones[:-1].flip(0)):
                # calculate the discounted value
                R = r + self.gamma * R * (1 - d)
                returns.append(R)
            returns = torch.stack(returns).flip(0)
        elif self.return_type == 'td':
            returns = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1])
        elif self.return_type == 'gae':
            values, values_next, rewards, dones = values[:-1], values[1:], rewards[:-1], dones[:-1]
            td_target = rewards + self.gamma * values_next * (1 - dones)
            td = td_target - values
            advantages = []
            A = 0.0
            for idx in reversed(range(td.shape[0])):
                A = self.gamma * self.gae_lam * A * (1 - dones[idx]) + td[idx]
                advantages.append(A)
            advantages = torch.stack(advantages).flip(0)
            returns = values + advantages
        else:
            raise NotImplementedError('set `return_type` to either `mc`, `td`, or `gae`')
        return returns

    def get_TD(self, returns, values):
        return returns - values

    def get_advantages(self, TD, mask):
        advantages = (TD.detach() * mask.unsqueeze(0)).mean(-1)
        # advantage normalization is necessary for each policy to have equal contributions to updates of shared weights
        advantages = (advantages - advantages.mean(0, keepdim=True)) / (advantages.std(0, keepdim=True) + 0.01)
        return advantages

    def loss_actor(self, log_probs, advantages):
        return -(log_probs * advantages).sum(-1).mean()

    def loss_actor_ppo(self, old_log_probs, new_log_probs, advantages):
        ratios = torch.exp(new_log_probs - old_log_probs)
        self.log_approx_kl(ratios)
        L1 = ratios * advantages
        L2 = torch.clamp(ratios, 1 - self.ppo_eps, 1 + self.ppo_eps) * advantages
        return -(torch.min(L1, L2)).sum(-1).mean()

    def log_approx_kl(self, ratios):
        approx_kl = ((ratios - 1) - torch.log(ratios + 1e-6)).mean().item()
        self.approx_kl = 0.95 * self.approx_kl + 0.05 * approx_kl
        return approx_kl

    def loss_critic(self, TD):
        return (TD ** 2).sum(1).mean()

    def loss_critic_ppo(self, old_values, new_values, returns):
        old_value_clipped = old_values + (new_values - old_values).clamp(-self.ppo_eps, self.ppo_eps)
        value_loss = (old_values - returns).pow(2)
        value_loss_clipped = (old_value_clipped - returns).pow(2)
        critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).sum(1).mean()
        return critic_loss

    def loss_entropy(self, entropy):
        return -self.entropy * entropy.mean()

    def update_actor(self, loss_actor):
        self.opt_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.opt_actor.step()

    def update_critic(self, loss_critic):
        self.opt_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()

    def schedule(self):
        self.q_table = self.q_table * self.mixing_coef + self.last_rewards * (1 - self.mixing_coef)
        self.lr = max(self.lr * self.lr_mult, self.lr_end)
        self.entropy = max(self.entropy * self.entropy_mult, self.entropy_end)
        for g in self.opt_actor.param_groups:
            g['lr'] = self.lr
        for g in self.opt_critic.param_groups:
            g['lr'] = self.lr

    def evaluate_policy(self, policy_idx, agents, aggregate=True):
        reward = self.q_table[policy_idx, agents]
        if aggregate:
            reward = reward.sum()
        return reward

    def get_cluster_statistics(self):
        cluster_sizes = [len(x) for x in self.assignment]
        cluster_centroids = [self.env.targets[a].mean().round(2) for a in self.assignment]
        return cluster_sizes, cluster_centroids

    def update_n_flips(self):
        n_flips = 0
        for i in range(self.n_policies):
            n_flips += len(set(self.assignment[i]).symmetric_difference(set(self.old_assignment[i])))
        self.n_flips = n_flips // 2
        self.old_assignment = self.assignment.copy()

    def log(self):
        print()
        cluster_sizes, cluster_centroids = self.get_cluster_statistics()
        self.update_n_flips()
        policy_locations = np.round(self.locs, 2).tolist()
        print(f'{self.name} iteration {self.iteration + 1}')
        print(f'Social Welfare = {round(self.SW / self.n_agents, 3)}')
        print('Cluster sizes: ', cluster_sizes)
        print(f"Cluster centroids: {cluster_centroids}")
        print(f"Policy  locations: {policy_locations}")
        print(f"Number of flips: {self.n_flips}")
        print(f"KL divergence: {round(self.approx_kl, 3)}")

        self.writer.add_scalar("social_welfare", self.SW / self.n_agents, self.iteration + 1)

        self.writer.add_scalar("KL_divergence", self.approx_kl, self.iteration + 1)
        self.writer.add_scalar("n_flips", self.n_flips, self.iteration + 1)

        self.writer.add_scalar("loss_actor", self.loss_actor_log, self.iteration + 1)
        self.writer.add_scalar("loss_critic", self.loss_critic_log, self.iteration + 1)

    def render(self):
        with torch.no_grad():
            if self.cfg.train and self.task == 'speed':
                clusters_histogram(
                    assignment_mask=self.get_assignment_mask().numpy(),
                    targets=self.env.targets, low=self.env.low, high=self.env.high,
                    path=self.log_dir + 'images/', n_ticks=min(self.n_policies+1, 6)
                )

            if not self.if_render:
                return

            verts = [[] for _ in range(self.n_policies)]
            for i in range(self.n_policies):
                state = self.env_render.reset()
                done = False
                # frames = []
                while not done:
                    # frames.append(self.env_render.render())
                    verts[i].append(state[:2])
                    state = self.process_state(np.array([state] * self.n_policies))
                    action, _, _ = self.policy(state.to(self.policy.device), deterministic=True)
                    action = action.cpu().squeeze(0)[i].item()
                    state, _, done, _ = self.env_render.step(action)
                verts[i].append(state[:2])
                # save_frames_as_gif(frames, path=self.log_dir + 'images/', filename=f'policy_{i}.gif')

            if self.task == 'resource':
                self.env_render.reset()
                frame = self.env_render.render()
                draw_paths(frame, verts, path=self.log_dir + 'images/', filename='paths.png')

            self.env_render.close()

    def sample_rewards(self, n_transitions=10000):
        state = self.env.reset()
        rewards = []
        for i in range(n_transitions):
            state = self.process_state(state)
            action, _, _ = self.policy(state)
            self.env.step_async(action.squeeze(0).detach().numpy())
            state, reward, _, _ = self.env.step_wait()
            rewards.append(reward)
        return rewards


class PolicyDiffRL(PolicyEMRL):
    def set_name(self):
        self.name = 'Diff-PPO'

    def init_assignment(self):
        self.assignment = np.array([])
        self.assignment_network = AssignmentNetwork(self.n_agents, self.n_policies)
        self.opt_assignment = torch.optim.Adam(self.assignment_network.parameters(), self.cfg.lr_assignment)
        self.old_assignment = self.assignment_network().detach().numpy()

    def warm_start(self):
        pass

    def E_step(self):
        pass

    def get_assignment_mask(self):
        return self.assignment_network()

    def update_actor(self, loss_actor):
        self.opt_actor.zero_grad()
        self.opt_assignment.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.opt_actor.step()
        self.opt_assignment.step()

    def get_cluster_statistics(self):
        assignment_mask = self.get_assignment_mask().detach().numpy()
        cluster_sizes = assignment_mask.sum(-1)
        targets = self.env.targets
        if len(targets.shape) == 2:
            targets = targets.mean(1)
        cluster_centroids = ((targets.reshape(1, -1) * assignment_mask).sum(-1) / cluster_sizes)
        # print(assignment_mask.T.round(2))
        return cluster_sizes.round(1), cluster_centroids.round(2)

    def update_n_flips(self):
        new_assignment = self.get_assignment_mask().detach().numpy()
        self.n_flips = round(np.abs(new_assignment - self.old_assignment).sum() / 2, 1)
        self.old_assignment = new_assignment


class PolicyClusterRL(PolicyEMRL):
    def set_name(self):
        self.name = 'Cluster-PPO'

    def warm_start(self):
        pass

    def E_step(self):
        pass

    def get_fixed_assignment(self):
        print(f'\n{self.name} clustering...')
        rewards = self.pretrainer.sample_rewards()
        rewards = np.concatenate(rewards).T

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_policies)
        km.fit(rewards)

        assignment = [[] for _ in range(self.n_policies)]
        for i, l in enumerate(km.labels_):
            assignment[l].append(i)
        self.fixed_assignment = assignment

        self.init_fixed_assignment()

    def pretrain_single_policy(self):
        print(f'\n{self.name} pretraining...')
        env = VecEnvCustom([self.cfg.env_fn])
        self.pretrainer = PolicyEMRL(env, self.n_agents, 1, self.max_iters//2, self.log_freq, self.cfg,
                                self.device, None, self.log_dir + 'pretraining/', False, None, self.task)
        self.pretrainer.train()

    def train(self):
        if self.n_policies > 1 and self.fixed_assignment is None:
            self.pretrain_single_policy()
            self.get_fixed_assignment()
        super(PolicyClusterRL, self).train()
