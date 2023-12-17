import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


ActorLinear = lambda inp_size, out_size: nn.Sequential(nn.Linear(inp_size, out_size), nn.Softmax(dim=-1))


class ActorDiscrete(nn.Module):
    def __init__(self, inp_size, hid_size, out_size, n_policies, n_hid_layers=0, device='cpu'):
        super(ActorDiscrete, self).__init__()
        layers = [layer_init(nn.Linear(inp_size, hid_size)), nn.Tanh()]
        for _ in range(n_hid_layers):
            layers.append(layer_init(nn.Linear(hid_size, hid_size)))
            layers.append(nn.Tanh())
        layers.append(layer_init(nn.Linear(hid_size, out_size * n_policies), std=0.01))
        self.fc = nn.Sequential(*layers)
        self.to(device)
        self.device = device
        self.n_policies = n_policies
        self.n_actions = out_size

    def forward(self, x, action=None, deterministic=False):
        """
        :param x: torch.FloatTensor with shape (batch_size, n_policies, inp_size)
        """
        logits = self.fc(x)
        logits = logits.view(x.shape[0], self.n_policies, self.n_policies, self.n_actions).diagonal(dim1=1, dim2=2).transpose(1, 2)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample() if not deterministic else logits.argmax(-1)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy().sum(-1)
        return action, log_probs, entropy


class ActorContinuous(nn.Module):
    eps = 1e-5

    def __init__(self, inp_size, hid_size, out_size, n_policies, mina, maxa, n_hid_layers=0, device='cpu'):
        super(ActorContinuous, self).__init__()
        layers = [layer_init(nn.Linear(inp_size, hid_size)), nn.Tanh()]
        for _ in range(n_hid_layers):
            layers.append(layer_init(nn.Linear(hid_size, hid_size)))
            layers.append(nn.Tanh())
        layers.append(layer_init(nn.Linear(hid_size, out_size * n_policies), std=0.01))
        self.fc = nn.Sequential(*layers)
        self.to(device)
        self.mina = torch.FloatTensor(np.array([mina])).to(device)
        self.maxa = torch.FloatTensor(np.array([maxa])).to(device)
        self.spread = (self.maxa - self.mina).view(1, 1, -1) / 2
        self.avg = (self.maxa + self.mina).view(1, 1, -1) / 2
        self.n_policies = n_policies
        self.n_actions = out_size
        self.device = device

        self.log_sigma = nn.Parameter(torch.zeros(1, 1, self.n_actions)) - 1

    def forward(self, x, action=None, deterministic=False):
        """
        :param x: torch.FloatTensor with shape (batch_size, n_policies, inp_size)
        """
        mu = self.fc(x)
        mu = mu.view(x.shape[0], self.n_policies, self.n_policies, self.n_actions).diagonal(dim1=1, dim2=2).transpose(1, 2)
        mu = torch.tanh(mu) * self.spread + self.avg
        sigma = self.log_sigma.exp()
        dist = Normal(mu, sigma)
        if action is None:
            action = dist.sample() if not deterministic else mu
        torch.clip_(action, self.mina, self.maxa)
        log_probs = dist.log_prob(action)
        log_probs = log_probs.sum(-1)  # sum over actions
        entropy = -log_probs.sum(-1)  # sum over policies
        return action, log_probs, entropy


class Critic(nn.Module):
    def __init__(self, inp_size, hid_size, n_agents, n_policies, n_hid_layers=0, device='cpu'):
        super(Critic, self).__init__()
        layers = [layer_init(nn.Linear(inp_size + n_policies, hid_size)), nn.Tanh()]
        for _ in range(n_hid_layers):
            layers.append(layer_init(nn.Linear(hid_size, hid_size)))
            layers.append(nn.Tanh())
        layers.append(layer_init(nn.Linear(hid_size, n_agents), std=1.0))
        self.fc = nn.Sequential(*layers)
        self.to(device)
        self.n_agents = n_agents
        self.n_policies = n_policies
        self.device = device

    def forward(self, state, policy_idx):
        policy_oh = F.one_hot(policy_idx, self.n_policies)
        x = torch.cat([state, policy_oh], dim=-1)
        values = self.fc(x)
        return values


class AssignmentNetwork(nn.Module):
    def __init__(self, n_agents, n_policies):
        super(AssignmentNetwork, self).__init__()
        self.n_agents = n_agents
        self.n_policies = n_policies
        self.assignment = nn.Parameter(torch.randn((self.n_policies, self.n_agents)))

    def forward(self):
        return F.softmax(self.assignment, dim=0)
