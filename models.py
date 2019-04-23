"""
Note: Danijar mentioned switching ReLU to ELU improves performance: https://github.com/google-research/planet/issues/21#issuecomment-485200191
"""
import torch
from torch import nn
from torch.nn import functional as F


# TODO: Make version that can operate on entire sequences at once
class TransitionModel(nn.Module):
  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, min_std_dev=1e-5):
    super().__init__()
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, hidden_size)
    self.rnn = nn.GRUCell(hidden_size, hidden_size)
    self.fc_embed_belief_prior = nn.Linear(hidden_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(hidden_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
    self.register_buffer('initial_belief', torch.zeros(1, belief_size))
    self.register_buffer('initial_state', torch.zeros(1, state_size))

  def forward(self, prev_state, prev_action, prev_belief, observation=None):
    hidden = F.elu(self.fc_embed_state_action(torch.cat([prev_state, prev_action], dim=1)))
    belief = self.rnn(hidden, prev_belief)
    # Compute state prior by applying transition dynamics
    hidden = F.elu(self.fc_embed_belief_prior(belief))
    prior_mean, prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
    prior_std_dev = F.softplus(prior_std_dev) + self.min_std_dev
    prior_state = prior_mean + prior_std_dev * torch.randn_like(prior_mean)
    if observation is None:
      return belief, prior_state, prior_mean, prior_std_dev
    else:
      # Compute state posterior by applying transition dynamics and using current observation
      hidden = F.elu(self.fc_embed_belief_posterior(torch.cat([belief, observation], dim=1)))
      posterior_mean, posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
      posterior_std_dev = F.softplus(posterior_std_dev) + self.min_std_dev
      posterior_state = posterior_mean + posterior_std_dev * torch.randn_like(posterior_mean)
      return belief, prior_state, prior_mean, prior_std_dev, posterior_state, posterior_mean, posterior_std_dev


class SymbolicObservationModel(nn.Module):
  def __init__(self, observation_size, belief_size, state_size, embedding_size):
    super().__init__()
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)

  def forward(self, belief, state):
    hidden = F.elu(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = F.elu(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(nn.Module):
  def __init__(self, belief_size, state_size, embedding_size):
    super().__init__()
    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

  def forward(self, belief, state):
    hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = F.elu(self.conv1(hidden))
    hidden = F.elu(self.conv2(hidden))
    hidden = F.elu(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size)
  else:
    return VisualObservationModel(belief_size, state_size, embedding_size)


class RewardModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size):
    super().__init__()
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

  def forward(self, belief, state):
    hidden = F.elu(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = F.elu(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward


class SymbolicEncoder(nn.Module):
  def __init__(self, observation_size, embedding_size):
    super().__init__()
    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)

  def forward(self, observation):
    hidden = F.elu(self.fc1(observation))
    hidden = F.elu(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(nn.Module):
  def __init__(self, embedding_size):
    super().__init__()
    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

  def forward(self, observation):
    hidden = F.elu(self.conv1(observation))
    hidden = F.elu(self.conv2(hidden))
    hidden = F.elu(self.conv3(hidden))
    hidden = F.elu(self.conv4(hidden))
    hidden = hidden.view(-1, self.embedding_size)
    return hidden


def Encoder(symbolic, observation_size, embedding_size):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size)
  else:
    return VisualEncoder(embedding_size)
