import torch
from torch.distributions import Normal
from models import bottle


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner():
  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates):
    self.action_size = action_size
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  def __call__(self, belief, state, transition_model, reward_model):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ← N(0, I)
    action_distribution = Normal(torch.zeros(self.planning_horizon, B, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, self.action_size, device=belief.device))
    for i in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      actions = action_distribution.sample([self.candidates]).permute(1, 2, 0, 3).reshape(self.planning_horizon, B * self.candidates, -1)  # Sample actions (and flip to time x (batch x candidates) x actions)
      # Sample next states
      beliefs, states, _, _ = transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = bottle(reward_model, (beliefs, states)).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, -1)
      # Update belief with new means and standard deviations
      action_distribution = Normal(best_actions.mean(dim=2), best_actions.std(dim=2, unbiased=False))
    # Return first action mean µ_t
    return action_distribution.mean[0]
