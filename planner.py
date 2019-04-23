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
    belief, state = belief.expand(self.candidates, -1), state.expand(self.candidates, -1)
    # Initialize factorized belief over action sequences q(a_t:t+H) ← N(0, I)
    action_distribution = Normal(torch.zeros(self.planning_horizon, self.action_size, device=belief.device), torch.ones(self.planning_horizon, self.action_size, device=belief.device))
    for i in range(self.optimisation_iters):
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      actions = action_distribution.sample([self.candidates]).transpose(0, 1) # Sample actions (and flip to time x batch)
      # Sample next states
      beliefs, states, _, _ = transition_model(state, actions, belief)
      # Calculate expected returns (technically sum of rewards over plannign horizon)
      returns = bottle(reward_model, (beliefs, states)).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.topk(self.top_candidates, largest=True, sorted=False)  # K ← argsort({R(j)}
      best_actions = actions[:, topk]
      # Update belief with new means and standard deviations
      action_distribution = Normal(best_actions.mean(dim=1), best_actions.std(dim=1, unbiased=False))
    # Return first action mean µ_t
    return action_distribution.mean[0].unsqueeze(dim=0)
