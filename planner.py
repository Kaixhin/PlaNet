import torch
from torch.distributions import Normal


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
      # Evaluate J action sequences from the current belief (in batch)
      beliefs, states = [belief], [state]
      actions = action_distribution.sample([self.candidates]) # Sample actions
      # Sample next states
      for t in range(self.planning_horizon):
        next_belief, next_state, _, _ = transition_model(states[-1], actions[:, t], beliefs[-1])
        beliefs.append(next_belief)
        states.append(next_state)
      # Calculate expected returns (batched over time x batch)
      beliefs = torch.stack(beliefs[1:], dim=0).view(self.planning_horizon * self.candidates, -1)
      states = torch.stack(states[1:], dim=0).view(self.planning_horizon * self.candidates, -1)
      returns = reward_model(beliefs, states).view(self.planning_horizon, self.candidates).sum(dim=0)
      # Re-fit belief to the K best action sequences
      _, topk = returns.topk(self.top_candidates, largest=True, sorted=False)  # K ← argsort({R(j)}
      best_actions = actions[topk]
      # Update belief with new means and standard deviations
      action_distribution = Normal(best_actions.mean(dim=0), best_actions.std(dim=0, unbiased=False))
    # Return first action mean µ_t
    return action_distribution.mean[0].unsqueeze(dim=0)
