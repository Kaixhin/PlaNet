import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
from planner import MPCPlanner
from utils import plot


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=5000000, metavar='D', help='Experience replay size')  # TODO: Maybe increase size? Seems like the original just stores everything...
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=2000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=1, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0.1, metavar='βg', help='Global KL weight')
parser.add_argument('--free-nats', type=float, default=2, metavar='F', help='Free nats')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=12, metavar='H', help='Planning horizon distance')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--checkpoint-interval', type=int, default=25, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--load-checkpoint', type=int, default=0, metavar='E', help='Load model checkpoint (from given episode)')
parser.add_argument('--render', action='store_true', help='Render environment')
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


# Setup
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
# Initialise environment, experience replay memory and planner
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat)
D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.device)
planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates)


# Initialise dataset D with S random seed episodes
for s in range(args.seed_episodes):
  observation, done = env.reset(), False
  while not done:
    action = env.sample_random_action()
    next_observation, reward, done = env.step(action)
    D.append(observation, action, reward, done)
    observation = next_observation


# Initialise model parameters randomly
transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size).to(device=args.device)
encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size).to(device=args.device)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
if args.load_checkpoint > 0:
  model_dicts = torch.load(os.path.join('checkpoints', 'models_%d.pth' % args.load_checkpoint))
  transition_model.load_state_dict(model_dicts['transition_model'])
  observation_model.load_state_dict(model_dicts['observation_model'])
  reward_model.load_state_dict(model_dicts['reward_model'])
  encoder.load_state_dict(model_dicts['encoder'])
optimiser = optim.Adam(param_list, lr=args.learning_rate, eps=1e-4)
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, device=args.device)  # Allowed deviation in KL divergence


metrics = {'episodes': [], 'rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'global_kl_loss': []}
for episode in tqdm(range(args.seed_episodes + 1, args.episodes + 1), total=args.episodes, initial=args.seed_episodes + 1):
  metrics['episodes'].append(episode)
  # Model fitting
  for s in tqdm(range(args.collect_interval)):
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
    observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size + 1)  # Transitions start at time t = 0
    # Create initial belief and state for time t = 0
    init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(args.batch_size, args.state_size, device=args.device)
    # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], init_belief, nonterminals[:-1], bottle(encoder, (observations[1:], )))
    # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch, sum over time
    observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=1).sum()
    reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[1:], reduction='none').mean(dim=1).sum()
    kl_loss = args.overshooting_distance * torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=1).sum()
    global_kl_loss = args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=1).sum()
    # Calculate latent overshooting objective for t > 0
    if args.overshooting_kl_beta != 0:  
      for t in range(1, args.chunk_size):  # TODO: See if this is worth parallelising too
        d = min(t + args.overshooting_distance, args.chunk_size)  # Overshooting distance
        t_, d_ = t - 1, d - 1 # Use t_ and d_ to deal with different time indexing for latent states
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        _, _, prior_means, prior_std_devs = transition_model(prior_states[t_], actions[t:d], beliefs[t_], nonterminals[t:d])
        # Calculate and update KL loss
        kl_loss += args.overshooting_kl_beta * torch.max(kl_divergence(Normal(posterior_means[t_ + 1:d_ + 1].detach(), posterior_std_devs[t_ + 1:d_ + 1].detach()), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=1).sum()
    # Update model parameters
    optimiser.zero_grad()
    (observation_loss + reward_loss + kl_loss + global_kl_loss).backward()
    nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
    optimiser.step()
    # Update loss metrics
    metrics['observation_loss'].append(observation_loss.item())
    metrics['reward_loss'].append(reward_loss.item())
    metrics['kl_loss'].append(kl_loss.item())
    metrics['global_kl_loss'].append(global_kl_loss.item())
  # Update and plot loss metrics
  metrics['observation_loss'] = metrics['observation_loss'][:-args.collect_interval] + [sum(metrics['observation_loss'][-args.collect_interval:]) / args.collect_interval]
  metrics['reward_loss'] = metrics['reward_loss'][:-args.collect_interval] + [sum(metrics['reward_loss'][-args.collect_interval:]) / args.collect_interval]
  metrics['kl_loss'] = metrics['kl_loss'][:-args.collect_interval] + [sum(metrics['kl_loss'][-args.collect_interval:]) / args.collect_interval]
  metrics['global_kl_loss'] = metrics['global_kl_loss'][:-args.collect_interval] + [sum(metrics['global_kl_loss'][-args.collect_interval:]) / args.collect_interval]
  plot(metrics, 'observation_loss')
  plot(metrics, 'reward_loss')
  plot(metrics, 'kl_loss')
  plot(metrics, 'global_kl_loss')
  
  # Data collection
  with torch.no_grad():
    observation, total_reward = env.reset(), 0
    belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
    for t in range(args.max_episode_length // args.action_repeat):
      # Infer belief over current state q(s_t|o≤t,a<t) from the history
      belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, None, encoder(observation.to(device=args.device)).unsqueeze(dim=0))  # Action and observation have time dimension
      belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
      action = planner(belief, posterior_state, transition_model, reward_model)  # action ← planner(q(s_t|o≤t,a<t), p)
      action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
      next_observation, reward, done = env.step(action[0].cpu())  # Perform environment step (action repeats handled internally)
      total_reward += reward
      D.append(observation, action.cpu(), reward, done)
      if not args.symbolic_env:
        save_image(torch.cat([observation, observation_model(belief, posterior_state).cpu()]), os.path.join('results', 'obs_pred_%s.png' % str(t * args.action_repeat).zfill(len(str(args.max_episode_length)))))  # Save predicted observation via posterior
      observation = next_observation
      if args.render:
        env.render()
      if done:
        break
  # Update and plot reward metrics
  metrics['rewards'].append(total_reward)
  plot(metrics, 'rewards')
  # Save metrics
  torch.save(metrics, os.path.join('results', 'metrics.pth'))

  # Checkpoint models
  if episode % args.checkpoint_interval == 0:
    torch.save({'transition_model': transition_model.state_dict(), 'observation_model': observation_model.state_dict(), 'reward_model': reward_model.state_dict(), 'encoder': encoder.state_dict()}, os.path.join('checkpoints', 'models_%d.pth' % episode))

env.close()
