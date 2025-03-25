# train_ppo.py

import gym
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, state_dim, nb_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.actor = nn.Linear(64, nb_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = x.float()
        features = self.fc(x)
        return self.actor(features), self.critic(features)

class Environments():
    def __init__(self, nb_actor, device):
        self.device = device
        self.envs = [self.get_env() for _ in range(nb_actor)]
        self.observations = [None for _ in range(nb_actor)]
        self.done = [False for _ in range(nb_actor)]
        self.total_rewards = [0 for _ in range(nb_actor)]
        self.nb_actor = nb_actor

        for env_id in range(nb_actor):
            self.reset_env(env_id)

    def len(self):
        return self.nb_actor

    def reset_env(self, env_id):
        self.total_rewards[env_id] = 0
        obs, _ = self.envs[env_id].reset()
        self.observations[env_id] = obs
        self.done[env_id] = False

    def step(self, env_id, action):
        next_obs, reward, terminated, truncated, _ = self.envs[env_id].step(action)
        done = terminated or truncated
        
        self.done[env_id] = done
        self.total_rewards[env_id] += reward
        self.observations[env_id] = next_obs
        
        return next_obs, reward, terminated, done

    def get_env(self):
        env = gym.make('CartPole-v1')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    def render(self, env_id=0):
        self.envs[env_id].render()

def PPO(envs, actorcritic, T=128, K=3, batch_size=256, gamma=0.99, 
        gae_lambda=0.95, vf_coeff=0.5, ent_coeff=0.01, nb_iterations=500, lr=3e-4):
    device = next(actorcritic.parameters()).device
    optimizer = torch.optim.Adam(actorcritic.parameters(), lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.0, total_iters=nb_iterations)
    
    rewards_history = []
    
    for iteration in tqdm(range(nb_iterations)):
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        # Collect trajectories
        for _ in range(T):
            for env_id in range(envs.len()):
                if envs.done[env_id]:
                    envs.reset_env(env_id)
                
                obs = torch.from_numpy(envs.observations[env_id]).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits, value = actorcritic(obs)
                    probs = torch.distributions.Categorical(logits=logits)
                    action = probs.sample()
                    log_prob = probs.log_prob(action)
                
                next_obs, reward, terminated, done = envs.step(env_id, action.item())
                
                states.append(obs)
                actions.append(action)
                rewards.append(torch.tensor([reward], device=device))
                dones.append(torch.tensor([done], dtype=torch.float32, device=device))
                values.append(value)
                log_probs.append(log_prob)
        
        # Convert to tensors
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        
        # Compute advantages and returns
        advantages = torch.zeros_like(rewards, device=device)
        returns = torch.zeros_like(rewards, device=device)
        last_gae = 0
        
        with torch.no_grad():
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[-1]
                    next_value = values[-1]
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    next_value = values[t+1]
                
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_gae
                last_gae = advantages[t]
                returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy
        for _ in range(K):
            indices = torch.randperm(len(states), device=device)
            for i in range(0, len(states), batch_size):
                idx = indices[i:i+batch_size]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Get new policy and values
                logits, new_values = actorcritic(batch_states)
                new_probs = torch.distributions.Categorical(logits=logits)
                new_log_probs = new_probs.log_prob(batch_actions)
                entropy = new_probs.entropy()
                
                # Policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actorcritic.parameters(), 0.5)
                optimizer.step()
        
        scheduler.step()
        
        # Logging
        if iteration % 10 == 0:
            avg_reward = np.mean([envs.total_rewards[i] for i in range(envs.len())])
            rewards_history.append(avg_reward)
            print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}")
            
    
    # Plot training progress
    plt.plot(rewards_history)
    plt.xlabel("Iteration (x10)")
    plt.ylabel("Average Reward")
    plt.title("PPO Training Progress")
    plt.savefig("ppo_training.png")
    plt.close()
    torch.save(actorcritic.state_dict(), "cartpole_ppo_final.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    nb_actor = 4
    envs = Environments(nb_actor, device)
    state_dim = envs.envs[0].observation_space.shape[0]
    nb_actions = envs.envs[0].action_space.n
    actorcritic = ActorCritic(state_dim, nb_actions).to(device)
    
    PPO(envs, actorcritic, nb_iterations=500, T=256)
    torch.save(actorcritic.state_dict(), "cartpole_ppo_final.pth")

