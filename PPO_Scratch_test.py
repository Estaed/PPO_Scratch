# test_ppo.py

import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
        env = gym.make('CartPole-v1', render_mode="human")  # Ensure environment is rendered
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    
    def render(self, env_id=0):
        self.envs[env_id].render()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    nb_actor = 1
    envs = Environments(nb_actor, device)
    state_dim = envs.envs[0].observation_space.shape[0]
    nb_actions = envs.envs[0].action_space.n
    actorcritic = ActorCritic(state_dim, nb_actions).to(device)

    try:
        actorcritic.load_state_dict(torch.load("cartpole_ppo_final.pth", map_location=device))
        print("Loaded pretrained weights")
    except FileNotFoundError:
        print("Pretrained weights not found.  Please run train_ppo.py first.")
        exit()
    
    env_id = 0
    obs, _ = envs.envs[env_id].reset()
    done = False
    total_reward = 0
    
    while not done:
        envs.render(env_id)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits, _ = actorcritic(obs_tensor)
            action = torch.argmax(logits).item()
        
        obs, reward, terminated, truncated = envs.step(env_id, action)
        done = terminated or truncated
        total_reward += reward
        
    print(f"Demo completed with total reward: {total_reward}")
