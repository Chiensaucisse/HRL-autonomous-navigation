import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, action_bounds, offset):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim + goal_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                        )
        # max value of actions
        self.action_bounds = action_bounds
        self.offset = offset
        
    def forward(self, state, goal):
        
        return (self.actor(torch.cat([state, goal], 1)) * self.action_bounds) + self.offset
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, H):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim + action_dim + goal_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        #nn.Sigmoid()
                        )
        self.H = H
        
    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        return self.critic(torch.cat([state, action, goal], 1)) #* self.H


class TD3:
    def __init__(self, observation_space, action_dim, action_bounds, offset, lr, H, 
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
      
        self.actor = Actor(observation_space, action_dim, action_bounds, offset).to(device)
        self.actor_target = Actor(observation_space, action_dim, action_bounds, offset).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
       
        self.critic1 = Critic(observation_space, action_dim, H).to(device)
        self.critic2 = Critic(observation_space, action_dim, H).to(device)
        self.critic1_target = Critic(observation_space, action_dim, H).to(device)
        self.critic2_target = Critic(observation_space, action_dim, H).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        
        self.mseLoss = torch.nn.MSELoss()
        
      
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0  
        
    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):
        for _ in range(n_iter):
            self.total_it += 1
            
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
       
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size,1)).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            
            with torch.no_grad():
                next_action = self.actor_target(next_state, goal)
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = next_action + noise
                
                target_Q1 = self.critic1_target(next_state, next_action, goal)
                target_Q2 = self.critic2_target(next_state, next_action, goal)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + ((1 - done) * gamma * target_Q)
            
            
            current_Q1 = self.critic1(state, action, goal)
            current_Q2 = self.critic2(state, action, goal)
            critic_loss = self.mseLoss(current_Q1, target_Q) + self.mseLoss(current_Q2, target_Q)
            
           
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            
            if self.total_it % self.policy_delay == 0:
                
                actor_loss = -self.critic1(state, self.actor(state, goal), goal).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                
                self.soft_update(self.actor, self.actor_target, self.tau)
                self.soft_update(self.critic1, self.critic1_target, self.tau)
                self.soft_update(self.critic2, self.critic2_target, self.tau)
                
    def soft_update(self, source, target, tau):
        
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic1.state_dict(), '%s/%s_critic1.pth' % (directory, name))
        torch.save(self.critic2.state_dict(), '%s/%s_critic2.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic1.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, name), map_location='cpu'))
        self.critic2.load_state_dict(torch.load('%s/%s_critic2.pth' % (directory, name), map_location='cpu'))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


class DDPG:
    def __init__(self, state_dim, action_dim, goal_dim,  action_bounds, offset, lr, H):
        
        self.actor = Actor(state_dim, action_dim, goal_dim,  action_bounds, offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, goal_dim, H).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.mseLoss = torch.nn.MSELoss()
    
    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer, n_iter, batch_size):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)
            
            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            gamma = torch.FloatTensor(gamma).reshape((batch_size,1)).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # select next action
            next_action = self.actor(next_state, goal).detach()
            
            # Compute target Q-value:
            target_Q = self.critic(next_state, next_action, goal).detach()
            target_Q = reward + ((1-done) * gamma * target_Q)
            
            # Optimize Critic:
            critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Compute actor loss:
            actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
                
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_crtic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_crtic.pth' % (directory, name), map_location='cpu'))  
        
        
        
        
        
      
        
        
