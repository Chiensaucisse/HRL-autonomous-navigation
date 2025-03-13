import torch
import gym
import os
import numpy as np
from HAC import HAC
from sim import SIM_ENV

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    #################### Hyperparameters ####################
    env_name = "MountainCarContinuous-h-v1"
    save_episode = 100               # keep saving every n episodes
    max_episodes = 100000             # max num of training episodes
    random_seed = 0
    render = False
    
    env = SIM_ENV()
    # env.reset()
    # for i in range(10000):
        
    #     done=False
    #     while done is False:
    #         a=np.random.uniform(0,0.5)
    #         b=np.random.uniform(-1.0,1.0)
    #         state,reward,done,reached=  env.step(lin_velocity=a,ang_velocity=b)
    #         if done:
    #             state,_=env.reset()

    # exit()

    state_dim = 25
    action_dim = 2
    goal_dim = 2
    
    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    
    # primitive action bounds and offset
    action_bounds = np.array([0.25,1.0])
    action_bounds = torch.FloatTensor(action_bounds.reshape(1, -1)).to(device)
    action_offset = np.array([0.25,0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([0.0,-1.0])
    action_clip_high = np.array([0.5,1.0])

    # primitive action bounds and offset
    goal_bounds = np.array([5.0,5.0])
    goal_bounds = torch.FloatTensor(goal_bounds.reshape(1, -1)).to(device)
    goal_offset = np.array([5.0,5.0])
    goal_offset = torch.FloatTensor(goal_offset.reshape(1, -1)).to(device)
    goal_clip_low = np.array([0.0,0.0])
    goal_clip_high = np.array([10.0,10.0])
    
    # state bounds and offset
    state_clip_low = np.concatenate([np.zeros(20), [0.0,-1.0, -1.0], np.zeros(2)])
    state_clip_high = np.ones(state_dim)
    state_bounds_np = (state_clip_high-state_clip_low)/2
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset =  (state_clip_high + state_clip_low)/2
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    
    
    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.1,0.1])        
    exploration_state_noise = np.ones(state_dim)*0.1
    exploration_goal_noise = np.ones(goal_dim)*1.0

    goal_state = np.zeros(state_dim)        # final goal state to be achived
    goal_state[20] = 0.1
    goal_state=np.array([[9.0],[9.0],[0.0]])
    threshold = np.ones(state_dim)* 0.03       # threshold value to check if goal state is achieved
    
    # HAC parameters:
    k_level = 1                 # num of levels in hierarchy
    H = 500                     # time horizon to achieve subgoal
    lamda = 0.3                 # subgoal testing parameter
    
    # DDPG parameters:
    gamma = 0.99                # discount factor for future rewards
    n_iter = 100                # update policy n_iter times in one DDPG update
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.0003
    delay_learning = 20
    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level) 
    os.makedirs(directory, exist_ok=True)
    filename = "HAC_{}".format(env_name)
    #########################################################
    
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # creating HAC agent and setting parameters
    agent = HAC(k_level, H, state_dim, action_dim, goal_dim, render, threshold, 
                action_bounds, action_offset, state_bounds, state_offset, goal_bounds, goal_offset, lr)
    
    agent.set_parameters(lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high,goal_clip_low, goal_clip_high, exploration_action_noise, exploration_state_noise, exploration_goal_noise)
    
    # logging file:
    log_f = open("log.txt","w+")
    
    # training procedure 
    for i_episode in range(1, max_episodes+1):
        agent.reward = 0
        agent.timestep = 0
        env.robot_goal = goal_state
        goal = goal_state.squeeze(axis=1)
        goal = goal_state[:2]
        state,_ = env.reset(random_obstacles=False)
        # collecting experience in environment
        last_state, done, goal_achieved = agent.run_HAC(env, k_level-1, state, goal, False)
        
        if agent.check_goal(last_state, goal, threshold):
            print("################ Solved! ################ ")
            name = filename + '_solved'
            agent.save(directory, name)
        
        # update all levels
        if i_episode > delay_learning:
            agent.update(n_iter, batch_size)
        
        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()
        
        if i_episode % save_episode == 0:
            agent.save(directory, filename)
        
        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))
        
    
if __name__ == '__main__':
    train()
 
