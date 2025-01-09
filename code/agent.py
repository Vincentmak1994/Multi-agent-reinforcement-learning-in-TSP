import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from utils import extract_tensors
from dqn_network import DQN
from utils import EpsilonGreedyStrategy
from memory_buffer import ReplayMemory
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, strategy, num_cities, policy_net, target_net, memory, lr, gamma, agent_name, budget, is_training=True):
        self.name = agent_name
        self.memory = memory
        self.gamma = gamma
        self._initial_budget = budget
        self._current_budget = budget
        self._prize_collected = 0
        self.current_step = 0 
        self.current_episode = 0
        self.strategy = strategy
        self.num_cities = num_cities
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device uses for tensor calculations eg. CPU or GPU
        self.epsilon = []
        self.policy_net = policy_net
        self.target_net = target_net
        self.is_training = is_training
        self.epsilon_values = []
        
        self.target_net.load_state_dict(self.policy_net.state_dict())  #initialize weights in target network be the same as policy network 
        self.target_net.eval() #tell PyTorch that this network is not in training mode
        print(self.target_net.eval())
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=lr, amsgrad=True)
    
    
    def reset_budget_n_prize(self):
        self._current_budget = self._initial_budget
        self._prize_collected = 0
        
    def collect_prize_n_adjust_budget(self, prize, cost):
        self._current_budget -= cost
        self._prize_collected += prize
    
    def current_budget(self):
        return self._current_budget
    
    def collected_prizes(self):
        return self._prize_collected
    
    def select_action(self, state, state_location, feasible_mask):
        # state is a city (int)
        if self.strategy:
            # update eplison by episode instead of steps 
            rate = self.strategy.get_exploration_rate(self.current_episode)
        else:
            rate = 0
            # self.strategy.get_exploration_rate(self.current_episode)
        self.epsilon_values.append(rate)
        self.current_step += 1
    
        input_city = state_location
        # torch.cat((state_location), 1)

        #turn off gradient tracking (no learning)
        with torch.no_grad():
            next_city_prediction = self.policy_net(input_city)
        
        feasible_mask = torch.tensor(feasible_mask, dtype=torch.float32).view(1, 1, -1)
        masked_predictions = next_city_prediction * feasible_mask
        masked_probabilities = masked_predictions / masked_predictions.sum(dim=-1, keepdim=True)

        if not self.is_training:
            return torch.argmax(masked_probabilities, dim=-1).item()
        
        # exploration
        if rate > random.random():
            ''' Example
            nonzero_indices = tensor([[0, 0, 1], [0, 0, 4], [0, 0, 5]])
            random_index = 1
            chosen_action = 4 
            '''
            nonzero_indices = torch.nonzero(masked_probabilities) #find index of non-zero porbability [[batch, _, city]]. return 2D tensor 
            random_index = torch.randint(0, len(nonzero_indices), (1,)).item() #randomly pick an index between 0 and len(nonzero_indices)
            chosen_action = nonzero_indices[random_index][-1].item()    #pick city based on index (row) given above. Note: city is at the end of an array 

        else:           
        # exploitation 
            chosen_action = torch.argmax(masked_probabilities, dim=-1).item()

        return chosen_action   
            
    def learn(self, experiences):
        (state_batch, state_locations_batch, action_batch, next_state_batch, reward_batch, is_done_batch, next_state_location_batch) = extract_tensors(experiences)

        current_q_values = QValues.get_current(self.policy_net, state_batch, state_locations_batch, action_batch)

        next_q_values = QValues.get_next(self.target_net, next_state_batch, next_state_location_batch).unsqueeze(1)

        target_q_values = reward_batch + (self.gamma * (1 - is_done_batch.float()) * next_q_values)

        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        # loss = F.mse_loss(current_q_values, target_q_values)
        # loss = Variable(loss, requires_grad = True)
        # print(f"Loss:{loss.requires_grad}")

        self.optimizer.zero_grad()  #zero out gradients to avoid accumulating the gradients 
        loss.backward()        #backprop

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        '''
        # for param in self.policy_net.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)  # In-place gradient clipping                
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        '''
        self.optimizer.step()   
        return loss
    
    def get_epsilon(self):
        return self.epsilon
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())  #update target network 
        
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def get_current(policy_net, states, state_locations, actions):
        input_city = state_locations
        # torch.cat((state_locations), 1)
        return policy_net(input_city).gather(dim=1, index=torch.argmax(actions, dim=1).unsqueeze(1))
    
    @staticmethod
    def get_next(target_net, next_states, next_state_locations):
        input_city = next_state_locations
        # torch.cat((next_state_locations), 1)        
        return target_net(input_city).max(1)[0]

''' Test epsilon 
def main():
    policy_net = DQN(n_observation=10,
                             n_actions=10,
                             hidden_size_1=32, 
                             hidden_size_2=32)

    target_net =  DQN(n_observation=10,
                        n_actions=10,
                        hidden_size_1=32, 
                        hidden_size_2=32)
    
    eps_start = 0.9   #Starting epsilon (for epsilon greedy strategy)
    eps_end = 0.1
    eps_decay = 2500
    memory_size = 200000 #replay memory capacity 
    lr = 0.001       #learning rate 
    gamma = 0.999   #discount rate
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, 10, policy_net, target_net, ReplayMemory(memory_size), lr=lr, gamma=gamma, agent_name='simple_dqn', budget=6000)

    n_ep = 5000
    rates = []
    for i in range(n_ep):
        agent.current_episode +=1 
        rate = agent.select_action(1, 1, 1)
        rates.append(rate)
    plt.plot(rates, 'b-', label='linear')
    plt.show()

if __name__ == '__main__':
    main()
'''    