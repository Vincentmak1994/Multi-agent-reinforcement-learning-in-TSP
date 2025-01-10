import numpy as np
import math
from math import radians, cos, sin, asin, sqrt 
import torch
from memory_buffer import Experience
import geopy.distance
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
import os 
import csv
import scipy.stats

class EpsilonGreedyStrategy():
    def __init__(self, start, end, epsilon_decay=5000, decay_type='linear'):
        self.start = start
        self.end = end
        self.epsilon_decay = epsilon_decay 
        self.decay_type = decay_type
    
    def get_exploration_rate(self, current_step):
        if self.decay_type == 'expo':
            return self.exponential_epsilon_decay(current_step)
        if self.decay_type == 'linear':
            return self.linear_epsilon_decay(current_step)
    
    def exponential_epsilon_decay(self, current_step):
        return self.end + (self.start - self.end) * \
                math.exp(-1. * current_step / self.epsilon_decay)
    
    def linear_epsilon_decay(self, current_step):
        return self.end + (self.start - self.end) * max(1- current_step/ self.epsilon_decay, 0)
        # we want to keep epsilon 
    
    # 0.1 + (0.9 - 0.1) * math.exp(-1. * 2500 * decay) = 0.1

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    
    state_batch = torch.cat(batch.state)
    state_locations_batch = torch.cat(batch.state_location)
    action_batch = torch.cat(batch.action)
    next_state_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)
    is_done_batch = torch.cat(batch.is_done)
    next_state_location_batch = torch.cat(batch.next_state_location)

    return (state_batch, state_locations_batch, action_batch, next_state_batch, reward_batch, is_done_batch, next_state_location_batch)

def distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1 )**2) + ((y2-y1)**2))

def geo_distince(lat1, lon1, lat2, lon2, unit):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    if unit=='mile':
        r= 3956
    else:
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


#convert distinct to transmission in J
def to_transmission(distance):
    elec=100*pow(10,-9)
    amp=100*pow(10,-12)
    k=3200
    return (2*elec*k) + (amp*k*pow(distance,2))

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)
        
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def plot_learning_curve(x, scores, figure_file, title, ylabel):
    running_avg = np.zeros(len(scores))
    for i in range(100,len(running_avg)):
        running_avg[i] = np.mean(scores[max(100, i-100):(i+1)])
        # running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x[200:], running_avg[200:])
    # plt.plot(x, running_avg)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(figure_file)

def get_learning_curve_data(x, scores):
    running_avg = np.zeros(len(scores))
    ci = np.zeros(len(scores))
    for i in range(len(running_avg)-100):
        running_avg[i] = np.mean(scores[i:i+100])
        ci[i] = scipy.stats.sem(scores[i:i+100]) * scipy.stats.t.ppf((1 + 0.95) / 2., 100)
    return (x[100:], running_avg[:len(running_avg)-100], ci[:len(running_avg)-100])

def save_to_file(path, file_name, data, has_filenames=True, fieldnames=[], type='csv'):
    supporting_type = ['csv', 'txt']
    os.makedirs(path, exist_ok=True)
    if type not in supporting_type:
        raise Exception(f"{type} is not supprot")
    
    if type == 'csv':
        file = f'{path}/{file_name}.csv'
        file_exists = os.path.isfile(file)
        with open(file, 'a', newline='') as csvfile:
            if has_filenames:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for line in data:
                    writer.writerow(data[line])
    
    if type == 'txt':
            file = f'{path}/{file_name}.txt'
            with open(file, 'w') as file:
                file.write(data)



def main():
    s = EpsilonGreedyStrategy(0.9, 0.1, 2500)
    # s2 = EpsilonGreedyStrategy(0.9, 0.1, decay_type='expo')
    
    n_ep = 5000
    lin_epsilon_values  =[s.get_exploration_rate(step) for step in range(n_ep)]
    # exp_epsilon_values = [s2.get_exploration_rate(step) for step in range(n_ep)]
    plt.plot(lin_epsilon_values, 'r-', label='linear')
    # plt.plot(exp_epsilon_values, 'r-', label='expo')
    plt.ylabel("Epsilon (ε)")
    plt.xlabel("Episode")
    plt.show()

if __name__ == '__main__':
    main()
