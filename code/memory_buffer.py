import random
from collections import namedtuple

Experience = namedtuple(
    'Experience',
    ('state', 'state_location', 'action', 'next_state', 'reward', 'is_done', 'next_state_location')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    # Push experience to replay memory
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience #replace the oldest experience 
        self.push_count += 1
    
    # return sample of the batch size 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    # return True if replay memory has size >= batch_size
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
    def get_current_push_count(self):
        return self.push_count