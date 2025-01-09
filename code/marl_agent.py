import random

class MARL_agent():
    def __init__(self, s, B):
        self.cur_node = s
        self.prize = 0
        self.budget = B
        self.cost = 0 
        self.is_done = False
        self.q = random.uniform(0, 1)
        self.visited = set({s})
        self.path = [s]
    
    def get_prize(self):
        return self.prize
    
    def get_budget_in_meter(self):
        return self.budget*0.002

    def get_budget(self):
        return self.budget
    
    def get_status(self):
        return self.is_done
    
    def set_status(self, status):
        self.is_done = status

    def get_q(self):
        return self.q
    
    def get_cost(self):
        return self.cost 
    
    def add_cost(self, cost):
        self.cost += cost

    def update_budget(self, budget):
        self.budget -= budget

    def update_prize(self, prize):
        self.prize += prize
    
    def is_visited(self, node):
        if node in self.visited:
            return True
        return False 
    
    def get_cur_node(self):
        return self.cur_node
    
    def set_cur_node(self, node):
        self.cur_node = node

    def visit_node(self, node):
        self.visited.add(node)
    
    def get_path(self):
        return self.path

    def add_path(self, node):
        self.path.append(node)
    
    def print_all_info(self):
        print("cur_node: {}, prizes:{}, budget: {}, cost:{}, is_done:{}, q:{}, visited:{}, path:{}".format(self.cur_node, self.prize, self.budget, self.cost, self.is_done, self.q, self.visited, self.path))