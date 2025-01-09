import random
import numpy as np
from node import Node
from utils import geo_distince, distance
import time
import json
import os 
from heapq import heappush, heapify, heappop
import networkx as nx
import matplotlib.pyplot as plt

class Network():
    def __init__(self, num_node=10, starting_node=0, width=10, length=10, num_data_node=9, max_capacity=50, transmission=5, chkpt_dir='data/', can_revisit=False):
        self.num_node = num_node              #number of nodes in the network
        self.width = width                    #width x of the network 
        self.length = length                  #length y of the network 
        self.num_data_node = num_data_node    #number of data nodes  
        self.max_capacity = max_capacity      #maximum capacity of each sensor node
        self.transmission = transmission      #transmission rage of each sensor node 
        self._nodes = []                      #the sensor network represnting in array
        self._network = {}                    #edges connecting from one node to another
        self.placeholder = 3000
        # int(self.transmission*1.5)
        self._network_matrix = [[self.placeholder for _ in range(self.num_node)] for _ in range(self.num_node)]
        self._visited = set()
        self._current_node = starting_node
        self.chkpt_dir = chkpt_dir+"{}_nodes".format(self.num_node)
        self.is_connected = False
        self._prizes = []
        self.min_cost_graph = {}
        self.start = starting_node
        self.end = num_node-1
        self.can_revisit = can_revisit
        self.temp_agent_budget = 0 #to store agent budget before taking action 
        self.temp_feasible_mask = None
        self.is_start = True #indicating rather the game has just started 
#         _{}x{}_{}_dn_{}_max_{}_transmission/

    '''
    generate_nodes() creates N number of nodes. 
    Each node is randomly generated with an unique x and y coordinate
    For each data node, it has unique amount of data package stored in it between [1, max_capacity]
    '''
    def generate_nodes(self):
        if self.num_data_node >= self.num_node:
            self.num_data_node = self.num_node-1 
            print("Number of data nodes can not be greater than or equal to the total number of nodes in the network. Setting the number of data nodes to {}.".format(self.num_data_node))
        else:
            self.num_data_node
        
        dn = random.sample(range(1,self.num_node), self.num_data_node)
        dn_package = random.sample(range(1,self.max_capacity+1), self.num_data_node)
#         dn_package = random.sample(range(self.max_capacity-self.num_data_node+1,self.max_capacity+1), self.num_data_node)
        dn_created=0
        for i in range(self.num_node):
            x = random.randint(0, self.width)
            y = random.randint(0, self.length)
            if i in dn:   #data node 
                node = Node(i, x, y, True, dn_package[dn_created], self.max_capacity)
                dn_created += 1
            else:       #data sink
                node = Node(i, x, y, False, 0, self.max_capacity)
            self._nodes.append(node)

    def find_edges(self, is_geo = False, unit='mile'):
        for i in range(self.num_node):
            if i not in self._network:
                self._network[i]={}
            for j in range(i+1, self.num_node):
                if j not in self._network:
                    self._network[j] = {}
                if is_geo:
                    d= geo_distince(self._nodes[i].get_x(), self._nodes[i].get_y(), self._nodes[j].get_x(), self._nodes[j].get_y(), unit=unit)
                else:
                    d=distance(self._nodes[i].get_x(), self._nodes[i].get_y(), self._nodes[j].get_x(), self._nodes[j].get_y())
                # tr_ = utils.to_transmission(d)
                if d <= self.transmission:  #if distince between two nodes are within transmisison range (in meter)
                    self._network[i][j] = d
                    self._network[j][i] = d

        self.build_cost_matrix()
    
    def build_city_sample(self, start_city, unit=None):
        '''
        Albany,NY          42.652552778 -73.75732222	100

        Annapolis,MD       38.978611111 -76.49111111	98

        Atlanta,GA         33.749272222 -84.38826111	84

        Augusta,ME         44.307236111 -69.78167778	74

        Austin,TX          30.274722222 -97.74055556	65

        BatonRouge,LA     30.457072222 -91.18740556	50

        Bismarck,ND        46.820813889 -100.7827417	43

        Boise,ID           43.617697222 -116.1996139	37

        Boston,MA          42.357708333 -71.06356389	28

        CarsonCity,NV     39.164075    -119.7662917	19
        '''
        
        self.city_map = {0: 'Albany,NY'
                         ,1: 'Annapolis,MD'
                         ,2: 'Atlanta,GA'
                         ,3: 'Augusta,ME'
                         ,4: 'Austin,TX'
                         ,5: 'BatonRouge,LA'
                         ,6: 'Bismarck,ND'
                         ,7: 'Boise,ID'
                         ,8 : 'Boston,MA'
                         ,9 : 'CarsonCity,NV'
                        }
        
        self.num_node = 10
        self.end = start_city
        self.start = start_city
        self._current_node = start_city #starting at city 7 
        self.transmission = float('inf')
        self.placeholder = 3000 #max dis is about 50 if all nodes are connected 
        
        self._nodes.append(Node(id=0, x=42.652552778, y=-73.75732222, is_data_node=True, data_packets=100))
        self._nodes.append(Node(id=1, x=38.978611111, y=-76.49111111, is_data_node=True, data_packets=98))
        self._nodes.append(Node(id=2, x=33.749272222, y=-84.38826111, is_data_node=True, data_packets=84))
        self._nodes.append(Node(id=3, x=44.307236111, y=-69.78167778, is_data_node=True, data_packets=74))
        self._nodes.append(Node(id=4, x=30.274722222, y=-97.74055556, is_data_node=True, data_packets=65))
        self._nodes.append(Node(id=5, x=30.457072222, y=-91.18740556, is_data_node=True, data_packets=50))
        self._nodes.append(Node(id=6, x=46.820813889, y=-100.7827417, is_data_node=True, data_packets=43))
        self._nodes.append(Node(id=7, x=43.617697222, y=-116.1996139, is_data_node=True, data_packets=37))
        self._nodes.append(Node(id=8, x=42.357708333, y=-71.06356389, is_data_node=True, data_packets=28))
        self._nodes.append(Node(id=9, x=39.164075, y=-119.7662917, is_data_node=True, data_packets=19))
        
        self.find_edges(is_geo=True, unit=unit)
        self.build_cost_matrix()
        self.build_prizes_array()
        self.dijkstra_min_cost_to_node()
        return self
        
    def build_all_city_sample(self, start_city, end_city=None, unit=None):
        self.end = end_city if end_city else start_city
        self.start = start_city
        self._current_node = start_city #starting at city 7 
        self.transmission = float('inf')
        self.placeholder = 50*2 #max dis is about 50 if all nodes are connected 

        self.city_map = {}
        city_id = 0
        with open('project/code/Capital_Cities.txt', 'r') as f:
            for line in f:
                if len(line.strip()) != 0:
                    item =line.split()
                    city, x, y , prize = item[0], float(item[1]), float(item[2]), int(item[3])

                    self.city_map[city_id] = city
                    self._nodes.append(Node(id=city_id, x=x, y=y, is_data_node=True, data_packets=prize))
                    city_id += 1

        self.num_node = city_id
        self._network_matrix = [[self.placeholder for _ in range(self.num_node)] for _ in range(self.num_node)]
        self.find_edges(is_geo=True, unit=unit)
        self.build_cost_matrix()
        self.build_prizes_array()
        self.dijkstra_min_cost_to_node()
        return self


    
    def build_sample_network(self):
        self.num_node = 7       
        self.end = 6
        self.width = 4                      
        self.length = 3                  
        self.num_data_node = 6    
        self.max_capacity = 5      
        self.transmission = 4    
        self.placeholder = int(self.transmission*2)
        self._network_matrix = [[self.placeholder for _ in range(self.num_node)] for _ in range(self.num_node)]
        
        self._nodes.append(Node(id=0, x=0, y=10, is_data_node=True, data_packets=0))
        self._nodes.append(Node(id=1, x=5, y=10, is_data_node=True, data_packets=2))
        self._nodes.append(Node(id=2, x=10, y=10, is_data_node=True, data_packets=2))
        self._nodes.append(Node(id=3, x=10, y=2, is_data_node=True, data_packets=1))
        self._nodes.append(Node(id=4, x=5, y=5, is_data_node=True, data_packets=3))
        self._nodes.append(Node(id=5, x=0, y=0, is_data_node=True, data_packets=5))
        self._nodes.append(Node(id=6, x=10, y=0, is_data_node=True, data_packets=4))
            
        self._network =  {0:{1:2, 4:2.5, 5:3},
                        1:{0:2, 2:2},
                        2:{1:2, 3:2.5},
                        3:{2:2.5, 6:1},
                        4:{0:2.5, 6:2.5},
                        5:{0:3, 6:4},
                        6:{3:1, 4:2.5, 5:4}}
        self.build_cost_matrix()
        self.build_prizes_array()
        self.dijkstra_min_cost_to_node() #ending vextor at node 6 
        self._visited.add(0)

        return self 
    
    def build_sample_network_2(self):
        self.num_node = 5       
        self.end = 4
        self.width = 10                      
        self.length = 10                  
        self.num_data_node =  5   
        self.max_capacity =  5     
        self.transmission = 1    
        self.placeholder = int(self.transmission*2)
        self._network_matrix = [[self.placeholder for _ in range(self.num_node)] for _ in range(self.num_node)]
        
        self._nodes.append(Node(id=0, x=0, y=0, is_data_node=True, data_packets=4))
        self._nodes.append(Node(id=1, x=0, y=0, is_data_node=True, data_packets=1))
        self._nodes.append(Node(id=2, x=0, y=0, is_data_node=True, data_packets=1))
        self._nodes.append(Node(id=3, x=0, y=0, is_data_node=True, data_packets=5))
        self._nodes.append(Node(id=4, x=0, y=0, is_data_node=True, data_packets=1))

            
        self._network =  {0:{1:2.5, 2:3, 3:7, 4:4},
                        1:{0:2.5, 2:0.5, 3:4.5, 4:5.5},
                        2:{0:3, 1:0.5, 3:4, 4:5},
                        3:{0:7, 1:4.5, 2:4, 4:3},
                        4:{0:4, 1:5.5, 2:5, 3:3}
                        }
        self.build_cost_matrix()
        self.build_prizes_array()
        self.dijkstra_min_cost_to_node() #ending vextor at node 6 
        # self._visited.add(4)

        return self 

    def build_cost_matrix(self):
        for i in self._network:
            for j in self._network[i]:
                self._network_matrix[i][j] = self._network[i][j]
        
    def list_nodes(self):
        for node in self._nodes:
            print(node.get_info())
            
    def save_network(self, file_name=""):
        if file_name == "":
            t = time.localtime()
            file_name = time.strftime("%Y_%m_%d_%H_%M_%S", t)
        
        tsp_data = {}
        cities = []
        for node in self._nodes:
            temp = {}
            temp['id'] = node.get_id()
            temp['x'] = node.get_x()
            temp['y'] = node.get_y()
            temp['is_data_node'] = node.is_DN()
            temp['data_packets'] = node.get_data_packets()
            cities.append(temp)

        tsp_data['cities'] = cities
        tsp_data['edges'] = self._network
        
        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)
        
        json_file_path = self.chkpt_dir+'/'+file_name+'.json'
        with open(json_file_path, 'w') as f:
            json.dump(tsp_data, f, indent=4)
    
    def load_network(self, file_name, num_node):
        json_file_path = "data/{}_nodes/{}.json".format(num_node, file_name)
        with open(json_file_path, 'r') as f:
            loaded_data = json.load(f)
        cities = loaded_data['cities']
        edges = loaded_data['edges']
        
        self.num_node = num_node
        self._nodes = []
        for i in range(self.num_node):
            city = cities[i]
            self._nodes.append(Node(id=city['id'], x=city['x'], y=city['y'], is_data_node=city['is_data_node'], data_packets=city['data_packets']))
        self._network = edges
        self._network = self.transfer_dict()
        self.build_cost_matrix()
        return self
    
    '''
    When saving edge as json object, the key has become string
    Using this function to convert string keys into int 
    '''
    def transfer_dict(self):
        temp = {}
        for city in self._network:
            city = int(city)
            temp[city] = {}
            for neighbor in self._network[str(city)]:
                neighbor = int(neighbor)
                temp[city][neighbor] = self._network[str(city)][str(neighbor)]
        return temp
    
    
    def build_network(self):
        while not self.is_connected:
            print("===== Building Sensor Network =====")
            self.generate_nodes()
            self.find_edges()
            # print(self._network_matrix)
            # print(self.visalize())
            if self.is_connect():
                self.is_connected = True
                print("Sensor network has successfully generated")
            else:
                print("Network is not a complete graph")
                self._nodes = []                     
                self._network = {}  
            
            self.build_prizes_array()
            self.dijkstra_min_cost_to_node()
        return self
    
    def build_prizes_array(self):
        self._prizes = []
        for city in self._nodes:
            # print(f"adding city:{city} prize:{city.data_packets}")
            prize = city.data_packets
            self._prizes.append(prize)
            # print(f"current _prizes_array:{self._prizes}")
    
    def get_prizes_array(self):
        return self._prizes

    def is_connect(self):
        
        def dfs(city):
            visited.add(city)
            for neighbor in self._network[city]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        visited = set()
        city = self._current_node
        dfs(city)
        
        return len(visited) == self.num_node 
                

    def get_all_nodes(self):
        return self._nodes

    def get_network(self):
        return self._network
    
    def get_network_matrix(self):
        return self._network_matrix
    
    def current_nodes(self, one_hot_encoding=False):
        return self._current_node if not one_hot_encoding else self.one_hot_encode()
    
    def state_location(self):
        return self._nodes[self._current_node].getLocation()
    
    def state_representation(self, one_hot_encode=True):
        if one_hot_encode:
            current_city = self.one_hot_encode()
        else:
            current_city = [self._current_node]
    
        cities = self._network_matrix[self._current_node]
        # masking
        cities = [self.placeholder if i in self._visited else cities[i] for i in range(len(cities))]
        # cities.append(self._current_node)
        return [*cities, *current_city]
    
    def one_hot_encode(self):
        current_city = np.zeros(self.num_node)
        current_city[self._current_node] = 1
        return current_city
    
    def get_feasible_mask(self, budget):
#         print("agent_budget:{}".format(budget))
        self.temp_agent_budget = budget
        mask = [0]*self.num_node
        neighbors = self._network_matrix[self._current_node]

        for i in range(len(neighbors)):

            '''
            Infeasible conditions (cannot revisit):
                1. city has been visited 
                2. city is not directly connected 
                3. agent cannot reach end vector after visiting the city 
                
            Infeasible conditions (can revisit):
                1. city is not directly connected
                2. agent cannot reach end vector after visiting the city
            '''
            if not self.can_revisit:
                if i in self._visited or neighbors[i] == self.placeholder or neighbors[i]+self.min_cost_graph[i]['min_cost'] > budget:
                    mask[i] = 0
                else:
                    mask[i] = 1
            else:
                if neighbors[i] == self.placeholder or neighbors[i]+self.min_cost_graph[i]['min_cost'] > budget:
                    mask[i] = 0
                else:
                    mask[i] = 1
                
        self.temp_feasible_mask = mask
        return mask
    
    def get_edges_pair(self):
        edges_pair = []
        for src in self._network:
            for dst in self._network[src]:
                edges_pair.append((src, dst))      
        return edges_pair
    
    def dijkstra_min_cost_to_node(self):
#         start = ending vertex 
        start = self.end
        node_data = {}
        for key in self._network.keys():
            node_data[key] = {'prizes':0,'min_cost': float('inf'), 'pred':[]}
        node_data[start]['min_cost'] = 0    #starting point needs 0 cost to reach 

        for idx in range(self.num_node):
            node_data[idx]['prizes'] = self._nodes[idx].get_data_packets()
        visited =  set()
        mid_heap = [(0, start)]
        # for i in range(len(node_data)):
        while len(mid_heap) > 0:
            (temp_cost, temp) =heappop(mid_heap)
            if temp not in visited:
                visited.add(temp)

                for neighbor in self._network[temp]:
                    if neighbor not in visited:
                        cost = node_data[temp]['min_cost'] + self._network[temp][neighbor]  #total cost of the neighbor from current node
                        if cost < node_data[neighbor]['min_cost']:
                            node_data[neighbor]['min_cost'] = cost
                            node_data[neighbor]['pred'] = node_data[temp]['pred'] + [temp]
                        heappush(mid_heap, (node_data[neighbor]['min_cost'], neighbor))
            heapify(mid_heap)
        
        self.min_cost_graph = node_data

    
    def visalize(self, show_edge=True):
        edges_pair = self.get_edges_pair()
#         print(edges_pair)
        
        G = nx.DiGraph()
        G.add_edges_from(edges_pair)
        
        black_edges = [edge for edge in G.edges()]
        pos={}
        color = []
        
        for i in range(self.num_node):
            node = self._nodes[i]
            pos[i] = np.array(node.getLocation())

            if node.is_DN():
                color.append('#8EF1FF')
            else:
                color.append('#FF6666')
#         print(pos)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                                node_size = 500, node_color = color)
        nx.draw_networkx_labels(G, pos)
#         nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True, width=2)
        if show_edge:
            nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
        plt.show()
        

    # visit() function visit the given node and add it to the is_visited list and return reward of the node: Int 
    def visit(self, node):
#         print("current_node:{}\ntrying to visit:{}\n".format(self._current_node, node))
        '''
        Reward (cannot revisit / can revisit):
            1. if city prize has been collected, reward = 0; otherwise, reward = self._nodes[node].data_packets
        '''
        if node not in self._visited:
            reward = self._nodes[node].data_packets
            self._visited.add(node)
        else:
            reward = 0

        cost = self._network_matrix[self._current_node][node]
        self._current_node = node 
               
        next_state_representation = self.state_representation(one_hot_encode=False)     
        '''
        Terminate game conditions (cannot revisit):
            1. visiting node = end vector 

        Terminate game conditions (can revisit):
            1. if visiting node = end vector:
                if there is feasible cities and there is prize to collect -> continue 
                else -> terminate 
            
            After visiting the node, if no feasible cities to visit 
        '''
        next_state_budget = self.temp_agent_budget - cost
        next_feasible_city = self.get_feasible_mask(next_state_budget)
        if self.can_revisit:
            if self._current_node == self.end:
                for i in range(len(next_feasible_city)):
                    if next_feasible_city[i] == 1:
                        if i in self._visited:
                            next_feasible_city[i] = 0

            is_done = True if max(next_feasible_city)==0 else False
        else:
            is_done = max(self._current_node == self.end, max(self.temp_feasible_mask) == 0, max(next_feasible_city)==0)
        
#         is_done = max((len(self._visited) == self.num_node), min(next_state_representation[:-1]) == self.placeholder, max(self.temp_feasible_mask)==0)
        next_state_location = self.state_location() if not is_done else [0,0]
        return next_state_representation, reward, is_done, next_state_location, cost
    
#     to be deprecated - replaced by setup 
    def reset(self):
        self._visited = set()
        self._current_node = self.start
        self._visited.add(self._current_node)
        self.temp_agent_budget = 0
        self.is_start = True
    
    def setup(self):
        self._visited = set()
        self._current_node = self.start
        self._visited.add(self._current_node)
        self.temp_agent_budget = 0
        self.is_start = True
        
        return self._nodes[self.start].data_packets
        


def main():    
    network = Network(num_node=10, width=10, length=10, num_data_node=9, max_capacity=10, transmission=5).build_all_city_sample(start_city=0, end_city=39,unit='mile')
    print(network.min_cost_graph)
    # .build_sample_network_2()
    # .build_city_sample(start_city=0, unit='mile')
    # .build_sample_network()
    # .build_all_city_sample(start_city=0, unit='mile')
    # print("===== Display Network =====")
    # print("# of cities: {}\nTotal prize:{}\nCan agent revisit?:{}\nStarting city:{}\nEnding city:{}\n".format(network.num_node, sum(network.get_prizes_array()), network.can_revisit, network.current_nodes(), network.end))
    # print(network.get_network_matrix())
    # print(sum(network.get_prizes_array()))
    # print("")
    # print(network.get_network())
    # print(network.visalize())



if __name__ == "__main__":
    main()