import random
import math
import networkx as nx
import time
import json
import os 
from node import Node 
import numpy as np
from utils import distance
import matplotlib.pyplot as plt



class Network():
    def __init__(self, num_node=10, width=10, length=10, num_data_node=9, max_capacity=50, transmission=5, chkpt_dir='data/'):
        self.num_node = num_node              #number of nodes in the network
        self.width = width                    #width x of the network 
        self.length = length                  #length y of the network 
        self.num_data_node = num_data_node    #number of data nodes  
        self.max_capacity = max_capacity      #maximum capacity of each sensor node
        self.transmission = transmission      #transmission rage of each sensor node 
        self._nodes = []                      #the sensor network represnting in array
        self._network = {}                    #edges connecting from one node to another
        self.placeholder = int(self.transmission*2)
        self._network_matrix = [[self.placeholder for _ in range(self.num_node)] for _ in range(self.num_node)]
        self._visited = set()
        self._current_node = 0 
        self.chkpt_dir = chkpt_dir+"{}_nodes".format(self.num_node)
        self.is_connected = False
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

    def find_edges(self):
        for i in range(self.num_node):
            if i not in self._network:
                self._network[i]={}
            for j in range(i+1, self.num_node):
                if j not in self._network:
                    self._network[j] = {}
                d=distance(self._nodes[i].get_x(), self._nodes[i].get_y(), self._nodes[j].get_x(), self._nodes[j].get_y())
                # tr_ = utils.to_transmission(d)
                if d <= self.transmission:  #if distince between two nodes are within transmisison range (in meter)
                    self._network[i][j] = d
                    self._network[j][i] = d
        self.build_cost_matrix()
    
    def build_sample_network(self):
        self.num_node = 7              
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
        self._nodes.append(Node(id=3, x=1, y=10, is_data_node=True, data_packets=1))
        self._nodes.append(Node(id=4, x=5, y=5, is_data_node=True, data_packets=3))
        self._nodes.append(Node(id=5, x=0, y=0, is_data_node=True, data_packets=5))
        self._nodes.append(Node(id=6, x=0, y=10, is_data_node=True, data_packets=4))
            
        self._network =  {0:{1:2, 4:2.5, 5:3},
                        1:{0:2, 2:2},
                        2:{1:2, 3:2.5},
                        3:{2:2.5, 6:0.5},
                        4:{0:2.5, 6:2.5},
                        5:{0:3, 6:4},
                        6:{3:0.5, 4:2.5, 5:4}}
        self.build_cost_matrix()
        self._visited.add(0)

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
        return self
    
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
    
    def current_nodes(self):
        return self._current_node
    
    def state_representation(self):
        cities = self._network_matrix[self._current_node]
        cities = [self.placeholder if i in self._visited else cities[i] for i in range(len(cities))]
        cities.append(self._current_node)
        return cities
    
    def get_edges_pair(self):
        edges_pair = []
        for src in self._network:
            for dst in self._network[src]:
                edges_pair.append((src, dst))      
        return edges_pair
    
    def visalize(self):
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
        nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
        plt.show()
        

    # visit() function visit the given node and add it to the is_visited list and return reward of the node: Int 
    def visit(self, node):
        if node in self._visited or self._network_matrix[self._current_node][node] == self.placeholder:
            reward = -self.max_capacity*0.2
        # TODO: assert - if node is in visited 
        else:
            reward = self._nodes[node].data_packets
            self._visited.add(node)
            self._current_node = node 
            
        next_state_representation = self.state_representation()     
        is_done = max((len(self._visited) == self.num_node), min(next_state_representation[:-1]) == self.placeholder)
        return next_state_representation, reward, is_done
    
    def reset(self):
        self._visited = set()
        self._current_node = 0 
        self._visited.add(0)


def main():
    network = Network(num_node=10, width=10, length=10, num_data_node=9, max_capacity=10, transmission=5).build_network()
    print("===== Display Network =====")
    # print(network.get_network_matrix())
    print(network.list_nodes())
    print(network.visalize())


if __name__ == "__main__":
    main()