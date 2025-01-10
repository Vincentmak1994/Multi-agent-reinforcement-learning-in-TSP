import numpy as np 
from marl_agent import MARL_agent
from sensor_network import Network
from heapq import heappush, heapify, heappop
from tqdm import tqdm
import time
import os 
import csv
from utils import save_to_file

def dijkstra_min_cost_to_node(nodes, graph, start):
    node_data = {}
    
    for key in graph.keys():
        node_data[key] = {'prizes':0,'cost': float('inf'), 'pred':[]}
    node_data[start]['cost'] = 0    #starting point needs 0 cost to reach 
    
    
    for node in nodes:
        id = node.get_id()
        node_data[id]['prizes'] = node.get_data_packets()
        node_data[id]['neighbors'] = list(graph[id].keys())
        node_data[id]['checked_nodes'] = []
    
    visited =  set()
    mid_heap = [(0, start)]
    # for i in range(len(node_data)):
    while len(mid_heap) > 0:
        (temp_cost, temp) =heappop(mid_heap)
        if temp not in visited:
            visited.add(temp)
            
            for neighbor in graph[temp]:
                if neighbor not in visited:
                    cost = node_data[temp]['cost'] + graph[temp][neighbor]  #total cost of the neighbor from current node
                    if cost < node_data[neighbor]['cost']:
                        node_data[neighbor]['cost'] = cost
                        node_data[neighbor]['pred'] = node_data[temp]['pred'] + list({temp})
                    heappush(mid_heap, (node_data[neighbor]['cost'], neighbor))
        heapify(mid_heap)
    # print(node_data)
    return node_data

def marl_(network, start_city, end_city=None, num_agent=3, budget=6000, epi=1000, version='v1', r_table=None, q_table=None):
    learning_rate = 0.1
    discount_factor = 0.3
    trade_off = 0.5
    delta = 1
    beta = 2
    w = 1500
    epi = epi
    is_done = False
    graph = network.get_network()
    nodes = network.get_all_nodes()
    current_max_prize = 0 
    current_max_prize_path = []
    training_detail = {}
    model_weights = {}
    save_training_step = 10
    best_model_weight = {}
    
    if not end_city:
        end_city = start_city

    # initalize r and Q tables
    node_data = dijkstra_min_cost_to_node(nodes, graph, end_city)
    # print(f"node_data: {node_data}")

    if r_table and q_table:
        R = r_table
        Q = q_table
    
    else:
        Q = {}
        R = {}
        C = {}
        for u in graph:
            Q[u] = {}
            R[u] = {}
            C[u] = {}
            for v in graph[u]:
                Q[u][v] = 0
                if version == "v2":
                    C[u][v] = (node_data[u]['prizes']+node_data[v]['prizes'])/graph[u][v]
                    R[u][v] = 0
                else: 
                    if node_data[v]['prizes'] > 0:
                        R[u][v] = (-1*graph[u][v])/ node_data[v]['prizes']
                        # R[u][v] = 0
                    else:
                        R[u][v] = 0
            
    # print(f"min_cost: {node_data}")
    # print(f'graph: {graph}')

    #Return - dict{} sorted by values in descending order (can be used by exploration)
    def exploitation(feasible_set, cur_node):
        d = {}
        for neighbor in feasible_set:
            # print("before - neighbor: {}, Q[{}][{}]: {}".format(neighbor, cur_node, neighbor, Q[cur_node][neighbor]))
            if version == 'v1':
                res = (Q[cur_node][neighbor]**delta)*node_data[neighbor]['prizes']/(graph[cur_node][neighbor]**beta)
            else:
                res = (Q[cur_node][neighbor]**delta)/(graph[cur_node][neighbor]**beta)
            # print("after - neighbor: {}, Q[{}][{}]: {}".format(neighbor, cur_node, neighbor, res))
            d[neighbor] = 0 if res < 0 else res 
        
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
    
    #Return: selected node (next node) 
    def exploration(feasible_set, cur_node):
        if len(feasible_set) == 1:
            return feasible_set[0]
        if version == 'v1' or version=='v2':
            p = {}
            exploite = exploitation(feasible_set, cur_node)
            # print("exploite: {}".format(exploite))
            base = sum(w for w in exploite.values() if w > 0)
            # base = sum(exploite.values())
            for neighbor in feasible_set:
                if base == 0:
                    p[neighbor] = 1/len(exploite)
                else:
                    if exploite[neighbor] > 0:
                        p[neighbor] = exploite[neighbor]/base
                    else:
                        p[neighbor] = 0
            next_node = np.random.choice(list(p.keys()), 1, p=list(p.values()))
            next_node = next_node[0]
        
        # if version == 'v2':
        #     d = {}
        #     for neighbor in feasible_set:
        #         res = (C[cur_node][neighbor]**delta)*node_data[neighbor]['prizes']/(graph[cur_node][neighbor]**beta)
                
        #         d[neighbor] = 0 if res < 0 else res 

        #     base = sum(w for w in d.values() if w > 0)
        #     p = {}

        #     for neighbor in feasible_set:
        #         if base == 0:
        #             p[neighbor] = 1/len(d)
        #         else:
        #             if d[neighbor] > 0:
        #                 p[neighbor] = d[neighbor]/base
        #             else:
        #                 p[neighbor] = 0
            
        #     next_node = dict(sorted(p.items(), key=lambda x: x[1], reverse=True))
        #     next_node = list(next_node.keys())[0]
        return next_node

    # update Q Table
    def update_Q_table(cur_node, next_node, is_learning, learning_rate):
        # TODO: update formula for learning stage 
        #print("...Updating Q Table. Current Q[{}][{}] = {}".format(cur_node, next_node, Q[cur_node][next_node]))
        # print("Q table for Next node: {}".format(Q[next_node]))
        node_w_max_q = max(Q[next_node].values())
        # sorted(Q[next_node], key=lambda x: Q[next_node][x], reverse=True)[0]
        #print("Max node: {}".format(node_w_max_q))
        if is_learning:
            Q[cur_node][next_node] = (1-learning_rate)*Q[cur_node][next_node] + learning_rate*(discount_factor*node_w_max_q)
        else:
            Q[cur_node][next_node] = (1-learning_rate)*Q[cur_node][next_node] + learning_rate*(R[cur_node][next_node]+ discount_factor*node_w_max_q)
            # temp = (1-learning_rate)*Q[cur_node][next_node] + learning_rate*(R[cur_node][next_node]+ discount_factor*node_w_max_q)
            # Q[cur_node][next_node] = 0 if temp < 0 else temp
        #print("New Q Table: {}".format(Q))
        
        
    # update R table
    def update_R_table(cur_node, next_node, max_prize):
        #print("...Updating R Table. Current R[{}][{}] = {}".format(cur_node, next_node, R[cur_node][next_node]))
        if version == 'v1':
            R[cur_node][next_node] = R[cur_node][next_node] + (w/max_prize)
        if version == 'v2':
            R[cur_node][next_node] = R[cur_node][next_node] + (ep*w)
            # R[cur_node][next_node] = R[cur_node][next_node] + (ep*w/max_prize)
        #print("New R Table: {}".format(R))

    def is_feasible(B, cost_to_node, cost_to_home):
        # print('B: {}, total_cost: {}, cost_to_node:{}, cost_to_home:{}'.format(B, cost_to_node+cost_to_home, cost_to_node, cost_to_home))
        return True if B >= (cost_to_node+cost_to_home) else False
    
    # return feasible set 
    def get_feasible_set(agent, cur_node, can_revisit):
        feasible_set=[]
        for neighbor in node_data[cur_node]['neighbors']:
            if can_revisit:
                if is_feasible(agent.get_budget(), graph[cur_node][neighbor], node_data[neighbor]['cost']):
                    feasible_set.append(neighbor)
            else:
                if not agent.is_visited(neighbor) and is_feasible(agent.get_budget(), graph[cur_node][neighbor], node_data[neighbor]['cost']):
                    feasible_set.append(neighbor)
        return feasible_set
    
    
    

    for ep in tqdm(range(epi)):
        # create / reset agent 
        round = 0
        max_prize = 0
        max_prize_agent = 0
        episode_detail = {}
        model_weight = {}

        if ep == 0:
            model_weight['episode'] = -1
            model_weight['starting_city'] = start_city
            model_weight['num_agent'] = num_agent
            model_weight['total_episode'] = epi
            model_weight['total_budget'] = budget
            model_weight['q_table'] = Q
            model_weight['r_table'] = R
            model_weights[-1] = model_weight
        
        agents = []
        for i in range(num_agent):
            agents.append(MARL_agent(start_city, budget))

        is_done = [False]*num_agent
        ep_start = time.time()
        while sum(is_done) != num_agent:    #not all agents are done
            round += 1
            #print("======= Round {} - {} agents are done =======".format(round, sum(is_done)))
            for j in range(len(agents)):
                # print("")
                # print("-- Agent {}, isDone: {}".format(j+1, is_done[j]))
                if not is_done[j]:
                    agent = agents[j]               #get agent 
                    cur_node = agent.get_cur_node() #current node of the agent 
                    # TODO: Enable agent to collect prize multiple time for learning
                    feasible_set = get_feasible_set(agent, cur_node, False)    # get feasible set
                    # print(f"feasible_set:{feasible_set}")
                    # print("Current Node: {}, Feasible Set: {}".format(cur_node, feasible_set))
                    if len(feasible_set) == 0: # if feasible_set is empty 
                        if cur_node != end_city:       #if current node is already in starting point, do nothing
                            # print("No feasible set, moving to starting point")
                            agent.add_cost(node_data[cur_node]['cost']) #update cost 
                            agent.update_budget(node_data[cur_node]['cost'])# update budget
                            home_path = node_data[cur_node]['pred']
                            for i in range(len(home_path)-1, -1, -1):   #add path back to starting point & update Q table along the way
                                temp = home_path[i]
                                agent.add_path(temp)    
                                if version == 'v1':
                                    update_Q_table(cur_node, temp, True, learning_rate)  #update Q table 
                                cur_node=temp
                            agent.set_cur_node(end_city)   #move to s 
                        
                            # print("No feasible set. Current location is at starting point. Agent {} is done". format(j+1))
                        is_done[j] = True      
                    else:   #feasible_set is not empty 
                        if agent.get_q() <= trade_off:
                            # print("Feasible set found. q = {}. Exploitation".format(agent.get_q()))
                            next_node = list(exploitation(feasible_set, cur_node).keys())[0]
                            # print("Exploitation returned {}".format(next_node))
                        else:
                            # print("Feasible set found. q = {}. Exploration".format(agent.get_q()))
                            next_node = exploration(feasible_set, cur_node)
                            # print("Exploration returned {}".format(next_node))
                        agent.add_path(next_node)
                        agent.add_cost(graph[cur_node][next_node])  # update cost
                        agent.update_budget(graph[cur_node][next_node])# update budget 
                        agent.update_prize(node_data[next_node]['prizes']) # update prize 
                        # if version == 'v1':
                        update_Q_table(cur_node, next_node, True, learning_rate) # update Q table 
                        agent.set_cur_node(next_node)    # move to next node 
                        agent.visit_node(next_node)   # mark next node as visit 
                        if agent.get_prize() > max_prize:
                            max_prize = agent.get_prize()
                            max_prize_agent = j
                            # print("Agent {} has the maximum prize with {}".format(j+1, max_prize))
            # print("Round {} done".format(round))
        # for i in range(len(agents)):
        #     agent = agents[i]
            # print("Agent {} - prize:{}, path:{}, cost:{}, budget left: {}".format(i+1, agent.get_prize(), agent.get_path(), agent.get_cost(), agent.get_budget()))
        #find route with max prize 
        ep_end = time.time()
        execution_time = ep_end-ep_start
        lowest_cost = budget
        for agent in agents:
            if agent.get_cost() < lowest_cost:
                lowest_cost = agent.get_cost()
                lowest_cost_agent = agent 

        if version == 'v1':
            max_prize_path = agents[max_prize_agent].get_path()
            for i in range(len(max_prize_path)-1):
                cur_node, next_node = max_prize_path[i], max_prize_path[i+1]
                update_R_table(cur_node, next_node, max_prize)  #update R table 
                update_Q_table(cur_node, next_node, False, learning_rate)  #update Q table 
        
        if version == 'v2':
            lowest_cost_path = lowest_cost_agent.get_path() 
            for i in range(len(lowest_cost_path)-1):
                cur_node, next_node = lowest_cost_path[i], lowest_cost_path[i+1]
                update_R_table(cur_node, next_node, None)  #update R table 
                update_Q_table(cur_node, next_node, False, learning_rate)  #update Q table 

            # if max_prize > current_max_prize:
            #     current_max_prize = max_prize

            #     max_prize_path = agents[max_prize_agent].get_path()
            #     current_max_prize_path = max_prize_path
            #     for i in range(len(max_prize_path)-1):
            #         cur_node, next_node = max_prize_path[i], max_prize_path[i+1]
            #         update_R_table(cur_node, next_node, max_prize)  #update R table 
            #         update_Q_table(cur_node, next_node, False, learning_rate)  #update Q table 
        
        episode_detail['starting_city'] = start_city
        episode_detail['episode'] = ep+1
        episode_detail['num_agent'] = num_agent
        episode_detail['total_episode'] = epi
        episode_detail['total_budget'] = budget
        episode_detail['max_reward'] = max_prize
        if version=='v1':
            episode_detail['max_reward_path'] = max_prize_path
        else:
            episode_detail['max_reward_path'] = lowest_cost_path
        episode_detail['remaining_budget'] = agents[max_prize_agent].get_budget()
        episode_detail['execution_time'] = execution_time
        training_detail[ep] = episode_detail

        
        if ep%save_training_step == 0: #TODO: save weights if max_reward
            model_weight['episode'] = ep+1
            model_weight['starting_city'] = start_city
            model_weight['num_agent'] = num_agent
            model_weight['total_episode'] = epi
            model_weight['total_budget'] = budget
            model_weight['q_table'] = Q
            model_weight['r_table'] = R
            model_weights[ep] = model_weight


        
        

    #execution stage 
    #print("========== Execution Stage ==========")
    print(Q)
    final_agent = MARL_agent(start_city, budget)
    final_agent_detail = {}
    # final_agent.update_prize(node_data[start]['prizes']) # collect prize at starting point
    is_done = False
    start_time = time.time()
    while not is_done:
        cur_node = final_agent.get_cur_node()
        feasible_set = get_feasible_set(final_agent, cur_node, False)
        #print("Current Node: {}, Feasible Set: {}".format(cur_node, feasible_set))
        if len(feasible_set) == 0:   
            if cur_node != end_city:       #if current node is already in starting point, do nothing
                #print("No feasible set, moving to starting point")
                final_agent.add_cost(node_data[cur_node]['cost']) #update cost 
                final_agent.update_budget(node_data[cur_node]['cost'])# update budget
                home_path = node_data[cur_node]['pred']
                for i in range(len(home_path)-1, -1, -1):   #add path back to starting point & update Q table along the way
                    temp = home_path[i]
                    final_agent.add_path(temp)    
                final_agent.set_cur_node(end_city)   #move to s 
            else:
                #print("No feasible set. Current location is at starting point. It is done")
                is_done = True  
        else:
            neighbors =  dict(sorted(Q[cur_node].items(), key=lambda x:x[1], reverse=True))
            for neighbor in neighbors:
                if neighbor in feasible_set:
                    next_node = neighbor
                    break
            # next_node =sorted(Q[cur_node], key=lambda x:Q[cur_node][x], reverse=True)[0]
            final_agent.add_path(next_node)
            final_agent.add_cost(graph[cur_node][next_node])  # update cost
            final_agent.update_budget(graph[cur_node][next_node])# update budget 
            final_agent.update_prize(node_data[next_node]['prizes']) # update prize 
            #print("Moving from {} to {}. Updated prize: {}, Updated Budget: {}".format(cur_node, next_node, final_agent.get_prize(), final_agent.get_budget()))
            final_agent.set_cur_node(next_node)    # move to next node 
            final_agent.visit_node(next_node)   # mark next node as visit 
        end_time = time.time()
        execution_time = end_time-start_time
    return final_agent.get_path(), final_agent.get_prize(), final_agent.get_budget(), training_detail, model_weights, execution_time


def main():
    test_number=1
    num_test = 1
    version = 'v1'
    start_city, end_city = 0, 41
    note = f'marl_{version}\n testing all 48 cities, starting_city = {start_city}, ending city={end_city}'
    save_to_file(path=f'project/code/different_city/model/marl_{version}/test_{test_number}/start_city_{start_city}_end_city_{end_city}', file_name='test_content', data=note, has_filenames=False, fieldnames=None, type='txt')
    budgets = [10000]

    epis= [5000]

    num_agents = [5]

    test_result = {}
    all_model_weights = {} #to save all (city) weight from Q-table 
    opt_solution = [479,481,495,505,514,529,536,542,551,551]
    
    network = Network(num_node=10, width=10, length=10, num_data_node=9, max_capacity=10, transmission=5).build_all_city_sample(start_city=start_city, end_city=end_city,unit='mile')
    
    # .build_sample_network()
    # .build_all_city_sample(start_city=0, unit='mile')
    for budget in budgets:
        for epi in epis:
            for num_agent in num_agents:
                for test in range(num_test):
                    print(f'************************************* test: {test+1} *************************************')
                    for city  in [start_city]: #, 15, 35, 42
                        start_time = time.time()
                        path, prize, remaining_budget, training_detail, model_weights, execution_time = marl_(network, start_city=start_city, end_city=end_city, num_agent=num_agent, budget=budget, epi=epi, version=version)
                        end_time = time.time()

                        # save training data
                        training_file_path = f'project/code/different_city/model/marl_{version}/test_{test_number}/start_city_{start_city}_end_city_{end_city}/ep_{epis[0]}_{epis[-1]}/budget_{budgets[0]}_{budgets[-1]}/num_agent_{num_agents[0]}_{num_agents[-1]}/training'
                        os.makedirs(training_file_path, exist_ok=True)
                        training_file = f'{training_file_path}/marl_10_city.csv'
                        file_exists = os.path.isfile(training_file)
                        with open(training_file, 'a', newline='') as csvfile:
                            fieldnames = training_detail[0].keys()
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            if not file_exists:
                                writer.writeheader()
                            for i in training_detail:
                                writer.writerow(training_detail[i])
                        
                        # save execution of each city in test_result
                        test_result[city]={}
                        test_result[city]['test'] = test+1
                        test_result[city]['num_agent'] = num_agent
                        test_result[city]['starting_city']= city
                        test_result[city]['reward'] = prize
                        test_result[city]['path'] = path
                        test_result[city]['total_budget'] = budget
                        test_result[city]['total_episode'] = epi
                        test_result[city]['remaining_budget'] = remaining_budget
                        test_result[city]['execution_time'] = execution_time
                        # test_result[city]['is_optimal'] = prize == opt_solution[city]
                        test_result[city]['training_data_path'] = training_file

                        # save city's model weights in csv
                        path = f'project/code/different_city/model/marl_{version}/test_{test_number}/start_city_{start_city}_end_city_{end_city}/ep_{epis[0]}_{epis[-1]}/budget_{budgets[0]}_{budgets[-1]}/num_agent_{num_agents[0]}_{num_agents[-1]}/weights/starting_city_{city}'
                        file_name = 'model_weight'
                        save_to_file(path, file_name, model_weights, has_filenames=True, fieldnames=model_weights[list(model_weights.keys())[0]], type='csv')
                        
                        

                        print(f"========= Starting city {city+1} =========\ntotal_prize: {prize}\npath:{path}\nremaining_budget:{remaining_budget}\n\n")

                    
                    # save execution data
                    file_path = f'project/code/different_city/model/marl_{version}/test_{test_number}/start_city_{start_city}_end_city_{end_city}/ep_{epis[0]}_{epis[-1]}/budget_{budgets[0]}_{budgets[-1]}/num_agent_{num_agents[0]}_{num_agents[-1]}/execution'
                    os.makedirs(file_path, exist_ok=True)
                    file = f'{file_path}/marl_10_city.csv'
                    file_exists = os.path.isfile(file)

                    with open(file, 'a', newline='') as csvfile:
                        fieldnames = test_result[start_city].keys()
                        # ['test', 'num_agent', 'starting_city', 'reward', 'path', 'total_budget', 'remaining_budget', 'execution_time', 'is_optimal']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        for i in test_result:
                            writer.writerow(test_result[i])
                        


if __name__ == '__main__':
    main()