from utils import EpsilonGreedyStrategy
import torch 
import os
from sensor_network import Network
from dqn_network import DQN, DQN_RNN
from agent import Agent 
from memory_buffer import ReplayMemory
from training_loop import training_loop
import csv
from utils import save_to_file

budgets = [1000]
# [4000, 6000, 8000, 10000]
num_test = 1
num_episodes= [5]
# 
test_number = -1
batch_size = 32
gamma = 0.999   #discount rate
eps_start = 0.9   #Starting epsilon (for epsilon greedy strategy)
eps_end = 0.1
eps_decay = 4000
target_update = 10 #how frequent to update weights of the target network 
memory_size = 400000 #replay memory capacity 
lr = 0.001       #learning rate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
hidden_size_1 = 128
hidden_size_2 = 256
best_model_info = {}
all_model_weights = {}
all_model_weight = {}
start_city, end_city = 0, 39

city_test_info = {}

note = f'''Test all 48 cities, budgets = [2000, 4000, 6000, 8000, 100000], starting_city = 0, ending_city = 39
This is the first test for different starting and ending cities.
'''
save_to_file(path=f'project/code/different_city/model/rnn/city_10/test_{test_number}', file_name='test_content', data=note, has_filenames=False, fieldnames=None, type='txt')
for test in range(num_test):
    for num_episode in num_episodes:
        for budget in budgets:
            for i in [start_city]:
                # range(1)
                city_test_info[f"starting_city_{i}"] = {}
                episode_info = {}
                city_network = Network(can_revisit=False).build_all_city_sample(start_city=i, end_city=end_city,unit='mile')
                # .build_sample_network()
                # .build_city_sample(start_city=i, unit='mile')
                # .build_all_city_sample(start_city=i, unit='mile')
                # print(city_network.get_network_matrix())
                num_cities = city_network.num_node
            #     print("# of cities: {}\nTotal prize:{}\nCan agent revisit?:{}\nStarting city:{}\nEnding city:{}\n".format(num_cities, sum(city_network.get_prizes_array()), city_network.can_revisit, city_network.current_nodes(), city_network.end))
                # print(f"************** model_size_{hidden_size_1}_{hidden_size_2} **************")
                # print(f'Total episode: {num_episode}, total budget: {budget}')
                # print("=== Test {} : starting city {} ===".format(i+1, i+1))

                rnn_policy_net = DQN_RNN(num_cities=num_cities,
                                            hidden_size_1=hidden_size_1, 
                                            hidden_size_2=hidden_size_2).to(device)
                rnn_target_net = DQN_RNN(num_cities=num_cities,
                                            hidden_size_1=hidden_size_1, 
                                            hidden_size_2=hidden_size_2).to(device)

                rnn_agent = Agent(strategy, num_cities, rnn_policy_net, rnn_target_net, ReplayMemory(memory_size), lr=lr, gamma=gamma, agent_name='rnn_dqn', budget=budget)

                # memory = ReplayMemory(memory_size)

                agents = [rnn_agent]

                game_info, model_weights = training_loop(city_network, num_episode, agents, batch_size, target_update, save_model=True)
                for key in game_info.keys():
                    print(f"agent:{key}, max_reward:{game_info[key]['max_reward']}, max_reward_ep:{game_info[key]['max_reward_episode']}")
                print("\n")


                # episode_detail_file =  f'{training_detail}/training_data_each_episode.csv'
                training_detail_path = f'project/code/different_city/model/rnn/city_{num_cities}/test_{test_number}/budget_{budget}/starting_city_{i}/total_ep_{num_episode+10}/num_test_{test}'
                for agent in game_info:
                    filename = game_info[agent]['detail'][list(game_info[agent]['detail'].keys())[0]]
                    save_to_file(training_detail_path, 'training_data_each_episode', game_info[agent]['detail'], has_filenames=True, fieldnames=filename,  type='csv')
                    


                city_test_info[f"starting_city_{i}"] = game_info
                all_model_weight[f"starting_city_{i}"] = model_weights

            best_model_info[f"model_size_{hidden_size_1}_{hidden_size_2}"] = city_test_info
            all_model_weights[f"model_size_{hidden_size_1}_{hidden_size_2}"] = all_model_weight

            opt_solution = [938]
            # [479,481,495,505,514,529,536,542,551,551]
            test_file = f'project/code/different_city/model/rnn/city_{num_cities}/test_{test_number}/episode_{num_episode}/num_test_{test}'
            os.makedirs(test_file, exist_ok=True)
            with open(f'{test_file}/summary.txt', 'w') as file:
                for model_size in best_model_info.keys():
                    file.write("===========================================================================\n")
                    file.write(f"====================== model_size: {model_size} ======================\n")
                    file.write("===========================================================================\n")
                    # print(f"******** model_size: {model_size} ********\n")
                    city = 0 
                    for starting_city in best_model_info[model_size].keys():
                        file.write(f"Starting City: {starting_city}\n")
                        # print(f"Starting City: {starting_city}\n")
                        for model, data in best_model_info[model_size][starting_city].items():
                            file.write(f"Model_name: {model}, max_reward: {data['max_reward']}, best_path: {data['best_path']}\n")
                            # , is_optimal: {opt_solution[city] == data['max_reward']}
                            # print(f"Model_name: {model}, max_reward: {data['max_reward']}, best_path: {data['best_path']}, is_optimal: {opt_solution[city] == data['max_reward']}")
                        city += 1
                        file.write("\n")
            
            
            training_detail = f'project/code/different_city/model/rnn/city_{num_cities}/test_{test_number}/num_test_{test}'
            os.makedirs(training_detail, exist_ok=True)
            training_file = f'{training_detail}/training_data.csv'
            training_file_exists = os.path.isfile(training_file)
            with open(training_file, 'a', newline='') as csvfile:
                fieldnames = ['starting_city', 'total_episode', 'total_budget', 'model_name', 'max_reward', 'best_path']
                # , 'is_optimal'
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not training_file_exists:
                    writer.writeheader()
                city = 0 
                for starting_city in best_model_info[model_size].keys():
                    for model, data in best_model_info[model_size][starting_city].items():
                        output = {'starting_city' : city
                                , 'total_episode': num_episode
                                , 'total_budget' : budget
                                , 'model_name' : model
                                , 'max_reward' : data['max_reward']
                                , 'best_path' : data['best_path']
                                # , 'is_optimal' : opt_solution[city] == data['max_reward']
                                }
                        writer.writerow(output)
                    city += 1 
        

            # save the best performed model weight 
            print(f'best_model_info: {best_model_info}')
            for model_size in best_model_info.keys():
                print(f"******** saving model_size: {model_size} ********\n")
                city = 0 
                for starting_city in best_model_info[model_size].keys():
                    for model, data in best_model_info[model_size][starting_city].items():
                        path = f'project/code/different_city/model/rnn/city_{num_cities}/test_{test_number}/budget_{budget}/starting_city_{i}/total_ep_{num_episode+10}/num_test_{test}/best_model'
                        # f'{test_file}/{model_size}/batch_{batch_size}_lr_{lr}_memoryBuffer_{memory_size}/starting_city_{city}/budget_{budget}/num_test_{test}/'
                        os.makedirs(path, exist_ok=True)
                        filename = f"best_model.tar"
                        torch.save({
                            'epoch': data['best_model_weights']['epoch'],
                            'model_state_dict': data['best_model_weights']['model_state_dict'],
                            'optimizer_state_dict': data['best_model_weights']['optimizer_state_dict'],
                            # Add any other relevant information
                        }, path+'/'+filename)
                    city += 1
'''
            # save all model weights 
            for model_size in all_model_weights.keys():
                print(f"******** saving model_size: {model_size} ********\n")
                city = 0 
                for starting_city in all_model_weights[model_size].keys():
                    for episode, data in all_model_weights[model_size][starting_city].items():
                        path = f'{test_file}/{model_size}/batch_{batch_size}_lr_{lr}_memoryBuffer_{memory_size}/starting_city_{city}/budget_{budget}/num_test_{test}'
                        os.makedirs(path, exist_ok=True)
                        filename = f"{episode}_model.tar"
                        torch.save({
                            'epoch': data['episode'],
                            'model_state_dict': data['model_state_dict'],
                            'optimizer_state_dict': data['optimizer_state_dict'],
                            # Add any other relevant information
                        }, path+'/'+filename)
                    city += 1
'''