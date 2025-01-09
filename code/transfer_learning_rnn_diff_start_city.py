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

def load_model(path, num_cities, hidden_1, hidden_2, name=None):
    checkpoint = torch.load(path)
    policy_net = DQN_RNN(num_cities, hidden_1, hidden_2)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    return policy_net



budgets = [40000]
# [10000]
'10000, 20000, 30000, 40000'
num_test = 1
num_episodes= [50000]
test_number = 112.3
batch_size = 32
gamma = 0.999   #discount rate
eps_start = 0.9   #Starting epsilon (for epsilon greedy strategy)
eps_end = 0.1
eps_decay = 25000
target_update = 10 #how frequent to update weights of the target network 
memory_size = 400000 #replay memory capacity 
lr = 0.001       #learning rate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
best_model_info = {}
all_model_weights = {}
all_model_weight = {}
city_test_info = {}
hidden_size_1=128
hidden_size_2=256
note = f'''Transfer Learning, test different starting city (33), original start from city 0- Test all 48 cities, from 40k budget best model, 50,000ep'''
save_to_file(path=f'project/code/model/rnn/city_10/test_{test_number}', file_name='test_content', data=note, has_filenames=False, fieldnames=None, type='txt')

# fine-tune
for test in range(num_test):
    for num_episode in num_episodes:
        for budget in budgets:
            for i in [33]:
                city_test_info[f"starting_city_{i}"] = {}
                episode_info = {}
                city_network = Network(can_revisit=False).build_all_city_sample(start_city=i, unit='mile')
                num_cities = city_network.num_node

                path="project/code/model/rnn/city_48/test_0.031/episode_5000/num_test_0/model_size_128_256/batch_32_lr_0.001_memoryBuffer_200000/starting_city_0/budget_40000/num_test_0/best_model.tar"
                pretrained_model = load_model(path, num_cities=48, hidden_1=128, hidden_2=256, name="pretrained_48_city_40000_budget_best_model")
                pretrained_target = load_model(path, num_cities=48, hidden_1=128, hidden_2=256, name="pretrained_48_city_40000_budget_best_model_target_net")

                agent = Agent(strategy, num_cities, pretrained_model, pretrained_target, ReplayMemory(memory_size), lr=lr, gamma=gamma, agent_name='fine-tune_48_city_40000_budget', budget=budget)

                agents = [agent]

                game_info, model_weights = training_loop(city_network, num_episode, agents, batch_size, target_update, save_model=True)
                for key in game_info.keys():
                    print(f"agent:{key}, max_reward:{game_info[key]['max_reward']}, max_reward_ep:{game_info[key]['max_reward_episode']}")
                print("\n")


                # episode_detail_file =  f'{training_detail}/training_data_each_episode.csv'
                training_detail_path = f'project/code/model/rnn/city_{num_cities}/test_{test_number}/budget_{budget}/starting_city_{i}/total_ep_{num_episode+10}/num_test_{test}'
                for agent in game_info:
                    filename = game_info[agent]['detail'][list(game_info[agent]['detail'].keys())[0]]
                    save_to_file(training_detail_path, 'training_data_each_episode', game_info[agent]['detail'], has_filenames=True, fieldnames=filename,  type='csv')
                    


                city_test_info[f"starting_city_{i}"] = game_info
                all_model_weight[f"starting_city_{i}"] = model_weights
            
            best_model_info[f"model_size_{hidden_size_1}_{hidden_size_2}"] = city_test_info
            all_model_weights[f"model_size_{hidden_size_1}_{hidden_size_2}"] = all_model_weight

            opt_solution = [938]
            # [479,481,495,505,514,529,536,542,551,551]
            test_file = f'project/code/model/rnn/city_{num_cities}/test_{test_number}/episode_{num_episode}/num_test_{test}'
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
                            file.write(f"Model_name: {model}, max_reward: {data['max_reward']}, best_path: {data['best_path']}, is_optimal: {opt_solution[city] == data['max_reward']}\n")
                            # print(f"Model_name: {model}, max_reward: {data['max_reward']}, best_path: {data['best_path']}, is_optimal: {opt_solution[city] == data['max_reward']}")
                        city += 1
                        file.write("\n")
            
            
            training_detail = f'project/code/model/rnn/city_{num_cities}/test_{test_number}/num_test_{test}'
            os.makedirs(training_detail, exist_ok=True)
            training_file = f'{training_detail}/training_data.csv'
            training_file_exists = os.path.isfile(training_file)
            with open(training_file, 'a', newline='') as csvfile:
                fieldnames = ['starting_city', 'total_episode', 'total_budget', 'model_name', 'max_reward', 'best_path', 'is_optimal']
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
                                , 'is_optimal' : opt_solution[city] == data['max_reward']}
                        writer.writerow(output)
                    city += 1 
        

            # save the best performed model weight 
            for model_size in best_model_info.keys():
                print(f"******** saving model_size: {model_size} ********\n")
                city = 0 
                for starting_city in best_model_info[model_size].keys():
                    for model, data in best_model_info[model_size][starting_city].items():
                        path = f'{test_file}/{model_size}/batch_{batch_size}_lr_{lr}_memoryBuffer_{memory_size}/starting_city_{city}/budget_{budget}/num_test_{test}'
                        os.makedirs(path, exist_ok=True)
                        filename = f"best_model.tar"
                        torch.save({
                            'epoch': data['best_model_weights']['epoch'],
                            'model_state_dict': data['best_model_weights']['model_state_dict'],
                            'optimizer_state_dict': data['best_model_weights']['optimizer_state_dict'],
                            # Add any other relevant information
                        }, path+'/'+filename)
                    city += 1




