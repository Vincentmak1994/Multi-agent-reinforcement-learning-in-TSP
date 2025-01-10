import numpy as np
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import count
from memory_buffer import Experience


def training_loop(network, num_episodes, agents, batch_size, target_update, collect_prize_from_start=False, save_model=True):
    game_info = {}
    num_cities = network.num_node
    save_model_step = 10 #model will be saved in every 10 episode
    model_weights = {} #store all model weights 

    for episode in tqdm(range(num_episodes)):
        model_weight = {} 

        for agent in agents:
            if agent.name not in game_info:
                game_info[agent.name] = {}
                game_info[agent.name]['max_reward'] = 0
                game_info[agent.name]['best_path'] = None
                game_info[agent.name]['max_remaining_budget'] = 0
                game_info[agent.name]['max_reward_episode'] = 0
                game_info[agent.name]['total_time_in_sec'] = 0
                if save_model:
                    game_info[agent.name]['best_model_weights'] = {}
                game_info[agent.name]['detail'] = {}
                
            start_time = time.time()
            start_city = network.start
            ep_loss = []

            if collect_prize_from_start:
                current_reward = network.setup() #reset env, collect frize from starting city
            else:
                network.reset()
                current_reward = 0
                
            path = [start_city] #add starting city to path
            agent.reset_budget_n_prize()
            agent.collect_prize_n_adjust_budget(current_reward, 0)
            agent.current_episode += 1

            for timestep in count():
                state = network.current_nodes(one_hot_encoding=False)
                state = F.one_hot(torch.tensor([state]), num_classes=num_cities).to(torch.float32)

                state_location = network.state_location()
                state_location = torch.tensor(state_location, dtype=torch.float32).unsqueeze(0)

                feasible_mask = network.get_feasible_mask(agent.current_budget())
                feasible_mask = torch.tensor(feasible_mask, dtype=torch.float32).view(1, 1, -1)
                
                

                action = agent.select_action(state, state_location, feasible_mask)

                next_state_representation, reward, is_done, next_state_location, cost = network.visit(action)
                path.append(action)
                agent.collect_prize_n_adjust_budget(reward, cost)
                current_reward += reward

                agent.memory.push(Experience(state
                                       ,state_location
                                       ,F.one_hot(torch.tensor([action]), num_classes=num_cities).to(torch.float32)
                                       ,F.one_hot(torch.tensor([action]), num_classes=num_cities).to(torch.float32)
                                       ,torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
                                       ,torch.tensor([is_done], dtype=torch.float32).unsqueeze(0)
                                       ,torch.tensor(next_state_location, dtype=torch.float32).unsqueeze(0)))

                if agent.memory.can_provide_sample(batch_size):
                    experiences = agent.memory.sample(batch_size)
                    loss = agent.learn(experiences) 

                    ep_loss.append(loss.item())
                    # print(f"Episode: {episode}, Step: {timestep}, Loss: {loss.item()}")

                if is_done:
                    end_time = time.time()
                    elapsed_time = end_time-start_time
                    game_info[agent.name]['total_time_in_sec'] += elapsed_time
                    # print("Game:{}\nAgent:{}\nReward:{}\nRemaining budget:{}\nPath:{}\nep_loss:{}\n".format(episode, agent.name, current_reward, agent.current_budget(), path, ep_loss))
                    has_repeat = len(set(path)) != len(path)
                    if action != start_city:
                        cost_to_starting = network.min_cost_graph[network.current_nodes()]['min_cost']
                        agent.collect_prize_n_adjust_budget(0, cost_to_starting)
                        path.append(start_city) #travel back to starting city
                    game_info[agent.name]['detail'][f"episode_{episode+1}"] = {}
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['episode'] = episode
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['reward'] = current_reward
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['avg_loss'] = np.mean(ep_loss)
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['path'] = path
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['remaining_budget'] = agent.current_budget()
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['has_repeated'] = has_repeat
                    game_info[agent.name]['detail'][f"episode_{episode+1}"]['execution_time'] = elapsed_time

                    # save model in every 10 episode
                    if episode % save_model_step == 0:
                        model_weight['episode'] = episode
                        model_weight['model_state_dict'] = agent.policy_net.state_dict()
                        model_weight['optimizer_state_dict'] = agent.optimizer.state_dict()
                        model_weights[f'episode_{episode}'] = model_weight

                    
                    if current_reward > game_info[agent.name]['max_reward']:
                        game_info[agent.name]['max_reward'] = current_reward
                        game_info[agent.name]['max_reward_episode'] = episode
                        game_info[agent.name]['best_path'] = path
                        game_info[agent.name]['max_remaining_budget'] = agent.current_budget()
                        if save_model:
                            game_info[agent.name]['best_model_weights']['epoch'] = episode
                            game_info[agent.name]['best_model_weights']['model_state_dict'] = agent.policy_net.state_dict()
                            game_info[agent.name]['best_model_weights']['optimizer_state_dict'] = agent.optimizer.state_dict()
                    break 

                if episode % target_update == 0:
                    agent.update_target_net()

    return game_info, model_weights