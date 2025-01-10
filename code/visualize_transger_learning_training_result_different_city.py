import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plot_learning_curve, get_learning_curve_data

budgets = [10000]

cities = [0]
for city in cities:
    for budget in budgets:
        # if budget == 10000:
        #     path = f'project/code/model/rnn/city_48/test_0.03/budget_10000/starting_city_0/total_ep_5010/num_test_0/training_data_each_episode.csv'
        # else:
        #     path = f"project/code/model/rnn/city_48/test_0.031/budget_{budget}/starting_city_0/total_ep_5010/num_test_0/training_data_each_episode.csv"
        path = f'project/code/different_city/model/rnn/city_48/test_1/budget_{budget}/starting_city_0/total_ep_5010/num_test_0/training_data_each_episode.csv'
        # f"project/code/model/rnn/city_48/test_250/budget_{budget}/starting_city_{city}/total_ep_5010/num_test_0/training_data_each_episode.csv"
        rnn_df = pd.read_csv(path)
        episode = rnn_df['episode']
        rnn_reward = rnn_df['reward']
        rnn_running_avg_data = get_learning_curve_data(episode, rnn_reward)

        marl_v1_transfer = pd.read_csv('project/code/transfer_learning/marl_v1/test_1/ep_5000_5000/num_agent_5/based_starting_city_0/transfer_starting_city_0_ending_city_41/training/marl_all_city.csv')
        # 'project/code/model/marl_v2/test_250/ep_5000_5000/budget_2000_10000/num_agent_5_5/training/marl_10_city.csv'
        marl_v1_transfer = marl_v1_transfer[(marl_v1_transfer['total_budget'] == budget) & (marl_v1_transfer['starting_city'] == city) ][:5000]
        marl_v2_training_reward = marl_v1_transfer['max_reward']
        marl_v2_running_avg_data = get_learning_curve_data(episode, marl_v2_training_reward)
        # marl_v2_training_time = marl_v1_transfer['execution_time']
        # marl_v2__time_avg = get_learning_curve_data(episode, marl_v2_training_time)

        marl_v1_df = pd.read_csv('project/code/different_city/model/marl_v1/test_1/start_city_0_end_city_41/ep_5000_5000/budget_10000_10000/num_agent_5_5/training/marl_10_city.csv')
        # "project/code/model/marl_v1/test_250.1/ep_5000_5000/budget_10000_10000/num_agent_5_5/training/marl_10_city.csv"
        
        marl_v1_df = marl_v1_df[(marl_v1_df['total_budget'] == budget) & (marl_v1_df['starting_city'] == city)][:5000]
        marl_v1_training_reward = marl_v1_df['max_reward']
        marl_v1_running_avg_data = get_learning_curve_data(episode, marl_v1_training_reward)


        plt.plot(marl_v1_running_avg_data[0], marl_v1_running_avg_data[1], label='Random Init, B=10000', color='g', marker='^', markevery=500)
        plt.fill_between(marl_v1_running_avg_data[0], marl_v1_running_avg_data[1]-marl_v1_running_avg_data[2], marl_v1_running_avg_data[1]+marl_v1_running_avg_data[2], color='#04fb2a')

        plt.plot(marl_v2_running_avg_data[0], marl_v2_running_avg_data[1], label='Transfer, B=10000', color='r', marker='o', markevery=500)
        plt.fill_between(marl_v2_running_avg_data[0], marl_v2_running_avg_data[1]-marl_v2_running_avg_data[2], marl_v2_running_avg_data[1]+marl_v2_running_avg_data[2], color='#FFBEAF')

        plt.legend(fontsize="15")
        # plt.title('Training Reward Comparison - RNN vs MARL')
        plt.ylabel('Reward', fontsize=20)
        plt.xlabel('Episode', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'result_different_city/transfer_learning/P-MARL_based_model_starting_city_0_ending_city_39_transfer_starting_city_0_ending_city_41.pdf')
        plt.show()




