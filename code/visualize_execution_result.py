import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

budgets = [4000, 6000, 8000, 10000]
cities = [0]
# , 15, 35, 42
# [10000, 20000, 30000, 40000]
marl_v1_df = pd.read_csv('project/code/different_city/model/marl_v1/test_1/ep_5000_5000/budget_4000_10000/num_agent_5_5/execution/marl_10_city.csv')
        # "project/code/model/marl_v1/test_250.1/ep_5000_5000/budget_10000_10000/num_agent_5_5/execution/marl_10_city.csv"
# marl_v1_df = pd.read_csv("project/code/model/marl_v1/test_250/ep_5000_5000/budget_2000_10000/num_agent_5_5/execution/marl_10_city.csv")
marl_v2_df = pd.read_csv('project/code/different_city/model/marl_v2/test_1/ep_5000_5000/budget_4000_10000/num_agent_5_5/execution/marl_10_city.csv')
# "project/code/model/marl_v2/test_250/ep_5000_5000/budget_2000_10000/num_agent_5_5/execution/marl_10_city.csv"
graph = ['reward', 'distance', 'time']

for city in cities:
#     for budget in budgets:
        rnn_pred_df = pd.read_csv('project/code/different_city/model/rnn/prediction/city_48/starting_city_0/test_1/prediction_data.csv')
        # f"project/code/model/rnn/city_48/starting_city_{city}/test_250.1/prediction_data.csv"
        rnn_pred_df = rnn_pred_df[rnn_pred_df['total_budget'] != 2000]
        marl_v1_df_temp = marl_v1_df[(marl_v1_df['starting_city'] == city) & (marl_v1_df['total_budget'] != 2000)]
        #  & (marl_v1_df['total_episode'] == budget)
        marl_v2_df_temp = marl_v2_df[(marl_v2_df['starting_city'] == city) & (marl_v2_df['total_budget'] != 2000)]
        #  & (marl_v2_df['total_episode'] == budget)

        barWidth = 0.20
        br1 = np.arange(len(budgets)) 
        br2 = [x + barWidth for x in br1] 
        br3 = [x + barWidth for x in br2] 

        for graph_type in graph:
                if graph_type == 'reward':
                        plt.bar(br1, marl_v1_df_temp['reward'], color='#04fb2a', width=barWidth, edgecolor ='grey', label ='P-MARL', hatch='-')
                        plt.bar(br2, marl_v2_df_temp['reward'] , color='#FFBEAF', width=barWidth, edgecolor ='grey', label ='Ant-Q', hatch='o')
                        plt.bar(br3, rnn_pred_df['reward'], color='#AFE9FF', width=barWidth, edgecolor ='grey', label ='RNN')
                        plt.xlabel('Budgets', fontsize=25)
                        plt.ylabel('Reward', fontsize=25)
                        plt.xticks([r + barWidth for r in range(len(budgets))], 
                                budgets, fontsize=25)
                        plt.yticks(fontsize=25)
                        plt.legend(fontsize="15")
                        plt.tight_layout()
                        plt.savefig(f"result_different_city/execution/antQ_marl_rnn_comparison_total_prize_starting_city_{city}.png") 
                        plt.show()
                if graph_type == 'distance':
                        plt.bar(br1, [a-b for a,b in zip(budgets,marl_v1_df_temp['remaining_budget'])], color='#04fb2a', width=barWidth, edgecolor ='grey', label ='P-MARL', hatch='-')
                        plt.bar(br2, [a-b for a,b in zip(budgets,marl_v2_df_temp['remaining_budget'])] , color='#FFBEAF', width=barWidth, edgecolor ='grey', label ='Ant-Q', hatch='o')
                        plt.bar(br3, [a-b for a,b in zip(budgets,rnn_pred_df['remaining_budget'])], color='#AFE9FF', width=barWidth, edgecolor ='grey', label ='RNN')
                        plt.xlabel('Budgets', fontsize=25)
                        plt.ylabel('Distance', fontsize=25)
                        plt.xticks([r + barWidth for r in range(len(budgets))], 
                                budgets, fontsize=25)
                        plt.yticks(fontsize=25)
                        plt.legend(fontsize="15")
                        plt.tight_layout()
                        plt.savefig(f"result_different_city/execution/antQ_marl_rnn_comparison_distance_traveled_starting_city_{city}.png") 
                        plt.show()
                if graph_type == 'time':
                        plt.bar(br1, marl_v1_df_temp['execution_time']*1000, color='#04fb2a', width=barWidth, edgecolor ='grey', label ='P-MARL', hatch='-')
                        plt.bar(br2, marl_v2_df_temp['execution_time']*1000 , color='#FFBEAF', width=barWidth, edgecolor ='grey', label ='Ant-Q', hatch='o')
                        plt.bar(br3, rnn_pred_df['time_taken']*1000, color='#AFE9FF', width=barWidth, edgecolor ='grey', label ='RNN')
                        plt.xlabel('Budgets', fontsize=25)
                        plt.ylabel('Time (ms)', fontsize=25)
                        plt.xticks([r + barWidth for r in range(len(budgets))], 
                                budgets,fontsize=25)
                        plt.yticks(fontsize=25)        
                        plt.ylim(0, 1.6)
                        plt.legend(fontsize="15")
                        plt.tight_layout()
                        plt.savefig(f"result_different_city/execution/antQ_marl_rnn_comparison_execute_time_starting_city_{city}.png") 
                        plt.show()




# marl_v1_df

# marl_v2_df = pd.read_csv("project/code/model/marl_v2/test_250/ep_5000_5000/budget_2000_10000/num_agent_5_5/execution/marl_10_city.csv")

# rnn_reward = rnn_pred_df['reward']
# # # [1390, 1878, 2213, 2452]
# marl_reward = [272, 930, 467, 462, 629]
# # # [1797, 2452, 2452, 2452]
# ant_q_reward = [272, 681, 1168, 1527, 1817]
# # # [1917, 2452, 2452, 2452]



# rnn_remaining = rnn_pred_df['remaining_budget']
# marl_remaining = []
# # [22.801456, 5691.950926, 15691.950926, 25691.950926]
# ant_q_remaining = []
# # [6.37626478477182, 5691.950926, 15691.950926, 25691.950926]
# # # rnn_distance = [19992.438273334068, 29578.568576329133, ]

# rnn_time = rnn_pred_df['time_taken']
# # marl_time = [9.12596154212949, 11.524316787719703, 13.16286849975584, 13.967457056045516 ]
# # ant_q_time = [18.453358650207498, 25.255861997604352, 26.316390752792344, 27.17423701286315]


# # # br3 = [x + barWidth for x in br2] 
# # # br4 = [x + barWidth for x in br3] 
# # # # br5 = [x + barWidth for x in br4] 
# # # # br6 = [x + barWidth for x in br5] 



# # plt.bar(br1, [a-b for a,b in zip(budgets,ant_q_remaining)], color='g', width=barWidth, edgecolor ='grey', label ='Ant-Q')
# # plt.bar(br2, [a-b for a,b in zip(budgets,marl_remaining)] , color='r', width=barWidth, edgecolor ='grey', label ='P-MARL')
# # plt.bar(br3, [a-b for a,b in zip(budgets,rnn_remaining)], color='b', width=barWidth, edgecolor ='grey', label ='RNN')
# # plt.xlabel('Budgets', fontsize=15)
# # plt.ylabel('Distance', fontsize=15)
# # plt.xticks([r + barWidth for r in range(len(budgets))], 
# #         budgets, fontsize=15)
# # plt.yticks(fontsize=15)
# # plt.legend(fontsize="15")
# # plt.tight_layout()
# # plt.savefig("antQ_marl_rnn_comparison_distance_traveled.png") 
# # plt.show()








# # plt.bar(br1, ant_q_time, color='g', width=barWidth, edgecolor ='grey', label ='Ant-Q')
# # plt.bar(br2, marl_time , color='r', width=barWidth, edgecolor ='grey', label ='P-MARL')
# # plt.bar(br3, rnn_time, color='b', width=barWidth, edgecolor ='grey', label ='RNN')
# # plt.xlabel('Budgets', fontsize=15)
# # plt.ylabel('Time in second', fontsize=15)
# # plt.xticks([r + barWidth for r in range(len(budgets))], 
# #         budgets,fontsize=15)
# # plt.yticks(fontsize=15)        
# # plt.legend(fontsize="15")
# # plt.tight_layout()
# # plt.savefig("antQ_marl_rnn_comparison_training_time.png") 
# # plt.show()
