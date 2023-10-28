from agent import Agent
from sensor_network import Network

def main():
    network = Network().build_network()
    agent_can_revisit = Agent(alpha=1e-5, n_actions=network.num_node, placeholder=network.placeholder, can_revisit=True, name='agent_can_revisit')
    agent_no_revisit = Agent(alpha=1e-5, n_actions=network.num_node, placeholder=network.placeholder, can_revisit=False, name='agent_no_revisit')

    n_games = 10

    file_name = 'tsp_agent_comparison_revisit_vs_non-revisit_no-epsilon_1000ep.png'
    figure_file = 'plots/'+file_name
    agents = [agent_can_revisit, agent_no_revisit]


    total_reward_history = []
    paths = []

    for i in range(n_games):
        game_path = []
        game_reward = []
        for agent in agents:
            print("==== Game: {} ====\nAgent: {}\n".format(i+1,agent.name))
            network.reset()
            state = network.state_representation()
            agent_path = []
            paths.append(state[-1])

            done = False
            agent_reward = 0 
            round_ = 1
            while not done:
                if round_ > 100:
                    break
                
                action = agent.select_action(state)
        #         print("action: {}".format(action))
                next_state, reward, done = network.visit(action)
                paths.append(next_state[-1])
        #         print(next_state, reward, done)
                agent_reward += reward
                print("Round: {}\nCurrent City: {}\nAction: {}\nReward: {}\nTotal Reward:{}\n".format(round_, state[-1], action, reward, agent_reward))

                agent.learn(state, reward, next_state, done)
                state = next_state
                round_ += 1

            game_path.append(agent_path)
            game_reward.append(agent_reward)
            
        total_reward_history.append(game_reward)
        paths.append(game_path)

if __name__ == '__main__':
    main()