import matplotlib.pyplot as plt
import csv

x = []
y_reward = []
y_cum_reward = []
y_timesteps = []

date = '00_52_15_01_10_2020'  # Date must be in format %Y_%m_%d_%H_%M_%S

directory = 'output_Q_parameters'
file_parameters = 'output_parameters_' + date + '.csv'

with open(directory + '/' + file_parameters, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

algorithm = parameters['algorithm_used']
print("The RL algorithm used is", algorithm)


directory = 'output_csv'
filename = 'output_' + algorithm + '_' + date + '.csv'
separate_plots = False

with open(directory+'/'+filename, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    for row in reader:
        x.append(int(row[0]))
        y_reward.append(int(row[1]))
        y_cum_reward.append(int(row[2]))
        y_timesteps.append(int(row[3]))

if separate_plots:
    plt.plot(x, y_reward, 'k--', label='rew')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward per episodes for ' + algorithm + ' algorithm')
    plt.legend(loc="center right")
    plt.grid(True)

    plt.show()

    plt.plot(x, y_cum_reward, 'k:', label='cum_rew')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Cumulative reward over episodes for ' + algorithm + ' algorithm')
    plt.legend(loc="center right")
    plt.grid(True)

    plt.show()

    plt.plot(x, y_timesteps, 'k', label='timesteps', marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Timesteps')
    plt.title('Timesteps per episode for ' + algorithm + ' algorithm')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.show()
else:
    plt.subplot(3, 1, 1)
    plt.plot(x, y_reward, 'k--', label='rew')
    plt.ylabel('Reward')
    plt.title('Statistics per episode for ' + algorithm + ' algorithm')
    plt.legend(loc="center right")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x, y_cum_reward, 'k:', label='cum_rew')
    plt.ylabel('Cumulative reward')
    plt.legend(loc="center right")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x, y_timesteps, 'k', label='timesteps', marker='o')
    plt.xlabel('Episodes')
    plt.ylabel('Timesteps')
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)

    plt.show()

# TODO anche per decidere l'algoritmo potrei usare una enum (1 sarsa, 2 sarsa_lambda, 3 qlearning)
