import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

from config import GlobalVar

target_output_dir = './'


def get_reward_for_episode(curr_date="2020_12_15_00_19_48", algorithm="qlearning"):
    if algorithm is None:
        directory = GlobalVar.directory + 'output/output_Q_parameters'
        file_parameters = 'output_parameters_' + curr_date + '.csv'

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        algorithm = parameters['algorithm_used']
    print("RL ALGORITHM:", algorithm)

    directory = GlobalVar.directory + 'output/output_csv'
    filename = 'output_' + algorithm + '_' + curr_date + '.csv'

    x = []
    y_reward = []
    y_cum_reward = []
    y_timesteps = []
    with open(directory + '/' + filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader, None)
        for row in reader:
            x.append(int(row[0]))
            y_reward.append(int(row[1]))
            y_cum_reward.append(int(row[2]))
            y_timesteps.append(int(row[3]))

    return x, y_reward


def animate(k):
    x, y = get_reward_for_episode()
    plt.cla()
    plt.plot(x, y, 'k')  # single line
    plt.xlabel('Episode')
    plt.ylabel('Final reward')
    plt.grid(True, color='gray', linestyle='dashed')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=20000)

plt.tight_layout()
plt.show()
