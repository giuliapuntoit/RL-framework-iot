"""
    Script to generate an automated plot in real time attaching to the execution of the current RL algorithm used
"""

import csv
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

from config import FrameworkConfiguration

target_output_dir = './'


# TODO this code is duplicated i think
def get_reward_for_episode(curr_date="2020_12_15_00_19_48", algorithm="qlearning"):
    """
    Function to read the output of RL algorithms and save the reward value per episode
    """
    if algorithm is None:
        directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
        file_parameters = 'output_parameters_' + curr_date + '.csv'

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        algorithm = parameters['algorithm_used']
    print("RL ALGORITHM:", algorithm)

    directory = FrameworkConfiguration.directory + 'output/output_csv'
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
    """
    Generate the animated plot with real time changes
    """
    x, y = get_reward_for_episode()
    plt.cla()
    plt.plot(x, y, 'k')  # single line
    plt.xlabel('Episode')
    plt.ylabel('Final reward')
    plt.grid(True, color='gray', linestyle='dashed')
    plt.tight_layout()


# Call the animate function, interval is the delay between frames in ms
ani = FuncAnimation(plt.gcf(), animate, interval=20000)

plt.tight_layout()
plt.show()
