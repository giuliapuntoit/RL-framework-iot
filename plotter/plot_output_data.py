"""
    Class for plotting reward and timesteps over episodes for 1 single execution of RL algorithm
"""

import matplotlib.pyplot as plt
import csv
from config import FrameworkConfiguration
plt.rcParams["font.family"] = "Times New Roman"


class PlotOutputData(object):
    # Plot results from a single run
    def __init__(self, date_to_retrieve='YY_mm_dd_HH_MM_SS', separate_plots=False):
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            print("Invalid date")
            exit(1)
        self.separate_plots = separate_plots

    def run(self):
        x = []
        y_reward = []
        y_cum_reward = []
        y_timesteps = []

        directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
        file_parameters = 'output_parameters_' + self.date_to_retrieve + '.csv'

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        algorithm = parameters['algorithm_used']
        print("RL ALGORITHM:", algorithm)
        print("PLOTTING GRAPHS...")

        directory = FrameworkConfiguration.directory + 'output/output_csv'
        filename = 'output_' + algorithm + '_' + self.date_to_retrieve + '.csv'

        with open(directory + '/' + filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None)
            for row in reader:
                x.append(int(row[0]))
                y_reward.append(int(row[1]))
                y_cum_reward.append(int(row[2]))
                y_timesteps.append(int(row[3]))

        if self.separate_plots:
            plt.plot(x, y_reward, 'k', label='rew')
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

            plt.plot(x, y_timesteps, 'k', label='timesteps')  # marker='o')
            plt.xlabel('Episodes')
            plt.ylabel('Timesteps')
            plt.title('Timesteps per episode for ' + algorithm + ' algorithm')
            plt.legend(loc="upper right")
            plt.grid(True)

            plt.show()
        else:
            plt.subplot(2, 1, 1)
            plt.plot(x, y_reward, 'k', label='rew')
            plt.ylabel('Reward')
            plt.title('Statistics per episode for ' + algorithm + ' algorithm')
            # plt.legend(loc="center right")
            plt.grid(True)

            # plt.subplot(3, 1, 2)
            # plt.plot(x, y_cum_reward, 'k:', label='cum_rew')
            # plt.ylabel('Cumulative reward')
            # plt.legend(loc="center right")
            # plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(x, y_timesteps, 'k', label='timesteps')
            plt.xlabel('Episodes')
            plt.ylabel('Timesteps')
            # plt.legend(loc="upper right")
            plt.grid(True)

            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)

            plt.show()
            print("Done.")


if __name__ == '__main__':
    PlotOutputData(date_to_retrieve='2020_11_21_11_09_13', separate_plots=False).run()
