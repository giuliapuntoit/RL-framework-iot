import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fastplot
from config import GlobalVar


class PlotOutputData(object):
    # Plot reward and timesteps over episodes for multiple executions
    # Deprecated!
    def __init__(self, date_to_retrieve='YY_mm_dd_HH_MM_SS', algos=[], paths=[], params=[], separate_plots=False):
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            print("Invalid date")
            exit(1)
        self.separate_plots = separate_plots
        self.algos = algos
        self.paths = paths
        self.params = params

    def run(self):

        for dat in self.algos:
            x = []
            y_reward = []
            y_cum_reward = []
            y_timesteps = []
            self.date_to_retrieve = dat
            directory = GlobalVar.directory + 'output/output_Q_parameters'
            file_parameters = 'output_parameters_' + self.date_to_retrieve + '.csv'

            with open(directory + '/' + file_parameters, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

            algorithm = parameters['algorithm_used']
            print("RL ALGORITHM:", algorithm)

            directory = GlobalVar.directory + 'output/output_csv'
            filename = 'output_' + algorithm + '_' + self.date_to_retrieve + '.csv'

            with open(directory + '/' + filename, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                next(reader, None)
                for row in reader:
                    x.append(int(row[0]))
                    y_reward.append(int(row[1]))
                    y_cum_reward.append(int(row[2]))
                    y_timesteps.append(int(row[3]))

        #     plt.plot(x, y_reward, label=algorithm)
        #     plt.xlabel('Episodes')
        #     plt.ylabel('Reward')
        #     plt.title('Reward per algorithm')
        #     plt.legend()
        # plt.savefig('0_algos.png')

        #     plt.plot(x, y_timesteps, label=algorithm)
        #     plt.xlabel('Episodes')
        #     plt.ylabel('Timesteps')
        #     plt.title('Timesteps per algorithm')
        #     plt.legend()
        #
        # plt.savefig('0_line_timesteps.png')

        # x = range(11)
        # y = [4, 150, 234, 465, 745, 612, 554, 43, 565, 987, 154]
        # fastplot.plot((x, y), '1_line.png', xlabel='X', ylabel='Y')

        # fastplot.plot((x, y_timesteps), '1_line.png', xlabel='Episodes', ylabel='Timesteps', style="latex")
        # fastplot.plot((x, y_reward), '2_line.png', xlabel='Episodes', ylabel='Reward', style="latex")

        # print("Done.")

        for index, dat in enumerate(self.paths):
            x = []
            y_reward = []
            y_cum_reward = []
            y_timesteps = []
            self.date_to_retrieve = dat
            directory = GlobalVar.directory + 'output/output_Q_parameters'
            file_parameters = 'output_parameters_' + self.date_to_retrieve + '.csv'

            with open(directory + '/' + file_parameters, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

            algorithm = parameters['algorithm_used']
            print("RL algorithm used:", algorithm)

            directory = GlobalVar.directory + 'output/output_csv'
            filename = 'output_' + algorithm + '_' + self.date_to_retrieve + '.csv'

            with open(directory + '/' + filename, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                next(reader, None)
                for row in reader:
                    x.append(int(row[0]))
                    y_reward.append(int(row[1]))
                    y_cum_reward.append(int(row[2]))
                    y_timesteps.append(int(row[3]))

            # plt.plot(x, y_reward, label="path"+str(index))
            # plt.xlabel('Episodes')
            # plt.ylabel('Reward')
            # plt.title('Reward per path')
            # plt.legend()

            # plt.plot(x, y_timesteps, label=algorithm)
            # plt.xlabel('Episodes')
            # plt.ylabel('Timesteps')
            # plt.title('Timesteps per algorithm')
            # plt.legend()

        # plt.savefig('0_paths.png')

        episodes = []
        rewards = []
        times = []
        timesteps = []

        for index, param in enumerate(self.params):
            x_p = []
            y_reward_p = []
            y_time_p = []
            y_timesteps_p = []
            self.date_to_retrieve = param
            directory = GlobalVar.directory + 'output/output_Q_parameters'
            file_parameters = 'output_parameters_' + self.date_to_retrieve + '.csv'

            with open(directory + '/' + file_parameters, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

            algorithm = parameters['algorithm_used']
            print("RL ALGORITHM:", algorithm)

            directory = GlobalVar.directory + 'output/output_csv'
            filename = 'partial_output_' + algorithm + '_' + self.date_to_retrieve + '.csv'

            with open(directory + '/' + filename, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                next(reader, None)
                for row in reader:
                    x_p.append(int(row[0]))
                    y_timesteps_p.append(int(row[1]))
                    y_reward_p.append(int(row[2]))
                    y_time_p.append(float(row[3]))

            episodes.append(x_p)
            timesteps.append(y_timesteps_p)
            rewards.append(y_reward_p)
            times.append(y_time_p)

            # plt.plot(x_p, y_reward_p, label="epsilon" + parameters['epsilon'])
            # plt.xlabel('Episodes')
            # plt.ylabel('Reward')
            # plt.title('Reward per epsilon')
            # plt.legend()

            # plt.plot(x, y_timesteps, label=algorithm)
            # plt.xlabel('Episodes')
            # plt.ylabel('Timesteps')
            # plt.title('Timesteps per algorithm')
            # plt.legend()

        # plt.savefig(GlobalVar.directory + 'plot/0_epsilon.png')
        # print(np.matrix(rewards).transpose())

        data = pd.DataFrame(np.matrix(timesteps).transpose(),
                            index=episodes[0],
                            columns=['0.3', '0.6', '0.9'], )
        fastplot.plot(data, GlobalVar.directory + 'plot/0_epsilon_bars_timesteps.png', mode='bars_multi', style='latex',
                      ylabel='Timesteps', legend=True, ylim=(0, 30), legend_ncol=3,
                      legend_args={'markerfirst': True})

        data = pd.DataFrame(np.matrix(rewards).transpose(),
                            index=episodes[0],
                            columns=['0.3', '0.6', '0.9'], )
        fastplot.plot(data, GlobalVar.directory + 'plot/0_epsilon_bars_rew.png', mode='bars_multi', style='latex',
                      ylabel='Timesteps', legend=True, legend_ncol=3,
                      legend_args={'markerfirst': True})


if __name__ == '__main__':
    algos = ['2020_10_26_01_51_42',
             '2020_10_26_18_55_47',
             '2020_10_26_07_57_37', ]
    paths = ['2020_10_26_01_51_42', '2020_10_30_02_10_16', '2020_10_30_11_23_33']  # with sarsa
    all_params = ['2020_10_28_01_32_46',
                  '2020_10_28_02_45_17',
                  '2020_10_28_03_54_08',
                  '2020_10_28_05_15_51',
                  '2020_10_28_06_32_32',
                  '2020_10_28_07_37_36',
                  '2020_10_28_08_57_07',
                  '2020_10_28_10_15_57',
                  '2020_10_28_12_05_00',
                  '2020_10_28_13_26_21',
                  '2020_10_28_14_49_55',
                  '2020_10_28_16_13_06',
                  '2020_10_28_17_35_23',
                  '2020_10_28_18_54_14',
                  '2020_10_28_20_42_49',
                  '2020_10_28_22_07_18',
                  '2020_10_28_23_57_19',
                  '2020_10_29_01_22_44',
                  '2020_10_29_02_53_47',
                  '2020_10_29_04_47_06',
                  '2020_10_29_06_50_02',
                  '2020_10_29_09_10_05',
                  '2020_10_29_11_03_54',
                  '2020_10_29_13_21_46',
                  '2020_10_29_15_23_32',
                  '2020_10_29_17_23_34',
                  '2020_10_29_19_46_10',
                  '2020_10_29_23_18_53',
                  '2020_10_29_23_59_49',
                  '2020_10_30_00_40_49', ]
    params = ['2020_10_28_03_54_08', '2020_10_28_16_13_06', '2020_10_29_06_50_02']
    PlotOutputData(date_to_retrieve='2020_10_28_03_54_08', algos=algos, paths=paths, params=params,
                   separate_plots=False).run()
