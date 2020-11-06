import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl

from config import GlobalVar


def plot_single_algo_single_run(date_to_retrieve):
    x = []
    y_reward = []
    y_cum_reward = []
    y_timesteps = []

    directory = GlobalVar.directory + 'output/output_Q_parameters'
    file_parameters = 'output_parameters_' + date_to_retrieve + '.csv'

    with open(directory + '/' + file_parameters, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

    algorithm = parameters['algorithm_used']
    print("RL ALGORITHM:", algorithm)
    print("PLOTTING GRAPHS...")

    directory = GlobalVar.directory + 'output/output_csv'
    filename = 'output_' + algorithm + '_' + date_to_retrieve + '.csv'

    with open(directory + '/' + filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader, None)
        for row in reader:
            x.append(int(row[0]))
            y_reward.append(int(row[1]))
            y_cum_reward.append(int(row[2]))
            y_timesteps.append(int(row[3]))

    df = pd.DataFrame({'x': x, 'y1': y_reward, 'y2': y_timesteps, 'y3': y_cum_reward})

    # ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])
    color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    fig, ax = plt.subplots()

    plt.plot(df['x'], df['y1'], data=None, label="reward", color=color[0])
    plt.plot(df['x'], df['y2'], data=None, label="cum", color=color[1])
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Cum reward per algorithm')
    plt.grid(True)
    ax.set_xlim(xmin=0)
    plt.legend()

    plt.show()

    window_size = 30

    # calculate the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(y_reward, weights, 'full')
    # plot results
    pl.plot(x, yMA[0:np.array(x).shape[0]], 'r', label='MA')
    pl.plot(x, df['y1'], label='data')
    pl.xlabel('Time')
    pl.ylabel('y')
    pl.legend(loc='lower right')
    pl.title('Moving Average with window size = 30')
    pl.grid(True)
    pl.show()

    print("Done.")


def plot_single_algo_multiple_runs(date_array, algorithm=None):
    x_all = []
    y_all_reward = []
    y_all_cum_reward = []
    y_all_timesteps = []

    x = []
    y_reward = []
    y_cum_reward = []
    y_timesteps = []
    # retrieve data for all dates
    for dat in date_array:
        if algorithm is None:
            directory = GlobalVar.directory + 'output/output_Q_parameters'
            file_parameters = 'output_parameters_' + dat + '.csv'

            with open(directory + '/' + file_parameters, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

            algorithm = parameters['algorithm_used']
        print("RL ALGORITHM:", algorithm)

        directory = GlobalVar.directory + 'output/output_csv'
        filename = 'output_' + algorithm + '_' + dat + '.csv'

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
        x_all.append(x)
        y_all_reward.append(y_reward)
        y_all_cum_reward.append(y_cum_reward)
        y_all_timesteps.append(y_timesteps)

    # compute average over multiple runs
    y_final_reward = []
    y_final_cum_reward = []
    y_final_timesteps = []
    for array_index in range(0, len(x_all[0])):
        sum_r = 0
        sum_cr = 0
        sum_t = 0
        count = 0
        for date_index in range(0, len(date_array)):  # compute average
            sum_r += y_all_reward[date_index][array_index]
            sum_cr += y_all_cum_reward[date_index][array_index]
            sum_t += y_all_timesteps[date_index][array_index]
            count += 1
        y_final_reward.append(sum_r / float(count))
        y_final_cum_reward.append(sum_cr / float(count))
        y_final_timesteps.append(sum_t / float(count))

    df_single_run = pd.DataFrame({'x': x, 'y1': y_reward, 'y2': y_timesteps, 'y3': y_cum_reward})
    df_final_avg_over_n_runs = pd.DataFrame(
        {'x': x_all[0], 'y1': y_final_reward, 'y2': y_final_timesteps, 'y3': y_final_cum_reward})

    # ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])
    color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    window_size = 10

    # calculate the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(df_final_avg_over_n_runs['y1'], weights, 'full')
    # plot results
    pl.plot(df_single_run['x'], df_single_run['y1'], label='single run', color=color[0])  # single line
    pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y1'],
            label="avg over " + str(len(date_array)) + " run", color=color[1])  # avg line
    pl.plot(df_final_avg_over_n_runs['x'], yMA[0:np.array(df_final_avg_over_n_runs['x']).shape[0]], 'r',
            label='moving average')  # moving avg line

    pl.xlabel('Episodes')
    pl.ylabel('Reward')
    pl.legend(loc='lower right')
    pl.title('Reward for ' + algorithm + ' algorithm over episodes')
    pl.grid(True)
    plt.savefig('all_reward_plot.png')
    plt.show()

    yMA_timesteps = np.convolve(df_final_avg_over_n_runs['y2'], weights, 'full')
    # plot results
    pl.plot(df_single_run['x'], df_single_run['y2'], label='single run', color=color[0])  # single line
    pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y2'],
            label="avg over " + str(len(date_array)) + " run", color=color[1])  # avg line
    pl.plot(df_final_avg_over_n_runs['x'], yMA_timesteps[0:np.array(df_final_avg_over_n_runs['x']).shape[0]], 'r',
            label='moving average')  # moving avg line

    pl.xlabel('Episodes')
    pl.ylabel('Number of steps')
    pl.legend(loc='lower right')
    pl.title('Steps for ' + algorithm + ' algorithm over episodes')
    pl.grid(True)
    plt.savefig('all_timesteps_plot.png')
    plt.show()

    return algorithm, x, yMA, yMA_timesteps


def plot_multiple_algo_moving_avg(algorithms_target, episodes_target, moving_average_rewards_target,
                                  moving_average_timesteps_target):
    color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    for i in range(0, len(algorithms_target)):
        pl.plot(episodes_target[i], moving_average_rewards_target[i][0:np.array(episodes_target[i]).shape[0]],
                label=algorithms_target[i], color=color[i])

    pl.xlabel('Episodes')
    pl.ylabel('Reward')
    pl.legend(loc='lower right')
    pl.title('Moving average of reward over episodes')
    pl.grid(True)
    plt.savefig('mavg_reward_plot.png')
    plt.show()

    for i in range(0, len(algorithms_target)):
        pl.plot(episodes_target[i], moving_average_timesteps_target[i][0:np.array(episodes_target[i]).shape[0]],
                label=algorithms_target[i], color=color[i])

    pl.xlabel('Episodes')
    pl.ylabel('Number of steps')
    pl.legend(loc='upper right')
    pl.title('Moving average of number of steps over episodes')
    pl.grid(True)
    plt.savefig('mavg_timesteps_plot.png')
    plt.show()


if __name__ == '__main__':
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    plot_single_algo_single_run(date_to_retrieve='2020_11_02_18_42_09')
    algos = []
    episodes = []
    moving_avgs_rewards = []
    moving_avgs_timesteps = []

    sarsa = ['2020_11_05_03_27_46',
             '2020_11_05_04_07_23',
             '2020_11_05_04_48_59',
             '2020_11_05_05_30_35',
             '2020_11_05_06_10_02', ]

    sarsa_lambda = [
        '2020_11_05_06_47_59',
        '2020_11_05_07_33_31',
        '2020_11_05_08_04_47',
        '2020_11_05_08_48_46',
        '2020_11_05_09_35_46', ]

    qlearning = [
        '2020_11_05_10_24_34',
        '2020_11_05_11_05_37',
        '2020_11_05_11_48_23',
        '2020_11_05_12_33_03',
        '2020_11_05_13_16_54', ]

    qlearning_lambda = [
        '2020_11_05_13_54_50',
        '2020_11_05_14_37_02',
        '2020_11_05_15_10_00',
        '2020_11_05_15_49_28',
        '2020_11_05_16_27_15', ]

    # sarsa = ['2020_11_04_21_41_26',
    #          '2020_11_04_21_54_34',
    #          '2020_11_04_22_06_05',
    #          '2020_11_04_22_19_05',
    #          '2020_11_04_22_30_16']
    #
    # sarsa_lambda = ['2020_11_04_22_40_03',
    #                 '2020_11_04_22_54_42',
    #                 '2020_11_04_23_08_01',
    #                 '2020_11_04_23_19_47',
    #                 '2020_11_04_23_29_09']
    #
    # qlearning = ['2020_11_04_23_44_01',
    #              '2020_11_04_23_55_34',
    #              '2020_11_05_00_08_03',
    #              '2020_11_05_00_19_35',
    #              '2020_11_05_00_31_08']
    #
    # qlearning_lambda = ['2020_11_05_00_41_57',
    #                     '2020_11_05_00_53_43',
    #                     '2020_11_05_01_03_24',
    #                     '2020_11_05_01_14_54',
    #                     '2020_11_05_01_24_42']

    # SARSA
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=sarsa, algorithm="sarsa")

    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    # SARSA(lambda)
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=sarsa_lambda, algorithm="sarsa_lambda")

    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    # Q-learning
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=qlearning, algorithm="qlearning")

    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    # Q(lambda)
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=qlearning_lambda, algorithm="qlearning_lambda")
    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    plot_multiple_algo_moving_avg(algos, episodes, moving_avgs_rewards, moving_avgs_timesteps)

# 1 sarsa, 2 sarsa_lambda, 3 qlearning, 4 qlearning_lambda
