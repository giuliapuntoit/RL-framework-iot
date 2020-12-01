import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib.font_manager import FontProperties
from config import GlobalVar

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

output_dir = './'


# Functions for plotting the moving average for multiple runs and multiple algorithms


def print_cute_algo_name(a):
    if a == "sarsa":
        return "SARSA"
    elif a == "sarsa_lambda":
        return "SARSA(λ)"
    elif a == "qlearning":
        return "Q-learning"
    elif a == "qlearning_lambda":
        return "Q(λ)"
    else:
        return "invalid"


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
    # plt.title('Cum reward per algorithm')
    plt.grid(True, color='gray', linestyle='dashed')
    ax.set_xlim(xmin=0)
    plt.legend(loc='lower right', prop=fontP, ncol=n_cols)

    plt.show()

    window_size = 10

    # calculate the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(y_reward, weights, 'valid')
    # plot results
    pl.plot(x[np.array(x).shape[0] - yMA.shape[0]:], yMA, 'r', label='MA')
    pl.plot(x, df['y1'], 'y--', label='data')
    pl.xlabel('Time')
    pl.ylabel('y')
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Moving Average with window size = ' + str(window_size))
    pl.grid(True, color='gray', linestyle='dashed')
    pl.show()

    print("Done.")


def plot_single_algo_multiple_runs(date_array, algorithm=None, path=None):
    target_output_dir = output_dir
    if path in [1, 2, 3]:
        target_output_dir = "../plot/path" + str(path) + "/"

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
    yMA = np.convolve(df_final_avg_over_n_runs['y1'], weights, 'valid')

    # plot results
    pl.plot(df_single_run['x'], df_single_run['y1'], ':', label='1 run', color="grey")  # single line
    pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y1'], 'k',
            label=str(len(date_array)) + " runs avg")  # avg line
    pl.plot(df_final_avg_over_n_runs['x'][np.array(df_final_avg_over_n_runs['x']).shape[0] - yMA.shape[0]:], yMA, 'r',
            label=str(len(date_array)) + ' runs moving avg')  # moving avg line

    pl.xlabel('Episode')
    pl.ylabel('Final reward')
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Final reward for ' + algorithm + ' algorithm over episodes')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(target_output_dir + 'all_reward_plot_' + algorithm + '.png')
    plt.show()

    yMA_timesteps = np.convolve(df_final_avg_over_n_runs['y2'], weights, 'valid')

    # plot results
    pl.plot(df_single_run['x'], df_single_run['y2'], ':', label='1 run', color="grey")  # single line
    pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y2'], 'k',
            label=str(len(date_array)) + " runs avg")  # avg line
    pl.plot(df_final_avg_over_n_runs['x'][np.array(df_final_avg_over_n_runs['x']).shape[0] - yMA_timesteps.shape[0]:],
            yMA_timesteps, 'r',
            label=str(len(date_array)) + ' runs moving avg')  # moving avg line

    pl.xlabel('Episode')
    pl.ylabel('Number of time steps')
    pl.legend(loc='upper right', prop=fontP, ncol=n_cols)
    # pl.title('Time steps for ' + algorithm + ' algorithm over episodes')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()

    plt.savefig(target_output_dir + 'all_timesteps_plot_' + algorithm + '.png')
    plt.show()

    return algorithm, x, yMA, yMA_timesteps


def plot_multiple_algo_moving_avg(algorithms_target, episodes_target, moving_average_rewards_target,
                                  moving_average_timesteps_target, path=None):
    target_output_dir = output_dir
    if path in [1, 2, 3]:
        target_output_dir = "../plot/path" + str(path) + "/"

    color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    for i in range(0, len(algorithms_target)):
        pl.plot(episodes_target[i][
                np.array(episodes_target[i]).shape[0] - np.array(moving_average_rewards_target[i]).shape[0]:],
                moving_average_rewards_target[i],
                label=print_cute_algo_name(algorithms_target[i]))

    pl.xlabel('Episode')
    pl.ylabel('Final reward')
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Moving average of final reward over episodes')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(target_output_dir + 'mavg_reward_plot.png')
    plt.show()

    for i in range(0, len(algorithms_target)):
        pl.plot(episodes_target[i][
                np.array(episodes_target[i]).shape[0] - np.array(moving_average_timesteps_target[i]).shape[0]:],
                moving_average_timesteps_target[i],
                label=print_cute_algo_name(algorithms_target[i]))

    pl.xlabel('Episode')
    pl.ylabel('Number of time steps')
    pl.legend(loc='upper right', prop=fontP, ncol=n_cols)
    # pl.title('Moving average of number of time steps over episodes')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(target_output_dir + 'mavg_timesteps_plot.png')
    plt.show()


def all_graphs_before_tuning():
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


def all_graphs_all_paths(sarsa, sarsa_lambda, qlearning, qlearning_lambda, path=None):
    algos = []
    episodes = []
    moving_avgs_rewards = []
    moving_avgs_timesteps = []

    # SARSA
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=sarsa, algorithm="sarsa", path=path)

    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    # SARSA(lambda)
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=sarsa_lambda, algorithm="sarsa_lambda", path=path)

    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    # Q-learning
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=qlearning, algorithm="qlearning", path=path)

    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    # Q(lambda)
    al, ep, ma, mats = plot_single_algo_multiple_runs(date_array=qlearning_lambda, algorithm="qlearning_lambda",
                                                      path=path)
    algos.append(al)
    episodes.append(ep)
    moving_avgs_rewards.append(ma)
    moving_avgs_timesteps.append(mats)

    plot_multiple_algo_moving_avg(algos, episodes, moving_avgs_rewards, moving_avgs_timesteps, path=path)


if __name__ == '__main__':
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    # plot_single_algo_single_run(date_to_retrieve='2020_11_05_03_27_46')
    # all_graphs_before_tuning()

    target_path = 1
    print("PATH ", target_path)
    sarsa_dates = [
        '2020_11_25_12_48_15',
        '2020_11_25_13_37_48',
        '2020_11_25_14_29_17',
        '2020_11_25_15_35_12',
        '2020_11_25_16_36_56',
        '2020_11_25_17_24_57',
        '2020_11_25_18_14_50',
        '2020_11_25_19_16_07',
        '2020_11_25_20_05_06',
        '2020_11_25_21_03_28',
    ]

    sarsa_lambda_dates = [
        '2020_11_25_21_54_44',
        '2020_11_25_22_49_27',
        '2020_11_25_23_40_38',
        '2020_11_26_00_37_06',
        '2020_11_26_01_30_01',
        '2020_11_26_02_25_21',
        '2020_11_26_03_22_27',
        '2020_11_26_04_12_40',
        '2020_11_26_05_05_32',
        '2020_11_26_05_52_05',
    ]

    qlearning_dates = [
        '2020_11_22_02_19_21',
        '2020_11_22_03_29_10',
        '2020_11_22_04_18_21',
        '2020_11_22_05_13_56',
        '2020_11_22_06_10_46',
        '2020_11_22_07_24_00',
        '2020_11_22_08_21_59',
        '2020_11_22_09_36_02',
        '2020_11_22_10_25_36',
        '2020_11_22_11_14_52',
    ]

    qlearning_lambda_dates = [
        '2020_11_22_12_02_03',
        '2020_11_22_12_50_14',
        '2020_11_22_13_55_12',
        '2020_11_22_14_52_53',
        '2020_11_22_15_45_11',
        '2020_11_22_16_39_57',
        '2020_11_22_17_35_14',
        '2020_11_22_18_43_39',
        '2020_11_22_19_49_24',
        '2020_11_22_20_46_00',
    ]
    all_graphs_all_paths(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    target_path = 2
    print("PATH ", target_path)

    sarsa_dates = [
        '2020_11_24_15_03_34',
        '2020_11_24_15_51_12',
        '2020_11_24_16_51_27',
        '2020_11_24_17_45_45',
        '2020_11_24_18_38_37',
        '2020_11_24_19_34_52',
        '2020_11_24_20_26_29',
        '2020_11_24_21_19_02',
        '2020_11_24_22_12_30',
        '2020_11_24_23_03_13',
    ]

    sarsa_lambda_dates = [
        '2020_11_25_00_05_17',
        '2020_11_25_01_00_18',
        '2020_11_25_02_01_42',
        '2020_11_25_03_02_56',
        '2020_11_25_03_55_24',
        '2020_11_25_04_47_54',
        '2020_11_25_05_45_45',
        '2020_11_25_06_46_58',
        '2020_11_25_07_39_08',
        '2020_11_25_08_34_32',
    ]

    qlearning_dates = [
        '2020_11_19_23_20_35',
        '2020_11_20_00_45_39',
        '2020_11_20_01_43_25',
        '2020_11_20_02_58_24',
        '2020_11_20_04_02_19',
        '2020_11_20_04_57_58',
        '2020_11_20_05_52_01',
        '2020_11_20_06_51_41',
        '2020_11_20_07_50_17',
        '2020_11_20_08_48_08',
    ]

    qlearning_lambda_dates = [
        '2020_11_20_09_41_30',
        '2020_11_20_10_36_29',
        '2020_11_20_11_27_28',
        '2020_11_20_12_27_46',
        '2020_11_20_13_35_56',
        '2020_11_20_14_39_49',
        '2020_11_20_15_45_16',
        '2020_11_20_16_43_07',
        '2020_11_20_17_39_35',
        '2020_11_20_18_44_51',
    ]

    all_graphs_all_paths(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

    target_path = 3
    print("PATH ", target_path)

    sarsa_dates = [
        '2020_11_22_22_20_40',
        '2020_11_22_23_44_50',
        '2020_11_23_01_14_07',
        '2020_11_23_03_01_07',
        '2020_11_23_04_33_00',
        '2020_11_23_05_50_31',
        '2020_11_23_07_13_52',
        '2020_11_23_08_35_52',
        '2020_11_23_09_57_28',
        '2020_11_23_11_36_57',
    ]

    sarsa_lambda_dates = [
        '2020_11_23_18_01_20',
        '2020_11_23_19_27_16',
        '2020_11_23_20_40_12',
        '2020_11_23_21_54_42',
        '2020_11_23_23_24_23',
        '2020_11_24_00_41_00',
        '2020_11_24_02_15_22',
        '2020_11_24_03_27_53',
        '2020_11_24_04_40_32',
        '2020_11_24_05_59_40',
    ]

    qlearning_dates = [
        '2020_11_20_19_55_08',
        '2020_11_20_21_25_32',
        '2020_11_20_22_52_02',
        '2020_11_21_00_14_24',
        '2020_11_21_01_42_09',
        '2020_11_21_03_01_06',
        '2020_11_21_04_21_03',
        '2020_11_21_05_36_05',
        '2020_11_21_07_01_27',
        '2020_11_21_08_22_04',
    ]

    qlearning_lambda_dates = [
        '2020_11_21_09_39_37',
        '2020_11_21_11_09_13',
        '2020_11_21_12_30_19',
        '2020_11_21_13_58_09',
        '2020_11_21_15_29_01',
        '2020_11_21_16_49_23',
        '2020_11_21_18_13_51',
        '2020_11_21_19_54_03',
        '2020_11_21_21_43_57',
        '2020_11_21_23_07_58',
    ]

    all_graphs_all_paths(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

# 1 sarsa, 2 sarsa_lambda, 3 qlearning, 4 qlearning_lambda
