"""
    Script for plotting the moving average of the reward and timesteps over episodes for different algorithms
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib.font_manager import FontProperties
from config import FrameworkConfiguration

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

    directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
    file_parameters = 'output_parameters_' + date_to_retrieve + '.csv'

    with open(directory + '/' + file_parameters, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

    algorithm = parameters['algorithm_used']
    print("RL ALGORITHM:", algorithm)
    print("PLOTTING GRAPHS...")

    directory = FrameworkConfiguration.directory + 'output/output_csv'
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
            directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
            file_parameters = 'output_parameters_' + dat + '.csv'

            with open(directory + '/' + file_parameters, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

            algorithm = parameters['algorithm_used']
        print("RL ALGORITHM:", algorithm)

        directory = FrameworkConfiguration.directory + 'output/output_csv'
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

    global_avg_reward = np.mean(y_final_reward)
    global_avg_timesteps = np.mean(y_final_timesteps)

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


def plot_single_algo_multiple_runs_for_avg_bars(date_array, algorithm=None):
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
            directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
            file_parameters = 'output_parameters_' + dat + '.csv'

            with open(directory + '/' + file_parameters, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

            algorithm = parameters['algorithm_used']
        print("RL ALGORITHM:", algorithm)

        directory = FrameworkConfiguration.directory + 'output/output_csv'
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

    global_avg_reward = np.mean(y_final_reward)
    global_avg_timesteps = np.mean(y_final_timesteps)

    return global_avg_reward, global_avg_timesteps


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

    from date_for_graphs_before_tuning_path2 import sarsa
    from date_for_graphs_before_tuning_path2 import sarsa_lambda
    from date_for_graphs_before_tuning_path2 import qlearning
    from date_for_graphs_before_tuning_path2 import qlearning_lambda

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


def plot_multiple_algos_rewards_timesteps(algos, avg_rew, avg_steps, path):
    target_output_dir = output_dir
    if path in [1, 2, 3]:
        target_output_dir = "../plot/path" + str(path) + "/"
    fig, ax = plt.subplots()
    cols_labels = []
    for al in algos:
        cols_labels.append(print_cute_algo_name(al))
    col = ax.bar(cols_labels, avg_rew, align='center')

    ax.set_ylabel('Avg reward for episode')
    # ax.set_title('Avg reward for algos')
    plt.axhline(0, color='black', lw=.3)
    fig.tight_layout()
    plt.savefig(target_output_dir + 'avg_rewards_for_algos.png')
    plt.show()

    fig, ax = plt.subplots()
    col = ax.bar(cols_labels, avg_steps, align='center', )
    ax.set_ylabel('Avg time steps for episode')
    fig.tight_layout()
    plt.savefig(target_output_dir + 'avg_steps_for_algos.png')
    plt.show()


if __name__ == '__main__':
    all_algo = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 1
    print("PATH ", target_path)

    from date_for_graphs_path1 import sarsa_dates
    from date_for_graphs_path1 import sarsa_lambda_dates
    from date_for_graphs_path1 import qlearning_dates
    from date_for_graphs_path1 import qlearning_lambda_dates

    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_all_paths(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_rewards_timesteps(all_algo, all_avg_rew, all_avg_timesteps, target_path)

    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 2
    print("PATH ", target_path)

    from date_for_graphs_path2 import sarsa_dates
    from date_for_graphs_path2 import sarsa_lambda_dates
    from date_for_graphs_path2 import qlearning_dates
    from date_for_graphs_path2 import qlearning_lambda_dates

    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_all_paths(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_rewards_timesteps(all_algo, all_avg_rew, all_avg_timesteps, target_path)

    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 3
    print("PATH ", target_path)

    from date_for_graphs_path3 import sarsa_dates
    from date_for_graphs_path3 import sarsa_lambda_dates
    from date_for_graphs_path3 import qlearning_dates
    from date_for_graphs_path3 import qlearning_lambda_dates

    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = plot_single_algo_multiple_runs_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_all_paths(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_rewards_timesteps(all_algo, all_avg_rew, all_avg_timesteps, target_path)
