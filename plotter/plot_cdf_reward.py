"""
    Script for plotting the CDF of the reward obtained per episode
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from config import FrameworkConfiguration

from plotter.plot_moving_avg import print_cute_algo_name

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

output_dir = './'


def fix_hist_step_vertical_line_at_end(ax):
    """
    Support function to adjust layout of plots
    """
    ax_polygons = [poly for poly in ax.get_children() if isinstance(poly, patches.Polygon)]
    for poly in ax_polygons:
        poly.set_xy(poly.get_xy()[:-1])


def compute_avg_reward_single_algo_multiple_runs(date_array, algorithm=None):
    x_all = []
    y_all_avg_rewards = []

    x = []
    y_avg_reward_for_one_episode = []
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
        y_avg_reward_for_one_episode = []
        with open(directory + '/' + filename, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            next(reader, None)
            for row in reader:
                x.append(int(row[0]))
                # TO COMPUTE OVER NUMBER OF COMMANDS
                # OTHERWISE REMOVE DIVISION BY ROW 3
                y_avg_reward_for_one_episode.append(float(row[1]) / float(row[3]))
        x_all.append(x)
        y_all_avg_rewards.append(y_avg_reward_for_one_episode)

    data = []
    fig, ax = plt.subplots()
    for i in range(0, len(x_all)):
        # plt.plot(episodes_target[i], avg_rew[i], label=algorithms_target[i], color=color[i])
        # First sorting the array
        plt.hist(np.sort(y_all_avg_rewards[i]), density=True, cumulative=True, label='CDF-run ' + str(i), bins=2000,
                 histtype='step', alpha=0.8)
        fix_hist_step_vertical_line_at_end(ax)

        # data.append(("run"+str(i), y_all_avg_rewards[i]))
    # fastplot.plot(data, 'CDF_PROVA.png', mode='CDF_multi', xlabel='Reward for algorithm ' + algorithm, legend=True,)

    plt.xlabel('Reward')
    plt.ylabel('CDF (Episode)')
    plt.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # plt.title('CDF of avg reward per sent command ' + algorithm)
    plt.ylim(0, 1.0)
    plt.grid(True, color='gray', linestyle='dashed')
    plt.tight_layout()
    plt.savefig('cdf_rewards_multiple_run_' + algorithm + '.png')
    plt.show()

    # compute average over multiple runs
    y_final_avg_rewards = []

    for array_index in range(0, len(x_all[0])):
        sum_r = 0
        count = 0
        for date_index in range(0, len(date_array)):  # compute average
            sum_r += y_all_avg_rewards[date_index][array_index]
            count += 1
        y_final_avg_rewards.append(sum_r / float(count))

    df_final_avg_over_n_runs = pd.DataFrame({'x': x_all[0], 'y1': y_final_avg_rewards})

    # ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])
    # color = ('#77FF82', '#47CC99', '#239DBA', '#006586')
    i = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"].index(algorithm)

    # plot results
    pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y1'],
            label="avg over " + str(len(date_array)) + " run")  # avg line

    pl.xlabel('Episodes')
    pl.ylabel('Avg reward obtained per episode')
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Reward for ' + algorithm + ' algorithm over episodes')
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig('avg_reward_plot_multiple_runs.png')
    # plt.show()
    plt.close()

    return algorithm, x_all[0], y_final_avg_rewards, y_all_avg_rewards


def plot_cdf_reward_multiple_algo(algorithms_target, episodes_target, avg_rew, path):
    target_output_dir = output_dir
    if path in [1, 2, 3]:
        target_output_dir = "../plot/path" + str(path) + "/"

    fig, ax = plt.subplots()

    for i in range(0, len(algorithms_target)):
        # plt.plot(episodes_target[i], avg_rew[i], label=algorithms_target[i],)
        # First sorting the array
        plt.hist(np.sort(avg_rew[i]), density=True, cumulative=True, label=print_cute_algo_name(algorithms_target[i]), bins=2000,
                 histtype='step', alpha=0.8)
        fix_hist_step_vertical_line_at_end(ax)

    plt.xlabel('Reward')
    plt.ylabel('CDF (Episode)')
    plt.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # plt.title('CDF of avg reward per sent command')
    plt.grid(True, color='gray', linestyle='dashed')
    plt.tight_layout()
    plt.ylim(0, 1.0)
    plt.savefig(target_output_dir + 'cdf_rewards_multiple_algo.png')
    plt.show()


def plot_cdf_unique_path(path=None):
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    algos = []
    episodes = []
    avg_rewards = []

    from date_for_graphs_before_tuning_path2 import sarsa
    from date_for_graphs_before_tuning_path2 import sarsa_lambda
    from date_for_graphs_before_tuning_path2 import qlearning
    from date_for_graphs_before_tuning_path2 import qlearning_lambda

    # SARSA
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=sarsa, algorithm="sarsa")

    algos.append(al)
    episodes.append(ep)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # SARSA(lambda)
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=sarsa_lambda,
                                                                          algorithm="sarsa_lambda")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # Q-learning
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning, algorithm="qlearning")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # Q(lambda)
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning_lambda,
                                                                          algorithm="qlearning_lambda")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    plot_cdf_reward_multiple_algo(algos, episodes, avg_rewards, None)


def plot_cdf_path_from_dates(sarsa, sarsa_lambda, qlearning, qlearning_lambda, path=None):
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    algos = []
    episodes = []
    avg_rewards = []

    # SARSA
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=sarsa, algorithm="sarsa")

    algos.append(al)
    episodes.append(ep)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # SARSA(lambda)
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=sarsa_lambda,
                                                                          algorithm="sarsa_lambda")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # Q-learning
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning, algorithm="qlearning")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # Q(lambda)
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning_lambda,
                                                                          algorithm="qlearning_lambda")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    plot_cdf_reward_multiple_algo(algos, episodes, avg_rewards, path)


if __name__ == '__main__':
    plot_cdf_unique_path()

    target_path = 1
    print("PATH ", target_path)
    from date_for_graphs_path1 import sarsa_dates
    from date_for_graphs_path1 import sarsa_lambda_dates
    from date_for_graphs_path1 import qlearning_dates
    from date_for_graphs_path1 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    target_path = 2
    print("PATH ", target_path)

    from date_for_graphs_path2 import sarsa_dates
    from date_for_graphs_path2 import sarsa_lambda_dates
    from date_for_graphs_path2 import qlearning_dates
    from date_for_graphs_path2 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

    target_path = 3
    print("PATH ", target_path)

    from date_for_graphs_path3 import sarsa_dates
    from date_for_graphs_path3 import sarsa_lambda_dates
    from date_for_graphs_path3 import qlearning_dates
    from date_for_graphs_path3 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)


# 1 sarsa, 2 sarsa_lambda, 3 qlearning, 4 qlearning_lambda
