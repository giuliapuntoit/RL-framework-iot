import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib import patches
plt.rcParams["font.family"] = "Times New Roman"

from config import GlobalVar


def fix_hist_step_vertical_line_at_end(ax):
    ax_polygons = [poly for poly in ax.get_children() if isinstance(poly, patches.Polygon)]
    for poly in ax_polygons:
        poly.set_xy(poly.get_xy()[:-1])


# Functions for plotting the CDF of the reward


def compute_avg_reward_single_algo_multiple_runs(date_array, algorithm=None):
    x_all = []
    y_all_avg_rewards = []

    x = []
    y_avg_reward_for_one_episode = []
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
        plt.hist(np.sort(y_all_avg_rewards[i]), density=True, cumulative=True, label='CDF-run ' + str(i), bins=1000,
                 histtype='step', alpha=0.8)
        fix_hist_step_vertical_line_at_end(ax)

        # data.append(("run"+str(i), y_all_avg_rewards[i]))
    # fastplot.plot(data, 'CDF_PROVA.png', mode='CDF_multi', xlabel='Reward for algorithm ' + algorithm, legend=True,)

    plt.xlabel('Reward')
    plt.ylabel('CDF (Episode)')
    plt.legend(loc='lower right')
    plt.title('CDF of avg reward per sent command ' + algorithm)
    plt.ylim(0, 1.0)
    plt.grid(True)
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
    color = ('#77FF82', '#47CC99', '#239DBA', '#006586')
    i = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"].index(algorithm)

    # plot results
    pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y1'],
            label="avg over " + str(len(date_array)) + " run", color=color[i])  # avg line

    pl.xlabel('Episodes')
    pl.ylabel('Avg reward obtained per episode')
    pl.legend(loc='lower right')
    pl.title('Reward for ' + algorithm + ' algorithm over episodes')
    pl.grid(True)
    plt.savefig('avg_reward_plot_multiple_runs.png')
    plt.show()
    plt.close()

    return algorithm, x_all[0], y_final_avg_rewards, y_all_avg_rewards


def plot_cdf_reward_multiple_algo(algorithms_target, episodes_target, avg_rew):
    color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    fig, ax = plt.subplots()

    for i in range(0, len(algorithms_target)):
        # plt.plot(episodes_target[i], avg_rew[i], label=algorithms_target[i], color=color[i])
        # First sorting the array
        plt.hist(np.sort(avg_rew[i]), density=True, cumulative=True, label='CDF-' + algorithms_target[i], bins=1000,
                 histtype='step', alpha=0.8, color=color[i])
        fix_hist_step_vertical_line_at_end(ax)

    plt.xlabel('Reward')
    plt.ylabel('CDF (Episode)')
    plt.legend(loc='lower right')
    plt.title('CDF of avg reward per sent command')
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.savefig('cdf_rewards_multiple_algo.png')
    plt.show()


if __name__ == '__main__':
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    algos = []
    episodes = []
    avg_rewards = []

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
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=sarsa, algorithm="sarsa")

    algos.append(al)
    episodes.append(ep)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
        print(tmp_arr)
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
        print(tmp_arr)
    avg_rewards.append(tmp_arr)

    # Q-learning
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning, algorithm="qlearning")

    algos.append(al)
    episodes.append(ep)
    # avg_rewards.append(avgr)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
        print(tmp_arr)
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
        print(tmp_arr)
    avg_rewards.append(tmp_arr)

    plot_cdf_reward_multiple_algo(algos, episodes, avg_rewards)

# 1 sarsa, 2 sarsa_lambda, 3 qlearning, 4 qlearning_lambda
