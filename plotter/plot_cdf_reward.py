"""
    Script for plotting the CDF of the reward obtained per episode
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib.font_manager import FontProperties
from plotter.plot_moving_avg import print_cute_algo_name
from plotter.support_plotter import fix_hist_step_vertical_line_at_end, read_avg_reward_from_output_file, \
    build_output_dir_from_path, get_font_family_and_size, get_extension

font_family, font_size = get_font_family_and_size()

plt.rcParams["font.family"] = font_family
plt.rcParams['font.size'] = font_size

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

output_dir = './'


def compute_avg_reward_single_algo_multiple_runs(date_array, algorithm=None):
    """
    Compute directly from the output file the average reward per time step for each episode
    """
    x_all = []
    y_all_avg_rewards = []

    # retrieve data for all dates
    for dat in date_array:
        x, y_avg_reward_for_one_episode = read_avg_reward_from_output_file(algorithm, dat)

        x_all.append(x)
        y_all_avg_rewards.append(y_avg_reward_for_one_episode)

    fig, ax = plt.subplots()
    for i in range(0, len(x_all)):
        plt.hist(np.sort(y_all_avg_rewards[i]), density=True, cumulative=True, label='CDF-run ' + str(i), bins=2000,
                 histtype='step', alpha=0.8)
        fix_hist_step_vertical_line_at_end(ax)

    plt.xlabel('Reward')
    plt.ylabel('CDF (Episode)')
    plt.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # plt.title('CDF of avg reward per sent command ' + algorithm)
    plt.ylim(0, 1.0)
    plt.grid(True, color='gray', linestyle='dashed')
    plt.tight_layout()
    # plt.savefig('cdf_rewards_multiple_run_' + algorithm + get_extension())
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
    plt.savefig('avg_reward_plot_multiple_runs' + get_extension())
    # plt.show()
    plt.close()

    return algorithm, x_all[0], y_final_avg_rewards, y_all_avg_rewards


def plot_cdf_reward_multiple_algo(algorithms_target, episodes_target, avg_rew, path):
    """
    Generate plot of the CDF of the average reward for episodes
    """
    target_output_dir = build_output_dir_from_path(output_dir, path)

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
    plt.savefig(target_output_dir + 'cdf_rewards_multiple_algo' + get_extension())
    plt.show()


def plot_cdf_unique_path(path=None):
    """
    Plot CDF of the reward with data before parameter tuning
    All runs refer to path 2
    """
    algos = []
    episodes = []
    avg_rewards = []

    from dates_for_graphs.date_for_graphs_before_tuning_path2 import sarsa
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import sarsa_lambda
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import qlearning
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import qlearning_lambda

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
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # Q-learning
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning, algorithm="qlearning")

    algos.append(al)
    episodes.append(ep)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    # Q(lambda)
    al, ep, avgr, all_avgr = compute_avg_reward_single_algo_multiple_runs(date_array=qlearning_lambda,
                                                                          algorithm="qlearning_lambda")

    algos.append(al)
    episodes.append(ep)
    tmp_arr = []
    for arr in all_avgr:
        tmp_arr = np.concatenate((np.array(tmp_arr), np.array(arr)))
    avg_rewards.append(tmp_arr)

    plot_cdf_reward_multiple_algo(algos, episodes, avg_rewards, None)


def plot_cdf_path_from_dates(sarsa, sarsa_lambda, qlearning, qlearning_lambda, path=None):
    """
    Generate plots for the CDF of the reward for all 4 algorithms
    """
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


def main():
    plot_cdf_unique_path()

    target_path = 1
    print("PATH ", target_path)
    from dates_for_graphs.date_for_graphs_path1 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path1 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    target_path = 2
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path2 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path2 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

    target_path = 3
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path3 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path3 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

    target_path = 4
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path4 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path4 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_lambda_dates

    plot_cdf_path_from_dates(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)



if __name__ == '__main__':
    main()
