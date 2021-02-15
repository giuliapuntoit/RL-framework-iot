"""
    Script for plotting the moving average of the reward and timesteps over episodes for multiple runs and different algorithms
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib.font_manager import FontProperties
from plotter.support_plotter import print_cute_algo_name, read_reward_timesteps_from_output_file, \
    compute_avg_over_multiple_runs, build_output_dir_from_path

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

output_dir = './'


def plot_single_algo_single_run(date_to_retrieve):
    """
    Generate plots showing reward and cumulative reward of a single execution of the learning process
    """
    x, y_reward, y_cum_reward, y_timesteps = read_reward_timesteps_from_output_file(None, date_to_retrieve)

    window_size = 10

    df = pd.DataFrame({'x': x, 'y1': y_reward, 'y2': y_timesteps, 'y3': y_cum_reward})

    # ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])

    fig, ax = plt.subplots()

    plt.plot(df['x'], df['y1'], data=None, label="reward")
    plt.plot(df['x'], df['y2'], data=None, label="cum")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    # plt.title('Cum reward per algorithm')
    plt.grid(True, color='gray', linestyle='dashed')
    ax.set_xlim(xmin=0)
    fig.tight_layout()
    plt.legend(loc='lower right', prop=fontP, ncol=n_cols)

    plt.show()

    # calculate the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(y_reward, weights, 'valid')
    # plot results
    pl.plot(x[np.array(x).shape[0] - yMA.shape[0]:], yMA, 'r', label='MA')
    pl.plot(x, df['y1'], 'y--', label='data')
    pl.xlabel('Time')
    pl.ylabel('y')
    fig.tight_layout()
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Moving Average with window size = ' + str(window_size))
    pl.grid(True, color='gray', linestyle='dashed')
    pl.show()


def plot_single_algo_multiple_runs(date_array, algorithm=None, path=None):
    """
    Generate plots with reward of a single execution over episodes, average reward and moving average
    reward computed over multiple executions of the same RL algorithm (same values for timesteps)
    """
    target_output_dir = build_output_dir_from_path(output_dir, path)

    window_size = 10

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
        x, y_reward, y_cum_reward, y_timesteps = read_reward_timesteps_from_output_file(algorithm, dat)

        x_all.append(x)
        y_all_reward.append(y_reward)
        y_all_cum_reward.append(y_cum_reward)
        y_all_timesteps.append(y_timesteps)

    # compute average over multiple runs
    y_final_reward, y_final_cum_reward, y_final_timesteps = compute_avg_over_multiple_runs(len(x_all[0]), len(date_array), y_all_reward, y_all_cum_reward, y_all_timesteps)

    df_single_run = pd.DataFrame({'x': x, 'y1': y_reward, 'y2': y_timesteps, 'y3': y_cum_reward})
    df_final_avg_over_n_runs = pd.DataFrame(
        {'x': x_all[0], 'y1': y_final_reward, 'y2': y_final_timesteps, 'y3': y_final_cum_reward})

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


def compute_single_algo_multiple_runs_global_values_for_avg_bars(date_array, algorithm=None):
    """
    Compute avg reward value averaged over all episodes and all executions
    Obtains a unique number for 1 algorithm (same values for timesteps)
    """
    x_all = []
    y_all_reward = []
    y_all_cum_reward = []
    y_all_timesteps = []

    # retrieve data for all dates
    for dat in date_array:
        x, y_reward, y_cum_reward, y_timesteps = read_reward_timesteps_from_output_file(algorithm, dat)

        x_all.append(x)
        y_all_reward.append(y_reward)
        y_all_cum_reward.append(y_cum_reward)
        y_all_timesteps.append(y_timesteps)

    # compute average over multiple runs
    y_final_reward, y_final_cum_reward, y_final_timesteps = compute_avg_over_multiple_runs(len(x_all[0]), len(date_array), y_all_reward, y_all_cum_reward, y_all_timesteps)

    # compute single average value over all episodes for reward and timesteps values
    global_avg_reward = np.mean(y_final_reward)
    global_avg_timesteps = np.mean(y_final_timesteps)

    return global_avg_reward, global_avg_timesteps


def plot_multiple_algo_moving_avg(algorithms_target, episodes_target, moving_average_rewards_target,
                                  moving_average_timesteps_target, path=None):
    """
    Generate plots having the moving average reward and timesteps values for all RL algorithms
    """
    target_output_dir = build_output_dir_from_path(output_dir, path)

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
    """
    Plot graphs for data before the tuning phase
    """
    algos = []
    episodes = []
    moving_avgs_rewards = []
    moving_avgs_timesteps = []

    from dates_for_graphs.date_for_graphs_before_tuning_path2 import sarsa
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import sarsa_lambda
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import qlearning
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import qlearning_lambda

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


def all_graphs_for_specified_path(sarsa, sarsa_lambda, qlearning, qlearning_lambda, path=None):
    """
    Plot all graphs for a single path
    """
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


def plot_multiple_algos_avg_rewards_timesteps_bars(algos, avg_rew, avg_steps, path):
    """
    Plot averaged bar graphs for 1 single path
    """
    target_output_dir = build_output_dir_from_path(output_dir, path)

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


def main():
    all_algo = ["sarsa", "sarsa_lambda", "qlearning", "qlearning_lambda"]

    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 1
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path1 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path1 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_lambda_dates

    # plot_single_algo_single_run(sarsa_dates[0])

    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_for_specified_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_avg_rewards_timesteps_bars(all_algo, all_avg_rew, all_avg_timesteps, target_path)

    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 2
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path2 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path2 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_lambda_dates

    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_for_specified_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_avg_rewards_timesteps_bars(all_algo, all_avg_rew, all_avg_timesteps, target_path)

    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 3
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path3 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path3 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_lambda_dates

    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_for_specified_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_avg_rewards_timesteps_bars(all_algo, all_avg_rew, all_avg_timesteps, target_path)


    all_avg_rew = []
    all_avg_timesteps = []

    target_path = 4
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path4 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path4 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_lambda_dates

    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_dates, all_algo[0])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(sarsa_lambda_dates, all_algo[1])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_dates, all_algo[2])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)
    gar, gat = compute_single_algo_multiple_runs_global_values_for_avg_bars(qlearning_lambda_dates, all_algo[3])
    all_avg_rew.append(gar)
    all_avg_timesteps.append(gat)

    all_graphs_for_specified_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    plot_multiple_algos_avg_rewards_timesteps_bars(all_algo, all_avg_rew, all_avg_timesteps, target_path)


if __name__ == '__main__':
    main()
