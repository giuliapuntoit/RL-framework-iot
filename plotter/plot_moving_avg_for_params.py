"""
    Script for plotting the moving average of the reward and timesteps over episodes for different params configurations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from matplotlib.font_manager import FontProperties
from plotter.support_plotter import read_reward_timesteps_from_output_file, compute_avg_over_multiple_runs, \
    return_greek_letter, read_parameters_from_output_file, build_output_dir_for_params

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 20

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 2

target_dir = "../plot/"  # Tuning


def compute_avg_single_configuration_multiple_runs(date_array, param, for_robustness=False):
    """
    Compute and return average values for reward and timesteps
    for multiple excutions of the same configuration of parameters
    """
    window_size = 10

    param_value = ""
    x_all = []
    y_all_reward = []
    y_all_cum_reward = []
    y_all_timesteps = []

    # retrieve data for all dates
    for dat in date_array:
        parameters = read_parameters_from_output_file(dat)
        param_value = parameters[param]

        x, y_reward, y_cum_reward, y_timesteps = read_reward_timesteps_from_output_file(None, dat)

        if for_robustness:
            if len(x) > 100:
                x = x[0:100]
                y_reward = y_reward[0:100]
                y_cum_reward = y_cum_reward[0:100]
                y_timesteps = y_timesteps[0:100]

        x_all.append(x)
        y_all_reward.append(y_reward)
        y_all_cum_reward.append(y_cum_reward)
        y_all_timesteps.append(y_timesteps)

    # compute average over multiple runs
    y_final_reward, y_final_cum_reward, y_final_timesteps = compute_avg_over_multiple_runs(len(x_all[0]),
                                                                                           len(date_array),
                                                                                           y_all_reward,
                                                                                           y_all_cum_reward,
                                                                                           y_all_timesteps)
    df_final_avg_over_n_runs = pd.DataFrame(
        {'x': x_all[0], 'y1': y_final_reward, 'y2': y_final_timesteps, 'y3': y_final_cum_reward})

    # Compute the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(df_final_avg_over_n_runs['y1'], weights, 'valid')
    yMA_timesteps = np.convolve(df_final_avg_over_n_runs['y2'], weights, 'valid')

    # Compute also the global avg reward and the number of rewards >= average in percentage
    # Compute also the global avg timesteps and the number of rewards <= average in percentage
    global_avg_reward = np.mean(y_final_reward)
    global_avg_timesteps = np.mean(y_final_timesteps)

    return param_value, x_all[0], yMA, yMA_timesteps, global_avg_reward, global_avg_timesteps


def plot_multiple_configuration_moving_avg(algorithm, param, param_values_target, episodes_target,
                                           moving_average_rewards_target,
                                           moving_average_timesteps_target):
    """
    Generate plots with the moving average of reward and timesteps
    for multiple executions and configurations of a single parameter
    """

    complete_target_dir = build_output_dir_for_params(target_dir, param, algorithm)

    for i in range(0, len(param_values_target)):
        if i == 0 and param == "lambda":  # lambda can be 0 so I do not remove the 0 in the legend of the plot
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_rewards_target[i]).shape[0]:],
                    moving_average_rewards_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i], )
        else:
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_rewards_target[i]).shape[0]:],
                    moving_average_rewards_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i].lstrip('0'), )  # remove 0 from legend, keep only decimals

    pl.xlabel('Episode')
    pl.ylabel('Final reward')
    pl.legend(loc='lower right', prop=fontP, ncol=n_cols)
    # pl.title('Moving average of reward over episodes for ' + algorithm)
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(complete_target_dir + 'mavg_reward_params.png')
    plt.show()

    for i in range(0, len(param_values_target)):
        if i == 0 and param == "lambda":
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_timesteps_target[i]).shape[0]:],
                    moving_average_timesteps_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i], )  # color=color[i])
        else:
            pl.plot(episodes_target[i][
                    np.array(episodes_target[i]).shape[0] - np.array(moving_average_timesteps_target[i]).shape[0]:],
                    moving_average_timesteps_target[i],
                    label=return_greek_letter(param) + r'$=$' + param_values_target[i].lstrip('0'), )

    pl.xlabel('Episode')
    pl.ylabel('Number of time steps')
    pl.legend(loc='upper right', prop=fontP, ncol=n_cols)
    # pl.title('Moving average of number of steps over episodes for ' + algorithm)
    pl.grid(True, color='gray', linestyle='dashed')
    pl.tight_layout()
    plt.savefig(complete_target_dir + 'mavg_timesteps_params.png')
    plt.show()


def plot_multiple_configuration_avg_rewards_timesteps_bars(algo, param, param_values, avg_rew, avg_steps):
    """
    Generate bar plots with the global average reward and timesteps values
    for multiple executions and configurations of a single parameter
    """

    complete_target_dir = build_output_dir_for_params(target_dir, param, algo)

    fig, ax = plt.subplots()
    param_labels = []
    cnt = 0
    for i in param_values:
        if cnt == 0 and param == "lambda":
            param_labels.append(return_greek_letter(param) + "=" + i)
            cnt += 1
        else:
            param_labels.append(return_greek_letter(param) + "=" + i.lstrip('0'))
    col = ax.bar(param_labels,
                 avg_rew,
                 align='center')

    ax.set_ylabel('Avg reward for episode')
    # ax.set_title('Avg reward for different configurations of ' + param)
    plt.axhline(0, color='black', lw=.3)
    fig.tight_layout()
    plt.savefig(complete_target_dir + 'avg_rewards_for_' + param + '.png')
    plt.show()

    fig, ax = plt.subplots()
    param_labels = []
    cnt = 0
    for i in param_values:
        if cnt == 0 and param == "lambda":
            param_labels.append(return_greek_letter(param) + "=" + i)
            cnt += 1
        else:
            param_labels.append(return_greek_letter(param) + "=" + i.lstrip('0'))
    col = ax.bar(param_labels,
                 avg_steps, align='center', )

    ax.set_ylabel('Avg time steps for episode')
    # ax.set_title('Avg steps for different configurations of ' + param)
    fig.tight_layout()
    plt.savefig(complete_target_dir + 'avg_steps_for_' + param + '.png')
    plt.show()


def boxplot_multiple_configurations_rewards_timesteps_last_episodes(algor, param, values_of_param, last_20_rewards,
                                                                    last_20_timesteps):
    """
    Generate boxplots for the value of reward over the last 20 episodes
    """

    # non sto più facendo una media, sto mettendo tutti i punti del reward medio
    # last 20 episodes rewards of 5 run -> 100 punti per box
    # [    run 1     run 2   run 3   run 4    run 5
    #     [ep 90]    ...     ...
    #     [ep 91]
    #     ...
    #     [ep 100]
    # ]
    fig, ax = plt.subplots()
    col = ax.boxplot(last_20_rewards)
    ax.set_xticklabels(values_of_param)
    ax.set_ylabel('Avg reward')
    ax.set_title('Avg reward in 5 runs of last 20 episodes per config of ' + param + ' for algo ' + algor)
    fig.tight_layout()
    plt.savefig('boxplot_param_reward_last_20.png')

    fig, ax = plt.subplots()
    col = ax.boxplot(last_20_timesteps)  # , ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])
    ax.set_xticklabels(values_of_param)
    ax.set_ylabel('Avg time steps')
    ax.set_title('Avg time steps in 5 runs of last 20 episodes per config of ' + param + ' for algo ' + algor)
    fig.tight_layout()
    plt.savefig('boxplot_param_timestep_last_20.png')


def plot_graphs_for_changing_param(algo, changing_param, for_robustness=False):
    """
    Build graphs for tuning of 1 single parameter and 1 single algorithm
    """

    changing_param_values = []
    episodes = []
    moving_avgs_rewards = []
    moving_avgs_timesteps = []
    avg_rewards = []
    avg_timesteps = []

    print("ALGO SHOULD BE", algo, "FOR ALL RESULTS")

    values = []

    if for_robustness:
        if algo == "qlearning":
            if changing_param == "epsilon":
                from date_for_robustness import qlearning_dates_eps as values
            elif changing_param == "alpha":
                from date_for_robustness import qlearning_dates_alp as values
            elif changing_param == "gamma":
                from date_for_robustness import qlearning_dates_gam as values
            else:
                print("Invalid changing_param")
                exit(1)
        else:
            print("Invalid algo for robustness")
            exit(1)
    else:
        if algo == "qlearning":
            if changing_param == "epsilon":
                from date_for_graphs_tuning import value_of_epsilon_qlearning as values
            elif changing_param == "alpha":
                from date_for_graphs_tuning import value_of_alpha_qlearning as values
            elif changing_param == "gamma":
                from date_for_graphs_tuning import value_of_gamma_qlearning as values
            else:
                print("Invalid changing_param")
                exit(2)
        elif algo == "qlearning_lambda":
            if changing_param == "lambda":
                from date_for_graphs_tuning import value_of_lambda_qlearning_lambda as values
            else:
                print("Invalid changing_param")
                exit(3)
        elif algo == "sarsa":
            if changing_param == "epsilon":
                from date_for_graphs_tuning import value_of_epsilon_sarsa as values
            elif changing_param == "alpha":
                from date_for_graphs_tuning import value_of_alpha_sarsa as values
            elif changing_param == "gamma":
                from date_for_graphs_tuning import value_of_gamma_sarsa as values
            else:
                print("Invalid changing_param")
                exit(4)
        elif algo == "sarsa_lambda":
            if changing_param == "lambda":
                from date_for_graphs_tuning import value_of_lambda_sarsa_lambda as values
            else:
                print("Invalid changing_param")
                exit(5)
        else:
            print("Invalid algo")
            exit(6)

    for val in values:
        p, ep, ma, mats, ar, at = compute_avg_single_configuration_multiple_runs(date_array=val, param=changing_param, for_robustness=for_robustness)
        changing_param_values.append(p)
        episodes.append(ep)
        moving_avgs_rewards.append(ma)
        moving_avgs_timesteps.append(mats)
        avg_rewards.append(ar)
        avg_timesteps.append(at)

    print(changing_param)

    plot_multiple_configuration_moving_avg(algo, changing_param, changing_param_values, episodes, moving_avgs_rewards,
                                           moving_avgs_timesteps)

    plot_multiple_configuration_avg_rewards_timesteps_bars(algo, changing_param, changing_param_values, avg_rewards,
                                                           avg_timesteps)


def main():
    global target_dir
    target_dir = "../plot/tuning"  # Tuning
    for alg in ["sarsa", "qlearning"]:
        for p in ["epsilon", "alpha", "gamma"]:
            plot_graphs_for_changing_param(alg, p, for_robustness=False)

    for alg in ["sarsa_lambda", "qlearning_lambda"]:
        plot_graphs_for_changing_param(alg, "lambda", for_robustness=False)

    target_dir = "../plot/robustness"  # only for robustness
    plot_graphs_for_changing_param("qlearning", "gamma", for_robustness=True)


if __name__ == '__main__':
    main()
