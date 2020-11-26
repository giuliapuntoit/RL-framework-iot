import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import pylab as pl
plt.rcParams["font.family"] = "Times New Roman"

from config import GlobalVar


# Functions for plotting the moving average for multiple runs and multiple configurations of params

target_dir = "../plot/tuning_"
complete_target_dir = ""


def plot_single_configuration_multiple_runs(date_array, param):
    param_value = ""
    x_all = []
    y_all_reward = []
    y_all_cum_reward = []
    y_all_timesteps = []

    # retrieve data for all dates
    for dat in date_array:
        directory = GlobalVar.directory + 'output/output_Q_parameters'
        file_parameters = 'output_parameters_' + dat + '.csv'

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        algorithm = parameters['algorithm_used']
        print("RL ALGORITHM:", algorithm)
        param_value = parameters[param]

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

    df_final_avg_over_n_runs = pd.DataFrame(
        {'x': x_all[0], 'y1': y_final_reward, 'y2': y_final_timesteps, 'y3': y_final_cum_reward})

    # color = ('#77FF82', '#47CC99', '#239DBA', '#006586')

    window_size = 10

    # Compute the smoothed moving average
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(df_final_avg_over_n_runs['y1'], weights, 'valid')
    yMA_timesteps = np.convolve(df_final_avg_over_n_runs['y2'], weights, 'valid')

    # Compute also the global avg reward and the number of rewards >= average in percentage
    # Compute also the global avg timesteps and the number of rewards <= average in percentage
    global_avg_reward = np.mean(y_final_reward)
    global_std_dev_reward = np.std(y_final_reward)
    global_avg_timesteps = np.mean(y_final_timesteps)
    global_std_dev_timesteps = np.std(y_final_timesteps)

    global_n_reward = sum(i >= global_avg_reward for i in y_final_reward) * 100 / float(len(y_final_reward))
    global_n_timesteps = sum(i <= global_avg_timesteps for i in y_final_timesteps) * 100 / float(len(y_final_timesteps))

    # plot results
    # pl.plot(df_single_run['x'], df_single_run['y1'], label='single run', color=color[0])  # single line
    # pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y1'],
    #         label="avg over " + str(len(date_array)) + " run", color=color[1])  # avg line
    # pl.plot(df_final_avg_over_n_runs['x'], yMA[0:np.array(df_final_avg_over_n_runs['x']).shape[0]], 'r',
    #         label='moving average')  # moving avg line
    #
    # pl.xlabel('Episodes')
    # pl.ylabel('Reward')
    # pl.legend(loc='lower right')
    # pl.title('Reward for ' + algorithm + ' algorithm over episodes')
    # pl.grid(True)
    # plt.savefig('all_reward_plot.png')
    # plt.show()

    # plot results
    # pl.plot(df_single_run['x'], df_single_run['y2'], label='single run', color=color[0])  # single line
    # pl.plot(df_final_avg_over_n_runs['x'], df_final_avg_over_n_runs['y2'],
    #         label="avg over " + str(len(date_array)) + " run", color=color[1])  # avg line
    # pl.plot(df_final_avg_over_n_runs['x'], yMA_timesteps[0:np.array(df_final_avg_over_n_runs['x']).shape[0]], 'r',
    #         label='moving average')  # moving avg line
    #
    # pl.xlabel('Episodes')
    # pl.ylabel('Number of steps')
    # pl.legend(loc='lower right')
    # pl.title('Steps for ' + algorithm + ' algorithm over episodes')
    # pl.grid(True)
    # plt.savefig('all_timesteps_plot.png')
    # plt.show()

    return param_value, x_all[
        0], yMA, yMA_timesteps, global_avg_reward, global_avg_timesteps, global_n_reward, global_n_timesteps, global_std_dev_reward, global_std_dev_timesteps


def plot_multiple_configuration_moving_avg(algorithm, param, param_values_target, episodes_target,
                                           moving_average_rewards_target,
                                           moving_average_timesteps_target):
    # 3 colors for params
    # color = ('#A66066', '#F2A172', '#858C4A')

    for i in range(0, len(param_values_target)):
        pl.plot(episodes_target[i][
                np.array(episodes_target[i]).shape[0] - np.array(moving_average_rewards_target[i]).shape[0]:],
                moving_average_rewards_target[i],
                label=param + "=" + param_values_target[i], )  # color=color[i])

    pl.xlabel('Episodes')
    pl.ylabel('Reward')
    pl.legend(loc='lower right')
    pl.title('Moving average of reward over episodes for ' + algorithm)
    pl.grid(True)
    plt.savefig(complete_target_dir + 'mavg_reward_params.png')
    plt.show()

    for i in range(0, len(param_values_target)):
        pl.plot(episodes_target[i][
                np.array(episodes_target[i]).shape[0] - np.array(moving_average_timesteps_target[i]).shape[0]:],
                moving_average_timesteps_target[i],
                label=param + "=" + param_values_target[i], )  # color=color[i])

    pl.xlabel('Episodes')
    pl.ylabel('Number of steps')
    pl.legend(loc='upper right')
    pl.title('Moving average of number of steps over episodes for ' + algorithm)
    pl.grid(True)
    plt.savefig(complete_target_dir + 'mavg_timesteps_params.png')
    plt.show()


def plot_multiple_configuration_rewards_timesteps(algo, param, param_values, avg_rew, avg_steps, n_rew, n_steps,
                                                  std_dev_rew, std_dev_steps):
    fig, ax = plt.subplots()
    col = ax.bar(param_values,
                 avg_rew,
                 align='center')
    # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg reward value')
    ax.set_title('Avg reward for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'avg_rewards_for_' + param + '.png')

    fig, ax = plt.subplots()
    col = ax.bar(param_values,
                 avg_steps, align='center', )
    # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg steps value')
    ax.set_title('Avg steps for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'avg_steps_for_' + param + '.png')

    fig, ax = plt.subplots()
    col = ax.bar(param_values,
                 n_rewards, align='center', )  # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of rewards upon avg (%)')
    ax.set_title('Percentage of rewards over avg for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'n_rewards_for_' + param + '.png')

    fig, ax = plt.subplots()
    col = ax.bar(param_values,
                 n_steps,
                 align='center', )  # color=('#A66066', '#F2A172', '#858C4A'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of steps below avg (%)')
    ax.set_title('Percentage of steps below avg for different configurations of ' + param)

    fig.tight_layout()

    plt.savefig(complete_target_dir + 'n_steps_for_' + param + '.png')


def boxplot_multiple_configurations_rewards_timesteps_last_episodes(algor, param, values_of_param, last_20_rewards, last_20_timesteps):
    fig, ax = plt.subplots()

    # non sto più facendo una media, sto mettendo tutti i punti del reward medio
    # last 20 episodes rewards of 5 run -> 100 punti per box
    # [    run 1     run 2   run 3   run 4    run 5
    #     [ep 90]    ...     ...
    #     [ep 91]
    #     ...
    #     [ep 100]
    # ]
    col = ax.boxplot(last_20_rewards)

    ax.set_xticklabels(values_of_param)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg reward')
    ax.set_title('Avg reward in 5 runs of last 20 episodes per config of ' + param + ' for algo ' + algor)

    fig.tight_layout()

    plt.savefig('boxplot_param_reward_last_20.png')

    fig, ax = plt.subplots()

    col = ax.boxplot(last_20_timesteps)  # , ["SARSA", "SARSA(λ)", "Q-learning", "Q(λ)"])

    ax.set_xticklabels(values_of_param)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg time steps')
    ax.set_title('Avg time steps in 5 runs of last 20 episodes per config of ' + param + ' for algo ' + algor)

    fig.tight_layout()

    plt.savefig('boxplot_param_timestep_last_20.png')


if __name__ == '__main__':
    # I could pass a list of dates, then do the average of these dates
    # Then put multiple lines inside 1 multiline plot
    changing_param_values = []
    episodes = []
    moving_avgs_rewards = []
    moving_avgs_timesteps = []
    avg_rewards = []
    avg_timesteps = []
    n_rewards = []
    n_timesteps = []
    std_dev_rewards = []
    std_dev_timesteps = []

    boxplot_last_rewards = []
    boxplot_last_timesteps = []

    changing_param = "lambda"
    algo = "sarsa_lambda"
    complete_target_dir = target_dir + changing_param + "/" + algo + "/"
    print("ALGO SHOULD BE", algo, "FOR ALL RESULTS")
    value_of_lambda = [
    [  # lambda = 0
        '2020_11_16_01_11_21',
        '2020_11_16_01_42_18',
        '2020_11_16_02_13_10',
        '2020_11_16_02_46_26',
        '2020_11_16_03_19_58', ],
    [  # lambda = 0.5
        '2020_11_16_03_49_30',
        '2020_11_16_04_20_40',
        '2020_11_16_04_50_52',
        '2020_11_16_05_19_47',
        '2020_11_16_05_48_35', ],
    [  # lambda = 0.8
        '2020_11_16_06_23_01',
        '2020_11_16_06_56_41',
        '2020_11_16_07_27_59',
        '2020_11_16_07_54_40',
        '2020_11_16_08_31_02', ],
    [  # lambda = 0.9
        '2020_11_16_09_00_46',
        '2020_11_16_09_37_10',
        '2020_11_16_10_10_24',
        '2020_11_16_10_44_49',
        '2020_11_16_11_14_23', ],
    [  # lambda = 0.95
        '2020_11_16_11_46_49',
        '2020_11_16_12_26_48',
        '2020_11_16_13_05_14',
        '2020_11_16_13_49_15',
        '2020_11_16_14_21_50', ],
    [  # lambda = 1
        '2020_11_16_14_56_06',
        '2020_11_16_15_30_21',
        '2020_11_16_16_03_15',
        '2020_11_16_16_39_18',
        '2020_11_16_17_07_38', ]
]
    for val in value_of_lambda:
        p, ep, ma, mats, ar, at, nr, nt, sdr, sdt = plot_single_configuration_multiple_runs(date_array=val,
                                                                                            param=changing_param)
        changing_param_values.append(p)
        episodes.append(ep)
        moving_avgs_rewards.append(ma)
        moving_avgs_timesteps.append(mats)

        # voglio gli ultimi 20 avg reward degli ultimi 20 episodi e 5 run
        # TODO
        boxplot_last_rewards.append()
        boxplot_last_timesteps.append()

        avg_rewards.append(ar)
        avg_timesteps.append(at)
        n_rewards.append(nr)
        n_timesteps.append(nt)
        std_dev_rewards.append(sdr)
        std_dev_timesteps.append(sdt)

    plot_multiple_configuration_moving_avg(algo, changing_param, changing_param_values, episodes, moving_avgs_rewards,
                                           moving_avgs_timesteps)

    plot_multiple_configuration_rewards_timesteps(algo, changing_param, changing_param_values, avg_rewards,
                                                  avg_timesteps, n_rewards, n_timesteps, std_dev_rewards,
                                                  std_dev_timesteps)

    boxplot_multiple_configurations_rewards_timesteps_last_episodes(algo, changing_param, changing_param_values, boxplot_last_rewards, boxplot_last_timesteps)
