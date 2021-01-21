"""
    Script containing methods useful for other plots
"""
import csv
from matplotlib import patches

from config import FrameworkConfiguration


def print_cute_algo_name(a):
    """
    Return algorithm with greek letters
    """
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


def fix_hist_step_vertical_line_at_end(ax):
    """
    Support function to adjust layout of plots
    """
    ax_polygons = [poly for poly in ax.get_children() if isinstance(poly, patches.Polygon)]
    for poly in ax_polygons:
        poly.set_xy(poly.get_xy()[:-1])


def build_directory_and_filename(algorithm, date):
    """
    Find directory and the filename to retrieve data
    """
    if algorithm is None:
        directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
        file_parameters = 'output_parameters_' + date + '.csv'

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        algorithm = parameters['algorithm_used']

    print("RL ALGORITHM:", algorithm)

    directory = FrameworkConfiguration.directory + 'output/output_csv'
    filename = 'output_' + algorithm + '_' + date + '.csv'

    return directory, filename


def read_avg_reward_from_output_file(algorithm, date_to_retrieve):
    """
    Retrieve and compute the average reward per time step for episodes from output
    """

    directory, filename = build_directory_and_filename(algorithm, date_to_retrieve)

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

    return x, y_avg_reward_for_one_episode


def read_reward_timesteps_from_output_file(algorithm, date_to_retrieve):
    """
    Read reward, cumulative reward and timesteps data from output file
    """
    directory, filename = build_directory_and_filename(algorithm, date_to_retrieve)

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

    return x, y_reward, y_cum_reward, y_timesteps


def compute_avg_over_multiple_runs(number_episodes, number_runs, y_all_reward, y_all_cum_reward, y_all_timesteps):
    """
    Compute average of reward and timesteps over multiple runs (different dates)
    """
    y_final_reward = []
    y_final_cum_reward = []
    y_final_timesteps = []
    for array_index in range(0, number_episodes):
        sum_r = 0
        sum_cr = 0
        sum_t = 0
        count = 0
        for date_index in range(0, number_runs):  # compute average
            sum_r += y_all_reward[date_index][array_index]
            sum_cr += y_all_cum_reward[date_index][array_index]
            sum_t += y_all_timesteps[date_index][array_index]
            count += 1
        y_final_reward.append(sum_r / float(count))
        y_final_cum_reward.append(sum_cr / float(count))
        y_final_timesteps.append(sum_t / float(count))

    return y_final_reward, y_final_cum_reward, y_final_timesteps
