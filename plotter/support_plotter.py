"""
    Script containing methods useful for other plots
"""

import csv
import pathlib
import shutil
import numpy as np
from matplotlib import patches
from config import FrameworkConfiguration


def get_font_family_and_size():
    """
    Function to globally set and get font family and font size for plots
    """
    font_family = "Times New Roman"
    font_size = 22
    return font_family, font_size


def get_extension():
    """
    Function to globally set and get the extension for plots
    """
    extension = '.pdf'
    return extension


def print_cute_algo_name(a):
    """
    Function to return algorithm with greek letters
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


def return_greek_letter(par):
    """
    Function to return the corresponding greek letter
    """
    if par == "epsilon":
        return "ε"
    elif par == "alpha":
        return "α"
    elif par == "gamma":
        return "γ"
    elif par == "lambda":
        return "λ"
    else:
        return "invalid"


def build_output_dir_from_path(output_dir, path, partial=None):
    target_output_dir = output_dir
    if path in [1, 2, 3, 4]:
        if partial is None:
            target_output_dir = "../plot/path" + str(path) + "/"
        else:
            target_output_dir = "../plot/partial/path" + str(path) + "/"
        pathlib.Path(target_output_dir).mkdir(parents=True, exist_ok=True)  # for Python > 3.5
    return target_output_dir


def build_output_dir_for_params(output_dir, changing_param, algo):
    target_output_dir = output_dir + "/" + changing_param + "/" + algo + "/"
    pathlib.Path(target_output_dir).mkdir(parents=True, exist_ok=True)  # for Python > 3.5
    return target_output_dir


def fix_hist_step_vertical_line_at_end(ax):
    """
    Support function to adjust layout of plots
    """
    ax_polygons = [poly for poly in ax.get_children() if isinstance(poly, patches.Polygon)]
    for poly in ax_polygons:
        poly.set_xy(poly.get_xy()[:-1])


def build_directory_and_filename(algorithm, date, partial=None):
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
    if partial is not None:
        filename = 'partial_output_' + algorithm + '_' + date + '.csv'

    return directory, filename


def build_directory_and_logfile(date_to_retrieve):
    """
    Find directory and the log name to retrieve data
    """
    directory = FrameworkConfiguration.directory + 'output/log/'
    log_file = directory + 'log_' + date_to_retrieve + '.log'
    return directory, log_file


def read_all_info_from_log(date_to_retrieve):
    """
    Retrieve all info from log file
    """
    directory, log_file = build_directory_and_logfile(date_to_retrieve)

    print(log_file)

    count = 0
    cum_reward = 0
    commands = []
    rewards = []
    cum_rewards = []
    episodes = []
    with open(log_file) as f:
        for line in f:
            if len(line.strip()) != 0:  # Not empty lines
                if line.startswith("Episode"):
                    episodes.append(count)
                if not line.startswith("Episode") and not line.startswith("Total"):
                    count += 1
                    commands.append(count)
                    tmp_reward = int(line.split()[5])
                    cum_reward += tmp_reward
                    rewards.append(tmp_reward)
                    cum_rewards.append(cum_reward)

    return episodes, commands, rewards, cum_rewards


def read_time_traffic_from_log(date_to_retrieve):
    """
    Retrieve only training time and traffic from log file
    """
    directory, log_file = build_directory_and_logfile(date_to_retrieve)

    print(log_file)

    # Each non empty line is a sent command
    # Command of power is substituted by episode finishing line
    # Minus last line that is the total time

    counter_line = -1
    with open(log_file) as f:
        for line in f:
            if len(line.strip()) != 0:  # Not empty lines
                counter_line += 1
        last_line = line

    secs = float(last_line.split()[3])
    np.set_printoptions(formatter={'float': lambda output: "{0:0.4f}".format(output)})

    print("Total lines", counter_line)
    print("Last line", last_line)
    print("Seconds", secs)

    # Number of lines in log file correspond to number of sent commands
    return secs, counter_line


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
            if float(row[3]) == 0.0:
                y_avg_reward_for_one_episode.append(float(row[1]) / 1.0)
            else:
                y_avg_reward_for_one_episode.append(float(row[1]) / float(row[3]))

    return x, y_avg_reward_for_one_episode


def read_parameters_from_output_file(date_to_retrieve):
    """
    Retrieve parameter value from output file
    """
    directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
    file_parameters = 'output_parameters_' + date_to_retrieve + '.csv'

    with open(directory + '/' + file_parameters, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

    return parameters


def read_reward_timesteps_from_output_file(algorithm, date_to_retrieve, partial=None):
    """
    Read reward, cumulative reward and timesteps data from output file
    """
    directory, filename = build_directory_and_filename(algorithm, date_to_retrieve, partial)

    x = []
    y_reward = []
    y_cum_reward = []
    y_timesteps = []

    with open(directory + '/' + filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader, None)
        for row in reader:
            if partial is None:
                #if int(row[0]) >= 100:
                #    break
                x.append(int(row[0]))
                y_reward.append(int(row[1]))
                y_cum_reward.append(int(row[2]))
                y_timesteps.append(int(row[3]))
            else:
                x.append(int(row[0]))
                y_timesteps.append(int(row[1]))
                y_reward.append(int(row[2]))
                y_cum_reward.append(0)  # don't care about cumulative reward if I want to analyze partial results

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


def clear_tmp_files():
    """
    Delete the tmp directory inside Plotter module containing temporary files used for plotting
    """
    shutil.rmtree("./tmp/")
