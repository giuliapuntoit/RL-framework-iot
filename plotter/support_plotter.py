"""
    Script containing methods useful for other plots
"""
import csv
from matplotlib import patches

from config import FrameworkConfiguration


def fix_hist_step_vertical_line_at_end(ax):
    """
    Support function to adjust layout of plots
    """
    ax_polygons = [poly for poly in ax.get_children() if isinstance(poly, patches.Polygon)]
    for poly in ax_polygons:
        poly.set_xy(poly.get_xy()[:-1])


def build_directory_and_filename(algorithm, date):
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


def read_avg_reward_from_output_file(algorithm, dat):
    directory, filename = build_directory_and_filename(algorithm, dat)
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
