"""
    Class to retrieve training time and traffic data about execution of RL algorithms
"""

import os
import csv
import pathlib

from plotter.support_plotter import read_time_traffic_from_log


output_dir = "./tmp/"


class GetTrainingTimeTraffic(object):
    def __init__(self, date_to_retrieve='YY_mm_dd_HH_MM_SS', target_output="algorithm.csv"):
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            print("Invalid date")
            exit(1)
        self.target_output = target_output

    def run(self):
        """
        Retrieve and save into csv files training time and traffic
        """
        secs, commands = read_time_traffic_from_log(self.date_to_retrieve)

        if not os.path.isfile(self.target_output):  # If file does not exist
            # Write header
            with open(self.target_output, mode='w') as output_file:
                output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow(['Date', 'Training_time', 'Sent_commands'])

        with open(self.target_output, mode="a") as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow([self.date_to_retrieve, secs, commands])


def get_data_before_tuning_unique_path():
    """
    Retrieve data for executions made before parameter tuning phase
    All the executions refer to the same path 2
    """

    from dates_for_graphs.date_for_graphs_before_tuning_path2 import sarsa
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import sarsa_lambda
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import qlearning
    from dates_for_graphs.date_for_graphs_before_tuning_path2 import qlearning_lambda

    for dat in sarsa:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'0_sarsa.csv').run()

    for dat in sarsa_lambda:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'0_sarsa_lambda.csv').run()

    for dat in qlearning:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'0_qlearning.csv').run()

    for dat in qlearning_lambda:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'0_qlearning_lambda.csv').run()


def get_data_algos_path(sarsa, sarsa_lambda, qlearning, qlearning_lambda, path=None):
    """
    Retrieve training time and traffic for all different algorithms and append into related csv files
    """

    for dat in sarsa:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'path' + str(path) + '_sarsa.csv').run()

    for dat in sarsa_lambda:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'path' + str(path) + '_sarsa_lambda.csv').run()

    for dat in qlearning:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'path' + str(path) + '_qlearning.csv').run()

    for dat in qlearning_lambda:
        GetTrainingTimeTraffic(date_to_retrieve=dat, target_output=output_dir+'path' + str(path) + '_qlearning_lambda.csv').run()


def main():
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)  # for Python > 3.5

    get_data_before_tuning_unique_path()

    # Plot all paths
    target_path = 1
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path1 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path1 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path1 import qlearning_lambda_dates

    get_data_algos_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)
    target_path = 2
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path2 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path2 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path2 import qlearning_lambda_dates

    get_data_algos_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

    target_path = 3
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path3 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path3 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path3 import qlearning_lambda_dates

    get_data_algos_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)

    target_path = 4
    print("PATH ", target_path)

    from dates_for_graphs.date_for_graphs_path4 import sarsa_dates
    from dates_for_graphs.date_for_graphs_path4 import sarsa_lambda_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_dates
    from dates_for_graphs.date_for_graphs_path4 import qlearning_lambda_dates

    get_data_algos_path(sarsa_dates, sarsa_lambda_dates, qlearning_dates, qlearning_lambda_dates, path=target_path)


if __name__ == '__main__':
    main()
