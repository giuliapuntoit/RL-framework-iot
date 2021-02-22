"""
    Class to follow the best policy found by a learning process
"""
import logging

import numpy as np
import csv
import time
from time import sleep
from config import FrameworkConfiguration

# Q will be read from output_Q_date.csv
# Retrieve actions and state from output_Q_date.csv
# Statistics to compute Q will be read from output_parameters_date.csv

# Identify which RL algorithm was used and use it

from device_communication.client import operate_on_bulb, operate_on_bulb_json
from discovery import network_analyzer, yeelight_analyzer
from formatter_for_output import format_console_output
from plotter.support_plotter import read_parameters_from_output_file
from state_machine.state_machine_yeelight import compute_reward_from_states, compute_next_state_from_props
from request_builder.builder import build_command


class RunOutputQParameters(object):
    def __init__(self, date_to_retrieve='YY_mm_dd_HH_MM_SS', show_retrieved_info=True, discovery_report=None):
        self.show_retrieved_info = show_retrieved_info
        self.discovery_report = discovery_report
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            logging.error("Invalid date")
            exit(1)
        self.storage_reward = 0

    def run(self):
        directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
        file_Q = 'output_Q_' + self.date_to_retrieve + '.csv'
        file_parameters = 'output_parameters_' + self.date_to_retrieve + '.csv'

        actions = []
        states = []
        Q = []

        # Retrieving Q matrix, states and actions
        with open(directory + '/' + file_Q, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for index_row, row in enumerate(reader):
                if index_row == 0:  # Remember to remove first cell
                    for index_col, col in enumerate(row):
                        if index_col != 0:
                            actions.append(str(col.strip()))
                else:
                    for index_col, col in enumerate(row):
                        if index_col == 0:
                            states.append(str(col))

        try:
            tmp_matrix = np.genfromtxt(directory + '/' + file_Q, delimiter=',', dtype=np.float32)
            Q = tmp_matrix[1:, 1:]

        except Exception as e:
            logging.error("Wrong file format:" + str(e))
            exit(1)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        if self.show_retrieved_info:
            logging.info("STATES:\t" + str(states))
            logging.info("ACTIONS:\t" + str(actions))
            logging.info("Q MATRIX:")
            logging.info(Q)

        if len(states) != len(Q) or len(actions) != len(Q[0]) or np.isnan(np.sum(Q)):
            logging.error("Wrong file format: wrong Q dimensions or nan values present")
            exit(2)

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        if self.show_retrieved_info:
            logging.info("USED PARAMETERS:\t" + str(parameters))  # For now the are all strings

        if parameters['num_actions_to_use'] is not None and len(actions) != int(parameters['num_actions_to_use']):
            logging.error("Different number of actions used")
            exit(3)

        if parameters['algorithm_used'] == 'sarsa_lambda' or parameters['algorithm_used'] == 'qlearning_lambda':

            file_E = 'output_E_' + self.date_to_retrieve + '.csv'

            E = []

            try:
                tmp_matrix = np.genfromtxt(directory + '/' + file_E, delimiter=',', dtype=np.float32)
                E = tmp_matrix[1:, 1:]

                if len(states) != len(E) or len(actions) != len(E[0]) or np.isnan(np.sum(E)):
                    logging.error("Wrong file format: wrong E dimensions or nan values present")
                    exit(4)

            except Exception as e:
                logging.error("Wrong file format:" + str(e))
                exit(5)

            if self.show_retrieved_info:
                logging.info("E MATRIX:")
                logging.info(E)

        optimal_policy = parameters['optimal_policy'].split('-')  # Split policy string and save it into a list
        seconds_to_wait = float(parameters['seconds_to_wait'])

        if self.show_retrieved_info:
            logging.info("RL ALGORITHM:\t " + str(parameters['algorithm_used']))
            logging.info("POSSIBLE OPTIMAL POLICY:\t " + str(optimal_policy))

        # time start
        start_time = time.time()

        # ### FOLLOW POLICY ###

        # Follow the found best policy:
        if self.show_retrieved_info:
            logging.info("------------------------------------------")
            logging.info("FOLLOW POLICY")
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\tREQUEST: Setting power off")
        operate_on_bulb("set_power", str("\"off\", \"sudden\", 0"), self.discovery_report, self.discovery_report['protocol'])
        sleep(seconds_to_wait)
        state1, old_props_values = compute_next_state_from_props(FrameworkConfiguration.path, 0, [], self.discovery_report)
        if FrameworkConfiguration.DEBUG:
            logging.debug("\tSTARTING FROM STATE " + str(states[state1]))

        t = 0
        final_policy = []
        final_reward = 0
        final_states = []

        while t < 20:
            final_states.append(states[state1])
            max_action = np.argmax(Q[state1, :])
            final_policy.append(max_action)
            logging.info("\tACTION TO PERFORM " + str(max_action))

            json_string = build_command(method_chosen_index=max_action, select_all_props=False, protocol=self.discovery_report['protocol'])
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\tREQUEST: " + str(json_string))
            reward_from_response = operate_on_bulb_json(json_string, self.discovery_report, self.discovery_report['protocol'])
            sleep(seconds_to_wait)

            state2, new_props_values = compute_next_state_from_props(FrameworkConfiguration.path, state1, old_props_values, self.discovery_report)

            if FrameworkConfiguration.DEBUG:
                logging.debug("\tFROM STATE " + str(states[state1]) + " TO STATE " + str(states[state2]))

            reward_from_props, self.storage_reward = compute_reward_from_states(FrameworkConfiguration.path, state1, state2, self.storage_reward)

            tmp_reward = -1 + reward_from_response + reward_from_props  # -1 for using a command more
            logging.info("\tTMP REWARD: " + str(tmp_reward))
            final_reward += tmp_reward

            if state2 == 5:
                final_states.append(states[state2])
                logging.info("DONE AT TIMESTEP " + str(t))
                break
            state1 = state2
            old_props_values = new_props_values
            t += 1

        # time finish
        final_time = time.time() - start_time

        dict_results = {'timesteps_from_run': t + 1,
                        'reward_from_run': final_reward,
                        'time_from_run': final_time,
                        'policy_from_run': final_policy,
                        'states_from_run': final_states, }

        logging.info("\tRESULTS:")
        logging.info("\t\tTotal time: " + str(final_time))
        logging.info("\t\tLength final policy: " + str(len(final_policy)))
        logging.info("\t\tFinal policy: " + str(final_policy))
        if len(final_policy) <= len(optimal_policy) and final_reward >= 190:
            logging.info("\t\tOptimal policy found with reward: " + str(final_reward))
            return True, dict_results
        else:
            logging.info("\t\tNot optimal policy found with reward: " + str(final_reward))
            return False, dict_results


def main(discovery_report=None):
    format_console_output()
    date_to_retrieve = "2021_02_09_15_25_02_123145422917632"
    if discovery_report is None:
        logging.info("No discovery report found. Analyzing LAN...")

        devices = network_analyzer.analyze_lan()
        # devices = yeelight_analyzer.main()

        if len(devices) > 0:
            logging.info("Found devices")
            for dev in devices:
                parameters = read_parameters_from_output_file(date_to_retrieve)
                if (parameters['protocol'] and dev.protocol == parameters['protocol']) \
                        or (not parameters['protocol'] and dev.protocol == "yeelight"):
                    # TODO Remove check after OR (just temporary, I just added the protocol inside parameters
                    if FrameworkConfiguration.DEBUG:
                        logging.debug("Waiting 5 seconds before verifying optimal policy")
                    sleep(5)
                    RunOutputQParameters(date_to_retrieve=date_to_retrieve, discovery_report=dev.as_dict()).run()

        else:
            logging.error("No device found.")
            exit(-1)
    elif discovery_report['ip']:
        logging.info("Discovery report found: IP" + str(discovery_report['ip']))

        logging.debug("Waiting 5 seconds before verifying optimal policy")
        sleep(5)

        flag, dict_res = RunOutputQParameters(date_to_retrieve=date_to_retrieve, discovery_report=discovery_report).run()
        logging.info(str(dict_res))


if __name__ == '__main__':
    main()
