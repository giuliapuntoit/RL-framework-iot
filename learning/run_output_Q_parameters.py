"""
    Class to follow the best policy found by a learning process
"""

import numpy as np
import csv
import time
from time import sleep
from config import FrameworkConfiguration

# Q will be read from output_Q_date.csv
# Retrieve actions and state from output_Q_date.csv
# Statistics to compute Q will be read from output_parameters_date.csv

# Identify which RL algorithm was used and use it

from device_communication.api_yeelight import operate_on_bulb, operate_on_bulb_json
from discovery import network_analyzer
from plotter.support_plotter import read_parameters_from_output_file
from state_machine.state_machine_yeelight import compute_reward_from_states, compute_next_state_from_props
from request_builder.builder_yeelight import BuilderYeelight


class RunOutputQParameters(object):
    def __init__(self, date_to_retrieve='YY_mm_dd_HH_MM_SS', show_retrieved_info=True, discovery_report=None):
        self.show_retrieved_info = show_retrieved_info
        self.discovery_report = discovery_report
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            print("Invalid date")
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
            print("Wrong file format:", e)
            exit(1)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        if self.show_retrieved_info:
            print("STATES:\n\t", states)
            print("ACTIONS:\n\t", actions)
            print("Q MATRIX:")
            print(Q)

        if len(states) != len(Q) or len(actions) != len(Q[0]) or np.isnan(np.sum(Q)):
            print("Wrong file format: wrong Q dimensions or nan values present")
            exit(2)

        with open(directory + '/' + file_parameters, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

        if self.show_retrieved_info:
            print("USED PARAMETERS:\n\t", parameters)  # For now the are all strings

        if parameters['num_actions_to_use'] is not None and len(actions) != int(parameters['num_actions_to_use']):
            print("Different number of actions used")
            exit(3)

        if parameters['algorithm_used'] == 'sarsa_lambda' or parameters['algorithm_used'] == 'qlearning_lambda':

            file_E = 'output_E_' + self.date_to_retrieve + '.csv'

            E = []

            try:
                tmp_matrix = np.genfromtxt(directory + '/' + file_E, delimiter=',', dtype=np.float32)
                E = tmp_matrix[1:, 1:]

                if len(states) != len(E) or len(actions) != len(E[0]) or np.isnan(np.sum(E)):
                    print("Wrong file format: wrong E dimensions or nan values present")
                    exit(4)

            except Exception as e:
                print("Wrong file format:", e)
                exit(5)

            if self.show_retrieved_info:
                print("E MATRIX:")
                print(E)

        optimal_policy = parameters['optimal_policy'].split('-')  # Split policy string and save it into a list
        seconds_to_wait = float(parameters['seconds_to_wait'])

        if self.show_retrieved_info:
            print("RL ALGORITHM:\n\t", parameters['algorithm_used'])
            print("POSSIBLE OPTIMAL POLICY:\n\t", optimal_policy)

        # time start
        start_time = time.time()

        # ### FOLLOW POLICY ###

        # Follow the found best policy:
        if self.show_retrieved_info:
            print("------------------------------------------")
            print("FOLLOW POLICY")
        print("\t\tREQUEST: Setting power off")
        operate_on_bulb("set_power", str("\"off\", \"sudden\", 0"), None)
        sleep(seconds_to_wait)
        state1, old_props_values = compute_next_state_from_props(0, [], None)
        print("\tSTARTING FROM STATE", states[state1])

        t = 0
        final_policy = []
        final_reward = 0
        final_states = []

        while t < 20:
            final_states.append(states[state1])
            max_action = np.argmax(Q[state1, :])
            final_policy.append(max_action)
            print("\tACTION TO PERFORM", max_action)

            json_string = BuilderYeelight(method_chosen_index=max_action).run()
            print("\t\tREQUEST:", str(json_string))
            reward_from_response = operate_on_bulb_json(json_string, self.discovery_report)
            sleep(seconds_to_wait)

            state2, new_props_values = compute_next_state_from_props(state1, old_props_values, None)

            print("\tFROM STATE", states[state1], "TO STATE", states[state2])

            reward_from_props, self.storage_reward = compute_reward_from_states(state1, state2, self.storage_reward)

            tmp_reward = -1 + reward_from_response + reward_from_props  # -1 for using a command more
            print("\tTMP REWARD:", str(tmp_reward))
            final_reward += tmp_reward

            if state2 == 5:
                final_states.append(states[state2])
                print("DONE AT TIMESTEP", t)
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

        print("\tRESULTS:")
        print("\t\tTotal time:", final_time)
        print("\t\tLength final policy:", len(final_policy))
        print("\t\tFinal policy:", final_policy)
        if len(final_policy) <= len(optimal_policy) and final_reward >= 190:
            print("\t\tOptimal policy found with reward:", final_reward)
            return True, dict_results
        else:
            print("\t\tNot optimal policy found with reward:", final_reward)
            return False, dict_results


def main(discovery_report=None):
    date_to_retrieve = "2020_10_30_02_10_16"
    if discovery_report is None:
        print("No discovery report found. Analyzing LAN...")

        devices = network_analyzer.analyze_lan()

        if len(devices) > 0:
            print("Found devices")
            for dev in devices:
                parameters = read_parameters_from_output_file(date_to_retrieve)
                if (parameters['protocol'] and dev.protocol == parameters['protocol']) \
                        or (not parameters['protocol'] and dev.protocol == "yeelight"):
                    # TODO Remove check after OR (just temporary, I just added the protocol inside parameters
                    print("Waiting 5 seconds before verifying optimal policy")
                    sleep(5)
                    RunOutputQParameters(date_to_retrieve=date_to_retrieve, discovery_report=discovery_report).run()

        else:
            print("No device found.")
            exit(-1)
    elif discovery_report['ip']:
        print("Discovery report found: IP", discovery_report['ip'])

        print("Waiting 5 seconds before verifying optimal policy")
        sleep(5)

        flag, dict_res = RunOutputQParameters(date_to_retrieve=date_to_retrieve, discovery_report=discovery_report).run()
        print(dict_res)


if __name__ == '__main__':
    main()


# TODO questo script è da vedere se worka e da sistemare le print? O forse no?
