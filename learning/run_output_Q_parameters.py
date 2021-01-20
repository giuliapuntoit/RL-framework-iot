import json
import os
import struct
import numpy as np
import csv
import time
from threading import Thread
from time import sleep
import socket
import fcntl
from config import FrameworkConfiguration

# Follow the best policy found by a learning process

# Q will be read from output_Q_data.csv
# Retrieve actions and state from output_Q_data.csv
# Statistics to compute Q will be read from output_parameters_data.csv

# Identify which RL algorithm was used and use it

from device_communication.api_yeelight import bulbs_detection_loop, display_bulbs, operate_on_bulb, operate_on_bulb_json
from state_machine.state_machine_yeelight import compute_reward_from_states, compute_next_state_from_props
from request_builder.builder_yeelight import ServeYeelight


class RunOutputQParameters(object):
    def __init__(self, id_lamp=0, date_to_retrieve='YY_mm_dd_HH_MM_SS', show_retrieved_info=True):
        self.id_lamp = id_lamp
        self.show_retrieved_info = show_retrieved_info
        if date_to_retrieve != 'YY_mm_dd_HH_MM_SS':
            self.date_to_retrieve = date_to_retrieve  # Date must be in format %Y_%m_%d_%H_%M_%S
        else:
            print("Invalid date")
            exit(1)

    def run(self):
        directory = FrameworkConfiguration.directory + 'output/output_Q_parameters'
        file_Q = 'output_Q_' + self.date_to_retrieve + '.csv'
        file_parameters = 'output_parameters_' + self.date_to_retrieve + '.csv'

        actions = []
        states = []
        Q = []

        parameters = {}

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
        operate_on_bulb(self.id_lamp, "set_power", str("\"off\", \"sudden\", 0"))
        sleep(seconds_to_wait)
        state1, old_props_values = compute_next_state_from_props(self.id_lamp, 0, [])
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

            json_string = ServeYeelight(id_lamp=self.id_lamp, method_chosen_index=max_action).run()
            print("\t\tREQUEST:", str(json_string))
            reward_from_response = operate_on_bulb_json(self.id_lamp, json_string)
            sleep(seconds_to_wait)

            state2, new_props_values = compute_next_state_from_props(self.id_lamp, state1, old_props_values)

            print("\tFROM STATE", states[state1], "TO STATE", states[state2])

            reward_from_props = compute_reward_from_states(state1, state2)

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


if __name__ == '__main__':
    # Socket setup
    FrameworkConfiguration.scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fcntl.fcntl(FrameworkConfiguration.scan_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    FrameworkConfiguration.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    FrameworkConfiguration.listen_socket.bind(("", 1982))
    fcntl.fcntl(FrameworkConfiguration.listen_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    # GlobalVar.scan_socket.settimeout(GlobalVar.timeout)  # set 2 seconds of timeout
    # GlobalVar.listen_socket.settimeout(GlobalVar.timeout)
    mreq = struct.pack("4sl", socket.inet_aton(FrameworkConfiguration.MCAST_GRP), socket.INADDR_ANY)
    FrameworkConfiguration.listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # First discover the lamp and connect to the lamp
    # Start the bulb detection thread
    detection_thread = Thread(target=bulbs_detection_loop)
    detection_thread.start()
    # Give detection thread some time to collect bulb info
    sleep(10)

    # Show discovered lamps
    display_bulbs()

    print(FrameworkConfiguration.bulb_idx2ip)
    max_wait = 0
    while len(FrameworkConfiguration.bulb_idx2ip) == 0 and max_wait < 10:
        sleep(1)
        max_wait += 1
    if len(FrameworkConfiguration.bulb_idx2ip) == 0:
        print("Bulb list is empty.")
    else:
        # If some bulbs were found inside the network do something
        display_bulbs()
        idLamp = list(FrameworkConfiguration.bulb_idx2ip.keys())[0]

        print("Waiting 5 seconds before verifying optimal policy")
        sleep(5)

        FrameworkConfiguration.RUNNING = False

        RunOutputQParameters(id_lamp=idLamp, date_to_retrieve="2020_10_30_02_10_16").run()

    # Goal achieved, tell detection thread to quit and wait
    RUNNING = False
    detection_thread.join()
    # Done
