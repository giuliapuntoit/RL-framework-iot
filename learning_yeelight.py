#!/usr/bin/python
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import pathlib
import socket
from datetime import datetime
import time
import fcntl
import os
import struct
from threading import Thread
from time import sleep

from plot_output_data import PlotOutputData
from run_output_Q_parameters import RunOutputQParameters
from serve_yeelight import ServeYeelight
from utility_yeelight import bulbs_detection_loop, operate_on_bulb, operate_on_bulb_json, \
    compute_reward_from_states, compute_next_state_from_props, display_bulbs

from config import GlobalVar

# Global variables for bulb connection
GlobalVar.detected_bulbs = {}
GlobalVar.bulb_idx2ip = {}
GlobalVar.RUNNING = True
GlobalVar.current_command_id = 1
GlobalVar.MCAST_GRP = '239.255.255.250'

# Global variables for RL
tot_reward = 0


class ReinforcementLearningAlgorithm(object):

    def __init__(self, epsilon=0.6,
                 total_episodes=10,
                 max_steps=100,
                 alpha=0.005,
                 gamma=0.95,
                 lam=0.9,
                 decay_episode=30,
                 decay_value=0.001,
                 show_graphs=False,
                 follow_policy=True,
                 use_old_matrix=False,
                 date_old_matrix='YY_mm_dd_HH_MM_SS',
                 seconds_to_wait=4,
                 follow_partial_policy=False,
                 follow_policy_every_tot_episodes=5,
                 num_actions_to_use=37,
                 algorithm='sarsa'):
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay_episode = decay_episode
        self.decay_value = decay_value
        self.show_graphs = show_graphs
        self.follow_policy = follow_policy
        self.seconds_to_wait = seconds_to_wait
        self.follow_partial_policy = follow_partial_policy
        self.follow_policy_every_tot_episodes = follow_policy_every_tot_episodes
        self.num_actions_to_use = num_actions_to_use
        self.algorithm = algorithm
        # lambda is needed only in case of sarsa(lambda) algorithm
        if self.algorithm == 'sarsa_lambda':
            self.lam = lam
        if date_old_matrix != 'YY_mm_dd_HH_MM_SS':
            self.use_old_matrix = use_old_matrix  # in sarsa lambda also E is needed
            self.date_old_matrix = date_old_matrix  # I should check it is in a correct format
        else:
            self.use_old_matrix = False

    # Function to choose the next action, same for all algorithms
    def choose_action(self, state, Qmatrix):
        # Here I should choose the method
        if np.random.uniform(0, 1) < self.epsilon:
            # print("\t\tSelect the action randomly")
            action = random.randint(0, self.num_actions_to_use - 1)  # don't use the first one
        else:
            # Select maximum, if multiple values select randomly
            # print("\t\tSelect maximum")
            # choose random action between the max ones
            action = np.random.choice(np.where(Qmatrix[state, :] == Qmatrix[state, :].max())[0])
        # The action then should be converted when used into a json_string returned by serve_yeelight
        # action is an index
        return action

    # SARSA
    # Function to learn the Q-value
    def update_sarsa(self, state, state2, reward, action, action2, Qmatrix):
        predict = Qmatrix[state, action]
        target = reward + self.gamma * Qmatrix[state2, action2]
        Qmatrix[state, action] = Qmatrix[state, action] + self.alpha * (target - predict)

    # SARSA(lambda)
    # Function to update the Q-value matrix and the Eligibility matrix
    def update_sarsa_lambda(self, state, state2, reward, action, action2, len_states, len_actions, Qmatrix, Ematrix):
        predict = Qmatrix[state, action]
        target = reward + self.gamma * Qmatrix[state2, action2]
        delta = target - predict
        Ematrix[state, action] = Ematrix[state, action] + 1
        # for all s, a
        for s in range(len_states):
            for a in range(len_actions):
                Qmatrix[s, a] = Qmatrix[s, a] + self.alpha * delta * Ematrix[s, a]
                Ematrix[s, a] = self.gamma * self.lam * Ematrix[s, a]

    # Q-learning
    # Function to learn the Q-value
    def update_qlearning(self, state, state2, reward, action, Qmatrix):
        predict = Qmatrix[state, action]
        maxQ = np.amax(Qmatrix[state2, :])  # find maximum value for the new state
        target = reward + self.gamma * maxQ
        Qmatrix[state, action] = Qmatrix[state, action] + self.alpha * (target - predict)

    def run(self):
        np.set_printoptions(formatter={'float': lambda output: "{0:0.4f}".format(output)})

        # Mi invento questi stati: lampadina parte da accesa, poi accendo, cambio colore, spengo
        states = ["0_off_start", "1_on", "2_rgb", "3_bright", "4_rgb_bright", "5_off_end",
                  "6_invalid"]  # 0 1 2 4 0 optimal path

        optimal = [5, 2, 4, 6]  # optimal policy

        current_date = datetime.now()

        log_dir = 'log'
        pathlib.Path(log_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5 YY_mm_dd_HH_MM_SS'
        log_filename = current_date.strftime(log_dir + '/' + 'log_' + '%Y_%m_%d_%H_%M_%S' + '.log')

        output_Q_params_dir = 'output_Q_parameters'
        pathlib.Path(output_Q_params_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5
        output_Q_filename = current_date.strftime(
            output_Q_params_dir + '/' + 'output_Q_' + '%Y_%m_%d_%H_%M_%S' + '.csv')
        output_parameters_filename = current_date.strftime(
            output_Q_params_dir + '/' + 'output_parameters_' + '%Y_%m_%d_%H_%M_%S' + '.csv')
        output_E_filename = ''
        if self.algorithm == 'sarsa_lambda':
            output_E_filename = current_date.strftime(
                output_Q_params_dir + '/' + 'output_E_' + '%Y_%m_%d_%H_%M_%S' + '.csv')

        output_dir = 'output_csv'
        pathlib.Path(output_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5
        output_filename = current_date.strftime(
            output_dir + '/' + 'output_' + self.algorithm + '_' + '%Y_%m_%d_%H_%M_%S' + '.csv')

        partial_output_filename = current_date.strftime(
            output_dir + '/' + 'partial_output_' + self.algorithm + '_' + '%Y_%m_%d_%H_%M_%S' + '.csv')

        # Write parameters in output_parameters_filename
        with open(output_parameters_filename, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            output_writer.writerow(['algorithm_used', self.algorithm])
            output_writer.writerow(['epsilon', self.epsilon])
            output_writer.writerow(['max_steps', self.max_steps])
            output_writer.writerow(['total_episodes', self.total_episodes])
            output_writer.writerow(['alpha', self.alpha])
            output_writer.writerow(['num_actions_to_use', self.num_actions_to_use])
            output_writer.writerow(['gamma', self.gamma])
            output_writer.writerow(['decay_episode', self.decay_episode])
            output_writer.writerow(['decay_value', self.decay_value])
            output_writer.writerow(['seconds_to_wait', self.seconds_to_wait])
            output_writer.writerow(['optimal_policy', "-".join(str(act) for act in optimal)])

            if self.algorithm == 'sarsa_lambda':
                output_writer.writerow(['lambda', self.lam])

        # SARSA algorithm SINCE algorithm is sarsa

        # Initializing the Q-matrix
        if self.show_graphs:
            print("States are ", len(states))
            print("Actions are ", self.num_actions_to_use)

        Q = np.zeros((len(states), self.num_actions_to_use))

        if self.use_old_matrix:
            # Retrieve from output_Q_data.csv
            # check the format of the matrix is correct
            file_Q = 'output_Q_' + self.date_old_matrix + '.csv'

            try:
                tmp_matrix = np.genfromtxt(output_Q_params_dir + '/' + file_Q, delimiter=',', dtype=np.float32)
                Q_tmp = tmp_matrix[1:, 1:]
                Q = Q_tmp

            except Exception as e:
                print("Wrong file format:", e)
                print("Using an empty Q matrix instead of the old one.")

            if len(states) != len(Q) or self.num_actions_to_use != len(Q[0]) or np.isnan(np.sum(Q)):
                print("Wrong file format: wrong Q dimensions or nan values present")
                print("Using an empty Q matrix instead of the old one.")

        print(Q)

        # Retrieve from output_E_data.csv
        # check the format of the matrix is correct

        E = []
        if self.algorithm == 'sarsa_lambda':
            E = np.zeros((len(states), self.num_actions_to_use))  # trace for state action pairs

            if self.use_old_matrix:
                # Retrieve from output_E_data.csv
                # check the format of the matrix is correct
                file_E = 'output_E_' + self.date_old_matrix + '.csv'

                try:
                    tmp_matrix = np.genfromtxt(output_Q_params_dir + '/' + file_E, delimiter=',', dtype=np.float32)
                    E_tmp = tmp_matrix[1:, 1:]
                    E = E_tmp

                except Exception as e:
                    print("Wrong file format:", e)
                    print("Using an empty E matrix instead of the old one")

                if len(states) != len(E) or self.num_actions_to_use != len(E[0]) or np.isnan(np.sum(E)):
                    print("Wrong file format: wrong E dimensions or nan values present")
                    print("Using an empty E matrix instead of the old one")

            print(E)

        start_time = time.time()

        x = range(0, self.total_episodes)
        y_timesteps = []
        y_reward = []
        y_cum_reward = []

        cumulative_reward = 0

        # Write into output_filename the header: Episodes, Reward, CumReward, Timesteps
        with open(output_filename, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(['Episodes', 'Reward', 'CumReward', 'Timesteps'])

        if self.follow_partial_policy:
            with open(partial_output_filename, mode='w') as partial_output_file:
                output_writer = csv.writer(partial_output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow(
                    ['CurrentEpisode', 'Timesteps', 'ObtainedReward', 'Time', 'PolicySelected', 'StatesPassed'])

        # Starting the SARSA learning
        for episode in range(self.total_episodes):
            print("----------------------------------------------------------------")
            print("Episode", episode)
            sleep(60)
            t = 0
            # Turn off the lamp
            print("\t\tREQUEST: Setting power off")
            operate_on_bulb(idLamp, "set_power", str("\"off\", \"sudden\", 0"))
            sleep(self.seconds_to_wait)
            state1, old_props_values = compute_next_state_from_props(idLamp, 0, [])
            print("\tSTARTING FROM STATE", states[state1])
            action1 = self.choose_action(state1, Q)
            done = False
            reward_per_episode = 0
            # exploration reduces every 50 episodes
            if (episode + 1) % self.decay_episode == 0:  # configurable parameter
                self.epsilon = self.epsilon - self.decay_value * self.epsilon  # could be another configurable parameter, decay of epsilon

            while t < self.max_steps:
                # Getting the next state
                if self.algorithm == 'qlearning':
                    action1 = self.choose_action(state1, Q)

                # print("\t\tDoing an action")
                json_string = ServeYeelight(id_lamp=idLamp, method_chosen_index=action1).run()
                print("\t\tREQUEST:", str(json_string))
                reward_from_response = operate_on_bulb_json(idLamp, json_string)
                sleep(self.seconds_to_wait)

                state2, new_props_values = compute_next_state_from_props(idLamp, state1, old_props_values)
                print("\tFROM STATE", states[state1], "TO STATE", states[state2])

                reward_from_states = compute_reward_from_states(state1, state2)
                tmp_reward = -1 + reward_from_response + reward_from_states  # -1 for using a command more

                if state2 == 5:
                    done = True

                if self.algorithm == 'sarsa_lambda':
                    # Choosing the next action
                    action2 = self.choose_action(state2, Q)

                    # Learning the Q-value
                    self.update_sarsa_lambda(state1, state2, tmp_reward, action1, action2, len(states),
                                             self.num_actions_to_use, Q, E)

                elif self.algorithm == 'qlearning':
                    action2 = -1  # Invalid action to avoid warnings
                    # Learning the Q-value
                    self.update_qlearning(state1, state2, tmp_reward, action1, Q)

                else:
                    # SARSA as default algorithm

                    # Choosing the next action
                    action2 = self.choose_action(state2, Q)

                    # Learning the Q-value
                    self.update_sarsa(state1, state2, tmp_reward, action1, action2, Q)

                with open(log_filename, "a") as write_file:
                    write_file.write("\nTimestep " + str(t) + " finished.")
                    write_file.write(" Temporary reward: " + str(tmp_reward))
                    write_file.write(" Previous state: " + str(state1))
                    write_file.write(" Current state: " + str(state2))
                    write_file.write(" Performed action: " + str(action1))
                    if self.algorithm != 'qlearning':
                        write_file.write(" Next action: " + str(action2))

                state1 = state2
                old_props_values = new_props_values
                if self.algorithm != 'qlearning':
                    action1 = action2

                # Updating the respective values
                t += 1
                reward_per_episode += tmp_reward
                # If at the end of learning process
                if done:
                    break
            cumulative_reward += reward_per_episode
            y_timesteps.append(t - 1)
            y_cum_reward.append(cumulative_reward)
            y_reward.append(reward_per_episode)
            with open(log_filename, "a") as write_file:
                write_file.write("\nEpisode " + str(episode) + " finished.\n")
            with open(output_filename, mode="a") as output_file:
                output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow([episode, reward_per_episode, cumulative_reward, t - 1])  # Episode or episode+1?
            print("\tREWARD OF THE EPISODE:", reward_per_episode)

            if self.follow_partial_policy:
                if (episode + 1) % self.follow_policy_every_tot_episodes == 0:  # ogni 2 episodi salva in Q e segue la policy (if <= len(optimal policy) end of learning)
                    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                    print("\tFOLLOW PARTIAL POLICY AT EPISODE", episode)
                    sleep(10)
                    header = ['Q']  # for correct output structure
                    for i in range(0, self.num_actions_to_use):
                        json_string = ServeYeelight(id_lamp=idLamp, method_chosen_index=i).run()
                        header.append(json.loads(json_string)['method'])

                    with open(output_Q_filename, "w") as output_Q_file:
                        output_Q_writer = csv.writer(output_Q_file, delimiter=',', quotechar='"',
                                                     quoting=csv.QUOTE_NONE)
                        output_Q_writer.writerow(header)
                        for index, stat in enumerate(states):
                            row = [stat]
                            for val in Q[index]:
                                row.append("%.4f" % val)
                            output_Q_writer.writerow(row)
                    found_policy, dict_results = RunOutputQParameters(id_lamp=idLamp,
                                                                      date_to_retrieve=current_date.strftime(
                                                                          '%Y_%m_%d_%H_%M_%S'),
                                                                      show_retrieved_info=False).run()
                    with open(partial_output_filename, mode="a") as partial_output_file:
                        output_writer = csv.writer(partial_output_file, delimiter=',', quotechar='"',
                                                   quoting=csv.QUOTE_MINIMAL)
                        output_writer.writerow(
                            [episode, dict_results['timesteps_from_run'], dict_results['reward_from_run'],
                             dict_results['time_from_run'], dict_results['policy_from_run'],
                             dict_results['states_from_run']])  # Episode or episode+1?

                    if found_policy:
                        # I could stop here if found good policy, could continue if you think you could find a better one
                        # Questo nel nostro caso essendo l'esecuzione molto lenta appena apprendo non ho bisogno di andare avanti
                        # In una situazione reale probabilmente non lo posso fare perché non so quale sia la policy ottimale
                        pass

        # Print and save the Q-matrix inside output_Q_data.csv file
        print(Q)
        header = ['Q']  # for correct output structure
        for i in range(0, self.num_actions_to_use):
            json_string = ServeYeelight(id_lamp=idLamp, method_chosen_index=i).run()
            header.append(json.loads(json_string)['method'])

        with open(output_Q_filename, "w") as output_Q_file:
            output_Q_writer = csv.writer(output_Q_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            output_Q_writer.writerow(header)
            for index, stat in enumerate(states):
                row = [stat]
                for val in Q[index]:
                    row.append("%.4f" % val)
                output_Q_writer.writerow(row)

        # Only for sarsa(lambda)
        if self.algorithm == 'sarsa_lambda':
            # Print and save the E-matrix inside output_E_data.csv file
            print("E matrix")
            print(E)
            # Use same header as before with the first cell different
            header[0] = 'E'  # for correct output structure

            with open(output_E_filename, "w") as output_E_file:
                output_E_writer = csv.writer(output_E_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
                output_E_writer.writerow(header)
                for index, stat in enumerate(states):
                    row = [stat]
                    for val in E[index]:
                        row.append("%.4f" % val)
                    output_E_writer.writerow(row)

        with open(log_filename, "a") as write_file:
            write_file.write("\nTotal time of %s seconds." % (time.time() - start_time))

        sleep(5)  # wait for writing to files
        if self.show_graphs:
            PlotOutputData(date_to_retrieve=current_date.strftime('%Y_%m_%d_%H_%M_%S'), separate_plots=False).run()

        # Following the best policy found
        if self.follow_policy:
            RunOutputQParameters(id_lamp=idLamp, date_to_retrieve=current_date.strftime('%Y_%m_%d_%H_%M_%S')).run()


if __name__ == '__main__':
    # Socket setup
    GlobalVar.scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fcntl.fcntl(GlobalVar.scan_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    GlobalVar.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    GlobalVar.listen_socket.bind(("", 1982))
    fcntl.fcntl(GlobalVar.listen_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    # GlobalVar.scan_socket.settimeout(GlobalVar.timeout)  # set 2 seconds of timeout -> could be a configurable parameter
    # GlobalVar.listen_socket.settimeout(GlobalVar.timeout)
    mreq = struct.pack("4sl", socket.inet_aton(GlobalVar.MCAST_GRP), socket.INADDR_ANY)
    GlobalVar.listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # Give socket some time to set up
    sleep(2)

    # First discover the lamp and connect to the lamp, with the bulb detection thread
    detection_thread = Thread(target=bulbs_detection_loop)
    detection_thread.start()
    # Give detection thread some time to collect bulb info
    sleep(10)

    # Show discovered lamps
    display_bulbs()
    print(GlobalVar.bulb_idx2ip)

    max_wait = 0
    while len(GlobalVar.bulb_idx2ip) == 0 and max_wait < 10:
        # Wait for 10 seconds to see if some bulb is present
        # The number of seconds could be extended if necessary
        sleep(1)
        max_wait += 1
    if len(GlobalVar.bulb_idx2ip) == 0:
        print("Bulb list is empty.")
    else:
        # if some bulb was found, take first bulb
        display_bulbs()
        idLamp = list(GlobalVar.bulb_idx2ip.keys())[0]

        print("Waiting 5 seconds before using RL algorithm")
        sleep(5)

        # Stop bulb detection loop
        GlobalVar.RUNNING = False

        print("\n############# Starting RL algorithm grid search #############")
        # Check if after 20 episodes it's able to follow the policy
        # Collecting data for future graphs
        # for eps in [0.3, 0.6, 0.9]:
        #     for alp in [0.005, 0.05, 0.5]:
        #         for gam in [0.45, 0.75, 0.95]:
        eps = 0.6
        alp = 0.005
        gam = 0.95
        max_st = 100
        for secs in [0.01, 0.1, 2]:
            ReinforcementLearningAlgorithm(max_steps=max_st, total_episodes=20,
                                           num_actions_to_use=37,
                                           seconds_to_wait=secs,
                                           epsilon=eps,
                                           alpha=alp,
                                           gamma=gam,
                                           show_graphs=False,
                                           follow_policy=False,
                                           follow_partial_policy=True,
                                           follow_policy_every_tot_episodes=2,
                                           algorithm='sarsa').run()  # 'sarsa' 'sarsa_lambda' 'qlearning'
            sleep(100)

        # Then max steps and seconds to wait manually, with best configured parameters (should be 27 runs)

        # print("\n############# Starting RL algorithm #############")
        # ReinforcementLearningAlgorithm(max_steps=10, total_episodes=5,
        #                                num_actions_to_use=37,
        #                                seconds_to_wait=7,
        #                                show_graphs=False,
        #                                follow_policy=True,
        #                                follow_partial_policy=True,
        #                                follow_policy_every_tot_episodes=2,
        #                                use_old_matrix=True,
        #                                date_old_matrix='2020_10_26_01_51_42',
        #                                algorithm='sarsa').run()  # 'sarsa' 'sarsa_lambda' 'qlearning'
        # print('sarsa end')
        # sleep(300)
        # ReinforcementLearningAlgorithm(max_steps=200, total_episodes=20,
        #                                num_actions_to_use=37,
        #                                seconds_to_wait=7,
        #                                show_graphs=False,
        #                                follow_policy=True,
        #                                algorithm='sarsa_lambda').run()  # 'sarsa' 'sarsa_lambda' 'qlearning'
        # print('sarsa_lambda end')
        # sleep(300)
        # ReinforcementLearningAlgorithm(max_steps=200, total_episodes=20,
        #                                num_actions_to_use=37,
        #                                seconds_to_wait=7,
        #                                show_graphs=False,
        #                                follow_policy=True,
        #                                algorithm='qlearning').run()  # 'sarsa' 'sarsa_lambda' 'qlearning'
        # print('qlearning end')
        print("############# Finish RL algorithm #############")

    # Goal achieved, tell detection thread to quit and wait
    RUNNING = False  # non credo serva di nuovo
    detection_thread.join()
    # Done

# Set number of ri-transmissions, with a configurable parameter
# TODO i test potrebbero testare che i parametri, stati, azioni ecc passati in input siano poi quelli scritti in output, \
#  che ci sia un certo file, che il nome del file corrisponde alla data giusta, al nome dell'algo ecc
# TODO Usa gli enum sia per gli state che per le actions!!! Le actions le puoi retrievare a run time, così poi le salvo come stringhe
