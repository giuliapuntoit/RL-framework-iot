"""
    Class that plays the Reinforcement Learning agent
"""

# !/usr/bin/python
import csv
import pprint
import threading
import numpy as np
import json
import random
import pathlib
from datetime import datetime
import time
import copy
from time import sleep
import logging
import sys

from formatter_for_output import format_console_output
from plotter.plot_output_data import PlotOutputData
from learning.run_output_Q_parameters import RunOutputQParameters
from request_builder.builder import build_command
from device_communication.client import operate_on_bulb, operate_on_bulb_json
from state_machine.state_machine_yeelight import compute_reward_from_states, compute_next_state_from_props, get_states, \
    get_optimal_policy, get_optimal_path

from config import FrameworkConfiguration


class ReinforcementLearningAlgorithm(object):

    def __init__(self, discovery_report, thread_id):
        self.discovery_report = discovery_report
        self.total_episodes = FrameworkConfiguration.total_episodes
        self.max_steps = FrameworkConfiguration.max_steps
        self.epsilon = FrameworkConfiguration.epsilon
        self.alpha = FrameworkConfiguration.alpha
        self.gamma = FrameworkConfiguration.gamma
        self.decay_episode = FrameworkConfiguration.decay_episode
        self.decay_value = FrameworkConfiguration.decay_value
        self.show_graphs = FrameworkConfiguration.show_graphs
        self.follow_policy = FrameworkConfiguration.follow_policy
        self.seconds_to_wait = FrameworkConfiguration.seconds_to_wait
        self.follow_partial_policy = FrameworkConfiguration.follow_partial_policy
        self.follow_policy_every_tot_episodes = FrameworkConfiguration.follow_policy_every_tot_episodes
        self.num_actions_to_use = FrameworkConfiguration.num_actions_to_use
        self.algorithm = FrameworkConfiguration.algorithm
        # lambda is needed only in case of sarsa(lambda) or Q(lambda) algorithms
        if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
            self.lam = FrameworkConfiguration.lam
        if FrameworkConfiguration.date_old_matrix != 'YY_mm_dd_HH_MM_SS':
            self.use_old_matrix = FrameworkConfiguration.use_old_matrix  # in sarsa lambda also E is needed
            self.date_old_matrix = FrameworkConfiguration.date_old_matrix  # I should check it is in a correct format
        else:
            self.use_old_matrix = False
        self.current_date = datetime.now()
        if thread_id:
            self.thread_id = thread_id
        self.id_for_output = '%Y_%m_%d_%H_%M_%S' + '_' + str(self.thread_id)
        self.storage_reward = 0  # temporary storage variable

    def choose_action(self, state, q_matrix):
        """
        Function to choose the next action, same for all algorithms
        """
        # Here I should choose the method
        if np.random.uniform(0, 1) < self.epsilon:
            # print("\t\tSelect the action randomly")
            action = random.randint(0, self.num_actions_to_use - 1)  # don't use the first one
        else:
            # Select maximum, if multiple values select randomly
            # print("\t\tSelect maximum")
            # choose random action between the max ones
            action = np.random.choice(np.where(q_matrix[state, :] == q_matrix[state, :].max())[0])
        # The action then should be converted when used into a json_string returned by builder_yeelight
        # action is an index
        return action

    def update_sarsa(self, state, state_2, reward, action, action_2, q_matrix):
        """
        SARSA function to learn the Q-value
        """
        predict = q_matrix[state, action]
        target = reward + self.gamma * q_matrix[state_2, action_2]
        q_matrix[state, action] = q_matrix[state, action] + self.alpha * (target - predict)

    def update_sarsa_lambda(self, state, state_2, reward, action, action_2, len_states, len_actions, q_matrix,
                            e_matrix):
        """
        SARSA(lambda) function to update the Q-value matrix and the Eligibility matrix
        """
        predict = q_matrix[state, action]
        target = reward + self.gamma * q_matrix[state_2, action_2]
        delta = target - predict
        e_matrix[state, action] = e_matrix[state, action] + 1
        # For all s, a
        for s in range(len_states):
            for a in range(len_actions):
                q_matrix[s, a] = q_matrix[s, a] + self.alpha * delta * e_matrix[s, a]
                e_matrix[s, a] = self.gamma * self.lam * e_matrix[s, a]

    def update_qlearning_lambda(self, state, state_2, reward, action, action_2, len_states, len_actions, q_matrix,
                                e_matrix):
        """
        Q-learning(lambda) (Watkins's Q(lambda) algorithm) function to update the Q-value matrix and the Eligibility matrix
        """
        predict = q_matrix[state, action]
        maxQ = np.amax(q_matrix[state_2, :])  # Find maximum value for the new state Q(s', a*)
        maxIndex = np.argmax(q_matrix[state_2, :])  # Find index of the maximum value a*
        target = reward + self.gamma * maxQ
        delta = target - predict
        e_matrix[state, action] = e_matrix[state, action] + 1
        # For all s, a
        for s in range(len_states):
            for a in range(len_actions):
                q_matrix[s, a] = q_matrix[s, a] + self.alpha * delta * e_matrix[s, a]
                if action_2 == maxIndex:
                    e_matrix[s, a] = self.gamma * self.lam * e_matrix[s, a]
                else:
                    e_matrix[s, a] = 0

    def update_qlearning(self, state, state_2, reward, action, q_matrix):
        """
        # Q-learning function to learn the Q-value
        """
        predict = q_matrix[state, action]
        maxQ = np.amax(q_matrix[state_2, :])  # Find maximum value for the new state
        target = reward + self.gamma * maxQ
        q_matrix[state, action] = q_matrix[state, action] + self.alpha * (target - predict)

    def initialize_log_files(self, output_directory, log_directory):
        """
        Get log filenames and build non-existing directories
        """
        log_dir = FrameworkConfiguration.directory + output_directory + '/' + log_directory
        pathlib.Path(log_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5 YY_mm_dd_HH_MM_SS'

        log_filename = self.current_date.strftime(log_dir + '/' + 'log_' + self.id_for_output + '.log')
        log_date_filename = FrameworkConfiguration.directory + output_directory + '/log_date.log'

        return log_filename, log_date_filename

    def initialize_output_q_params_files(self, output_directory, q_params_directory):
        """
        Get output filenames for saving Q and parameters and build non-existing directories
        """
        output_Q_params_dir = FrameworkConfiguration.directory + output_directory + '/' + q_params_directory
        pathlib.Path(output_Q_params_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5

        output_Q_filename = self.current_date.strftime(
            output_Q_params_dir + '/' + 'output_Q_' + self.id_for_output + '.csv')
        output_parameters_filename = self.current_date.strftime(
            output_Q_params_dir + '/' + 'output_parameters_' + self.id_for_output + '.csv')
        output_E_filename = ''
        if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
            output_E_filename = self.current_date.strftime(
                output_Q_params_dir + '/' + 'output_E_' + self.id_for_output + '.csv')

        return output_Q_filename, output_parameters_filename, output_E_filename

    def initialize_output_csv_files(self, output_directory, output_csv_directory):
        """
        Get output filenames for saving all episodes result and build non-existing directories
        """
        output_dir = FrameworkConfiguration.directory + output_directory + '/' + output_csv_directory
        pathlib.Path(output_dir + '/').mkdir(parents=True, exist_ok=True)  # for Python > 3.5
        output_filename = self.current_date.strftime(
            output_dir + '/' + 'output_' + self.algorithm + '_' + self.id_for_output + '.csv')

        partial_output_filename = self.current_date.strftime(
            output_dir + '/' + 'partial_output_' + self.algorithm + '_' + self.id_for_output + '.csv')
        return output_filename, partial_output_filename

    def write_date_id_to_log(self, log_date_filename):
        """
        Write the identifier of files (date) and corresponding algorithm to log_date.log file
        """
        with open(log_date_filename, mode='a') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
            output_writer.writerow([self.current_date.strftime(self.id_for_output), self.algorithm])

    def write_params_to_output_file(self, output_parameters_filename, optimal_policy, optimal_path):
        """
        Write all parameters of the algorithm to output file
        """
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
            output_writer.writerow(['optimal_policy', "-".join(str(act) for act in optimal_policy)])
            output_writer.writerow(['optimal_path', "-".join(str(pat) for pat in optimal_path)])
            output_writer.writerow(['path', FrameworkConfiguration.path])
            output_writer.writerow(['protocol', self.discovery_report['protocol']])

            if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
                output_writer.writerow(['lambda', self.lam])

    def retrieve_old_q_matrix(self, output_directory, q_params_directory, len_states, len_actions, empty_matrix):
        """
        Retrieve old save Q matrix
        """
        file_Q = 'output_Q_' + self.date_old_matrix + '.csv'
        try:
            output_Q_params_dir = FrameworkConfiguration.directory + output_directory + '/' + q_params_directory
            tmp_matrix = np.genfromtxt(output_Q_params_dir + '/' + file_Q, delimiter=',', dtype=np.float32)
            Q_tmp = tmp_matrix[1:, 1:]
            Q = copy.deepcopy(Q_tmp)

        except Exception as e:
            logging.warning("Wrong file format: " + str(e))
            logging.warning("Using an empty Q matrix instead of the old one.")
            return empty_matrix

        # Check the format of the matrix is correct
        if len_states != len(Q) or len_actions != len(Q[0]) or np.isnan(np.sum(Q)):
            logging.warning("Wrong file format: wrong Q dimensions or nan values present")
            logging.warning("Using an empty Q matrix instead of the old one.")
            return empty_matrix

        return Q

    def retrieve_old_e_matrix(self, output_directory, q_params_directory, len_states, len_actions, empty_matrix):
        """
        Retrieve old save Q matrix
        """
        file_E = 'output_E_' + self.date_old_matrix + '.csv'
        try:
            output_Q_params_dir = FrameworkConfiguration.directory + output_directory + '/' + q_params_directory
            tmp_matrix = np.genfromtxt(output_Q_params_dir + '/' + file_E, delimiter=',', dtype=np.float32)
            E_tmp = tmp_matrix[1:, 1:]
            E = copy.deepcopy(E_tmp)

        except Exception as e:
            logging.warning("Wrong file format: " + str(e))
            logging.warning("Using an empty E matrix instead of the old one.")
            return empty_matrix

        # Check the format of the matrix is correct
        if len_states != len(E) or len_actions != len(E[0]) or np.isnan(np.sum(E)):
            logging.warning("Wrong file format: wrong E dimensions or nan values present")
            logging.warning("Using an empty E matrix instead of the old one.")
            return empty_matrix
        return E

    def write_headers_to_output_files(self, output_filename, partial_output_filename):
        """
        Write headers to output csv files
        """
        with open(output_filename, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(['Episodes', 'Reward', 'CumReward', 'Timesteps'])

        if self.follow_partial_policy:
            with open(partial_output_filename, mode='w') as partial_output_file:
                output_writer = csv.writer(partial_output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                output_writer.writerow(
                    ['CurrentEpisode', 'Timesteps', 'ObtainedReward', 'Time', 'PolicySelected', 'StatesPassed'])

    def set_initial_state(self):
        """
        Set device to starting state (e.g. power off)
        """
        num_actions = 0
        if FrameworkConfiguration.path == 3:
            # Special initial configuration for visual checks on the bulb
            # ONLY FOR PATH 3
            operate_on_bulb("set_power", str("\"on\", \"sudden\", 0"), self.discovery_report, self.discovery_report['protocol'])
            num_actions += 1
            sleep(self.seconds_to_wait)
            operate_on_bulb("set_rgb", str("255" + ", \"sudden\", 500"), self.discovery_report, self.discovery_report['protocol'])
            num_actions += 1
            sleep(self.seconds_to_wait)
        elif FrameworkConfiguration.path == 4:
            # Special initial configuration for for path 4, starting to power on
            # ONLY FOR PATH 4
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\tREQUEST: Setting power on")
            operate_on_bulb("set_power", str("\"on\", \"sudden\", 0"), self.discovery_report,
                            self.discovery_report['protocol'])
            num_actions += 1
            sleep(self.seconds_to_wait)
            return num_actions

        # Turn off the lamp
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\tREQUEST: Setting power off")
        operate_on_bulb("set_power", str("\"off\", \"sudden\", 0"), self.discovery_report, self.discovery_report['protocol'])
        num_actions += 1
        return num_actions

    def write_log_file(self, log_filename, t, tmp_reward, state1, state2, action1, action2):
        """
        Write data at each time step
        """
        with open(log_filename, "a") as write_file:
            write_file.write("\nTimestep " + str(t) + " finished.")
            write_file.write(" Temporary reward: " + str(tmp_reward))
            write_file.write(" Previous state: " + str(state1))
            write_file.write(" Current state: " + str(state2))
            write_file.write(" Performed action: " + str(action1))
            if self.algorithm != 'qlearning':
                write_file.write(" Next action: " + str(action2))

    def write_episode_summary(self, log_filename, output_filename, episode, reward_per_episode, cumulative_reward, t):
        """
        Write data at the end of each episode
        """
        with open(log_filename, "a") as write_file:
            write_file.write("\nEpisode " + str(episode) + " finished.\n")
        with open(output_filename, mode="a") as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow([episode, reward_per_episode, cumulative_reward, t - 1])

    def save_matrix(self, output_filename, states, matrix, label):
        """
        Save Q-matrix
        """
        header = [label]  # For correct output structure
        for i in range(0, self.num_actions_to_use):
            json_string = build_command(method_chosen_index=i, select_all_props=False, protocol=self.discovery_report['protocol'])
            header.append(json.loads(json_string)['method'])

        with open(output_filename, "w") as output_matrix_file:
            output_matrix_writer = csv.writer(output_matrix_file, delimiter=',', quotechar='"',
                                              quoting=csv.QUOTE_NONE)
            output_matrix_writer.writerow(header)
            for index, stat in enumerate(states):
                row = [stat]
                for val in matrix[index]:
                    row.append("%.4f" % val)
                output_matrix_writer.writerow(row)

    def run(self):
        """
        Run RL algorithm
        """

        # INITIALIZATION PHASE
        np.set_printoptions(formatter={'float': lambda output: "{0:0.4f}".format(output)})

        # Obtain data about states, path and policy
        states = get_states(FrameworkConfiguration.path)
        optimal_policy = get_optimal_policy(FrameworkConfiguration.path)
        optimal_path = get_optimal_path(FrameworkConfiguration.path)

        # Initialize filenames to be generated
        output_dir = 'output'
        q_params_dir = 'output_Q_parameters'
        log_filename, log_date_filename = self.initialize_log_files(output_dir, 'log')
        output_Q_filename, output_parameters_filename, output_E_filename = self.initialize_output_q_params_files(
            output_dir, q_params_dir)
        output_filename, partial_output_filename = self.initialize_output_csv_files(output_dir, 'output_csv')

        self.write_date_id_to_log(log_date_filename)
        self.write_params_to_output_file(output_parameters_filename, optimal_policy, optimal_path)

        if self.show_graphs:
            logging.debug("States are " + str(len(states)))
            logging.debug("Actions are " + str(self.num_actions_to_use))

        # Initializing the Q-matrix
        # to 0 values
        # Q = np.zeros((len(states), self.num_actions_to_use))
        # or to random values from 0 to 1
        Q = np.random.rand(len(states), self.num_actions_to_use)

        if self.use_old_matrix:
            # Retrieve from output_Q_data.csv an old matrix for "transfer learning"
            Q = self.retrieve_old_q_matrix(output_dir, q_params_dir, len(states), self.num_actions_to_use, Q)
        # if FrameworkConfiguration.DEBUG:
        #     logging.debug(Q)
        E = []
        if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
            # Initializing the E-matrix
            E = np.zeros((len(states), self.num_actions_to_use))  # trace for state action pairs

            if self.use_old_matrix:
                # Retrieve from output_E_data.csv
                # Check the format of the matrix is correct
                # TODO or should I start always from an empty E matrix?
                E = self.retrieve_old_e_matrix(output_dir, q_params_dir, len(states), self.num_actions_to_use, E)
            # if FrameworkConfiguration.DEBUG:
            #     logging.debug(E)

        start_time = time.time()

        y_timesteps = []
        y_reward = []
        y_cum_reward = []

        cumulative_reward = 0
        count_actions = 0

        # Write into output_filename the header: Episodes, Reward, CumReward, Timesteps
        self.write_headers_to_output_files(output_filename, partial_output_filename)

        # STARTING THE LEARNING PROCESS
        # LOOP OVER EPISODES
        for episode in range(self.total_episodes):
            logging.info("----------------------------------------------------")
            logging.info("Episode " + str(episode))
            sleep(3)
            t = 0
            count_actions += self.set_initial_state()
            sleep(self.seconds_to_wait)
            state1, old_props_values = compute_next_state_from_props(FrameworkConfiguration.path, 0, [], self.discovery_report)
            if FrameworkConfiguration.DEBUG:
                logging.debug("\tSTARTING FROM STATE " + str(states[state1]))
            action1 = self.choose_action(state1, Q)
            done = False
            reward_per_episode = 0
            # Exploration reduces every some episodes
            if (episode + 1) % self.decay_episode == 0:  # configurable parameter
                self.epsilon = self.epsilon - self.decay_value * self.epsilon  # could be another configurable parameter, decay of epsilon
                if self.epsilon < 0.1:
                    self.epsilon = 0.1
                    self.decay_value = 0

            # LOOP OVER TIME STEPS
            while t < self.max_steps:
                if count_actions > 55:  # To avoid crashing the lamp (rate of 60 commands/minute)
                    sleep(60)
                    count_actions = 0
                # Getting the next state
                if self.algorithm == 'qlearning':
                    action1 = self.choose_action(state1, Q)

                # Perform an action on the bulb sending a command
                json_string = build_command(method_chosen_index=action1, select_all_props=False, protocol=self.discovery_report['protocol'])
                if FrameworkConfiguration.DEBUG:
                    logging.debug("\t\tREQUEST: " + str(json_string))
                reward_from_response = operate_on_bulb_json(json_string, self.discovery_report, self.discovery_report['protocol'])
                count_actions += 1
                sleep(self.seconds_to_wait)

                state2, new_props_values = compute_next_state_from_props(FrameworkConfiguration.path, state1, old_props_values,
                                                                         self.discovery_report)
                if FrameworkConfiguration.DEBUG:
                    logging.debug("\tFROM STATE " + states[state1] + " TO STATE " + states[state2])

                reward_from_states, self.storage_reward = compute_reward_from_states(FrameworkConfiguration.path, state1, state2,
                                                                                     self.storage_reward)
                tmp_reward = -1 + reward_from_response + reward_from_states  # -1 for using a command more
                if FrameworkConfiguration.use_colored_output:
                    LOG = logging.getLogger()
                    if tmp_reward >= 0:
                        LOG.debug("\t\tREWARD: " + str(tmp_reward))
                    else:
                        LOG.error("\t\tREWARD: " + str(tmp_reward))
                    sleep(0.1)
                else:
                    logging.info("\t\tREWARD: " + str(tmp_reward))

                if state2 == 5 or (state2 == 4 and FrameworkConfiguration.path == 4):
                    done = True

                if self.algorithm == 'sarsa_lambda':
                    # Choosing the next action
                    action2 = self.choose_action(state2, Q)

                    # Learning the Q-value
                    self.update_sarsa_lambda(state1, state2, tmp_reward, action1, action2, len(states),
                                             self.num_actions_to_use, Q, E)

                elif self.algorithm == 'qlearning_lambda':
                    # Choosing the next action
                    action2 = self.choose_action(state2, Q)
                    # Learning the Q-value
                    self.update_qlearning_lambda(state1, state2, tmp_reward, action1, action2, len(states),
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

                # Update log file
                self.write_log_file(log_filename, t, tmp_reward, state1, state2, action1, action2)

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

            self.write_episode_summary(log_filename, output_filename, episode, reward_per_episode, cumulative_reward, t)

            if FrameworkConfiguration.use_colored_output:
                LOG = logging.getLogger()
                if reward_per_episode >= 0:
                    LOG.debug("\tREWARD OF THE EPISODE: " + str(reward_per_episode))
                else:
                    LOG.error("\tREWARD OF THE EPISODE: " + str(reward_per_episode))
                sleep(0.1)
            else:
                logging.info("\tREWARD OF THE EPISODE: " + str(reward_per_episode))

            if self.follow_partial_policy:
                if (episode + 1) % self.follow_policy_every_tot_episodes == 0:
                    # Follow best policy found after some episodes
                    logging.info("- - - - - - - - - - - - - - - - - - - - - -")
                    logging.info("\tFOLLOW PARTIAL POLICY AT EPISODE " + str(episode))
                    if count_actions > 35:  # To avoid crashing lamp
                        sleep(60)
                        count_actions = 0

                    # Save Q-matrix
                    self.save_matrix(output_Q_filename, states, Q, 'Q')

                    # Save E-matrix
                    # Only for sarsa(lambda) and Q(lambda)
                    if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
                        self.save_matrix(output_E_filename, states, E, 'E')

                    found_policy, dict_results = RunOutputQParameters(
                        date_to_retrieve=self.current_date.strftime(self.id_for_output), show_retrieved_info=False,
                        discovery_report=self.discovery_report).run()
                    count_actions += 20

                    with open(partial_output_filename, mode="a") as partial_output_file:
                        output_writer = csv.writer(partial_output_file, delimiter=',', quotechar='"',
                                                   quoting=csv.QUOTE_MINIMAL)
                        output_writer.writerow(
                            [episode, dict_results['timesteps_from_run'], dict_results['reward_from_run'],
                             dict_results['time_from_run'], dict_results['policy_from_run'],
                             dict_results['states_from_run']])  # Episode or episode+1?

                    if found_policy:
                        # I could stop here if found good policy, could continue if you think you could find a better one
                        pass

        # SAVE DATA
        # Print and save the Q-matrix inside external file
        if FrameworkConfiguration.DEBUG:
            logging.debug("Q MATRIX:")
            logging.debug(Q)
        self.save_matrix(output_Q_filename, states, Q, 'Q')

        # Only for sarsa(lambda) and Q(lambda)
        if self.algorithm == 'sarsa_lambda' or self.algorithm == 'qlearning_lambda':
            # Print and save the E-matrix inside external file
            if FrameworkConfiguration.DEBUG:
                logging.debug("E matrix")
                logging.debug(E)
            self.save_matrix(output_E_filename, states, E, 'E')

        # Write total time for learning algorithm
        with open(log_filename, "a") as write_file:
            write_file.write("\nTotal time of %s seconds." % (time.time() - start_time))

        sleep(5)  # Wait for writing to files
        # PLOT DATA
        if self.show_graphs:
            PlotOutputData(date_to_retrieve=self.current_date.strftime(self.id_for_output), separate_plots=False).run()

        # FOLLOW BEST POLICY FOUND
        if self.follow_policy:
            RunOutputQParameters(date_to_retrieve=self.current_date.strftime(self.id_for_output),
                                 discovery_report=self.discovery_report).run()


def main(discovery_report=None):
    format_console_output()
    # if FrameworkConfiguration.DEBUG:
    #     logging.debug(str(FrameworkConfiguration().as_dict()))

    if discovery_report is None:
        logging.error("No discovery report found.")
        logging.error("Please run this framework from the main script.")
        exit(-1)
    elif discovery_report['ip']:
        if FrameworkConfiguration.DEBUG:
            logging.debug("Received discovery report:")
            logging.debug(str(discovery_report))
        logging.info("Discovery report found at " + discovery_report['ip'])
        logging.info("Waiting...")
        sleep(5)
        for i in range(4):
            logging.info("INDEX " + str(i))
            logging.info("####### Starting RL algorithm path " + str(FrameworkConfiguration.path) + " #######")
            logging.info("ALGORITHM " + FrameworkConfiguration.algorithm
                         + " - PATH " + str(FrameworkConfiguration.path)
                         + " - EPS " + str(FrameworkConfiguration.epsilon)
                         + " - ALP " + str(FrameworkConfiguration.alpha)
                         + " - GAM " + str(FrameworkConfiguration.gamma))
            ReinforcementLearningAlgorithm(discovery_report=discovery_report, thread_id=threading.get_ident()).run()
            logging.info("####### Finish RL algorithm #######")
            sleep(50)


if __name__ == '__main__':
    main()
