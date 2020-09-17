#!/usr/bin/python

import time
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import random
import socket
import sys
from datetime import datetime
import time
import fcntl
import re
import os
import errno
import struct
from threading import Thread
from time import sleep
from collections import OrderedDict

# Global variables
from serve_yeelight import ServeYeelight

detected_bulbs = {}
bulb_idx2ip = {}
RUNNING = True
current_command_id = 0
MCAST_GRP = '239.255.255.250'

# Variables for RL
tot_reward = 0

current_state = 0

# Socket setup
scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fcntl.fcntl(scan_socket, fcntl.F_SETFL, os.O_NONBLOCK)
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listen_socket.bind(("", 1982))
fcntl.fcntl(listen_socket, fcntl.F_SETFL, os.O_NONBLOCK)
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

sleep(2)


def next_cmd_id():
    global current_command_id
    current_command_id = 1
    return current_command_id


def send_search_broadcast():
    # multicast search request to all hosts in LAN, do not wait for response
    print("send_search_broadcast running")
    multicase_address = (MCAST_GRP, 1982)
    msg = "M-SEARCH * HTTP/1.1\r\n"
    msg = msg + "HOST: 239.255.255.250:1982\r\n"
    msg = msg + "MAN: \"ssdp:discover\"\r\n"
    msg = msg + "ST: wifi_bulb"
    scan_socket.sendto(msg.encode(), multicase_address)


def bulbs_detection_loop():
    # a standalone thread broadcasting search request and listening on all responses
    print("bulbs_detection_loop running")
    search_interval = 30000
    read_interval = 100
    time_elapsed = 0

    while RUNNING:
        if time_elapsed % search_interval == 0:
            send_search_broadcast()

        # scanner
        while True:
            try:
                data = scan_socket.recv(2048)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    break
                else:
                    print(e)
                    sys.exit(1)
            handle_search_response(data)

        # passive listener
        while True:
            try:
                data, addr = listen_socket.recvfrom(2048)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    break
                else:
                    print(e)
                    sys.exit(1)
            handle_search_response(data)

        time_elapsed += read_interval
        sleep(read_interval / 1000.0)
    scan_socket.close()
    listen_socket.close()


def get_param_value(data, param):
    # match line of 'param = value'
    param_re = re.compile(param + ":\s*([ -~]*)")  # match all printable characters
    match = param_re.search(data.decode())
    value = ""
    if match != None:
        value = match.group(1)
        return value


def get_support_value(data):
    # match line of 'support = value'
    support_re = re.compile("support" + ":\s*([ -~]*)")  # match all printable characters
    match = support_re.search(data.decode())
    value = ""
    if match != None:
        value = match.group(1)
        # print(value)
        return value


def handle_search_response(data):
    '''
  Parse search response and extract all interested data.
  If new bulb is found, insert it into dictionary of managed bulbs.
  '''
    location_re = re.compile("Location.*yeelight[^0-9]*([0-9]{1,3}(\.[0-9]{1,3}){3}):([0-9]*)")
    match = location_re.search(data.decode())
    if match == None:
        print("invalid data received: " + data.decode())
        return

    host_ip = match.group(1)
    if host_ip in detected_bulbs:
        bulb_id = detected_bulbs[host_ip][0]
    else:
        bulb_id = len(detected_bulbs) + 1
    host_port = match.group(3)
    model = get_param_value(data, "model")
    power = get_param_value(data, "power")
    bright = get_param_value(data, "bright")
    rgb = get_param_value(data, "rgb")
    supported_methods = get_support_value(data).split(sep=None)
    # print(supported_methods)
    # use two dictionaries to store index->ip and ip->bulb map
    detected_bulbs[host_ip] = [bulb_id, model, power, bright, rgb, host_port, supported_methods]
    bulb_idx2ip[bulb_id] = host_ip


def handle_response(data):
    # This reward should be higher if you are following the desired path. How to enforce it?
    global tot_reward
    tot_reward = 0
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    print("Json received is ")
    print(json_received)
    if 'id' in json_received and json_received['id'] == current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("Result is", json_received['result'])
            tot_reward = 0
        elif 'error' in json_received and json_received['error'] is not None:
            print("Error is", json_received['error'])
            tot_reward = -100
        else:
            print("No result or error found in answer.")
            tot_reward = -1000  # non è colpa di nessuno?
    else:
        print("Bad format response.")
        tot_reward = -1000  # non è colpa di nessuno?


def display_bulb(idx):
    if idx not in bulb_idx2ip:
        print("error: invalid bulb idx")
        return
    bulb_ip = bulb_idx2ip[idx]
    model = detected_bulbs[bulb_ip][1]
    power = detected_bulbs[bulb_ip][2]
    bright = detected_bulbs[bulb_ip][3]
    rgb = detected_bulbs[bulb_ip][4]
    print(str(idx) + ": ip=" \
          + bulb_ip + ",model=" + model \
          + ",power=" + power + ",bright=" \
          + bright + ",rgb=" + rgb)


def display_bulbs():
    print(str(len(detected_bulbs)) + " managed bulbs")
    for i in range(1, len(detected_bulbs) + 1):
        display_bulb(i)


def operate_on_bulb(idx, method, params):
    '''
  Operate on bulb; no guarantee of success.
  Input data 'params' must be a compiled into one string.
  E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"
  '''
    if idx not in bulb_idx2ip:
        print("error: invalid bulb idx")
        return

    bulb_ip = bulb_idx2ip[idx]
    port = detected_bulbs[bulb_ip][5]
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = "{\"id\":" + str(next_cmd_id()) + ",\"method\":\""
        msg += method + "\",\"params\":[" + params + "]}\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        handle_response(data)
        tcp_socket.close()
    except Exception as e:
        print("Unexpected error:", e)


def operate_on_bulb_json(id_lamp, json_string):
    '''
  Operate on bulb; no guarantee of success.
  Input data 'params' must be a compiled into one string.
  E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"
  '''
    if id_lamp not in bulb_idx2ip:
        print("error: invalid bulb idx")
        return

    bulb_ip = bulb_idx2ip[id_lamp]
    port = detected_bulbs[bulb_ip][5]
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = str(json_string) + "\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        handle_response(data)
        tcp_socket.close()
    except Exception as e:
        print("Unexpected error:", e)


class SarsaSimplified(object):

    def __init__(self, epsilon=0.4, total_episodes=10, max_steps=100, alpha=0.005, gamma=0.95, disable_graphs=False,
                 seconds_to_wait=4):
        self.epsilon = epsilon
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma
        self.disable_graphs = disable_graphs
        self.seconds_to_wait = seconds_to_wait

    # Function to choose the next action
    def choose_action(self, state, Qmatrix):
        # Here I should choose the method
        if np.random.uniform(0, 1) < self.epsilon:
            # Select the action randomly
            # action = random.randint(0, 35 - 1)
            action = random.randint(0, 3)
        else:
            # Select maximum, if multiple values select randomly
            # choose random action between the max ones
            action = np.random.choice(np.where(Qmatrix[state, :] == Qmatrix[state, :].max())[0])
        # The action then should be converted when used into a json_string returned by serve_yeelight
        # action is an index
        return action

    # Function to learn the Q-value
    def update(self, state, state2, reward, action, action2, Qmatrix):
        predict = Qmatrix[state, action]
        target = reward + self.gamma * Qmatrix[state2, action2]
        Qmatrix[state, action] = Qmatrix[state, action] + self.alpha * (target - predict)

    def run(self):

        # Mi invento questi stati: lampadina parte da accesa, poi accendo, cambio colore, spengo (?)
        states = ["off_start", "on", "rgb", "brightness", "off_end", "invalid"]  # 0 1 2 3 4 5

        log_filename = datetime.now().strftime('./log/logfile_%H_%M_%S_%d_%m_%Y.log')

        # SARSA algorithm

        # Initializing the Q-matrix
        if not self.disable_graphs:
            print("N states: ", len(states))
            # print("N actions: ", 35)  # 35 methods, bruttissimo questo parametro statico
            print("N actions: ", 4)
        # Q = np.zeros((len(states), 35))
        Q = np.zeros((len(states), 4))

        start_time = time.time()

        x = range(0, self.total_episodes)
        y_timesteps = []
        y_reward = []
        y_cum_reward = []

        cumulative_reward = 0

        global current_state

        # Starting the SARSA learning
        for episode in range(self.total_episodes):
            if not self.disable_graphs:
                print("Episode", episode)
            t = 0
            # Turn off the lamp
            print("Setting power off")
            operate_on_bulb(idLamp, "set_power", str("\"off\", \"sudden\", 0"))
            sleep(self.seconds_to_wait)

            state1 = states.index("off_start")
            current_state = state1
            action1 = self.choose_action(state1, Q)
            done = False
            reward_per_episode = 0

            while t < self.max_steps:
                # Getting the next state

                print("Doing an action")
                json_string = ServeYeelight(idLamp=idLamp, method_chosen_index=action1 + 2).run()
                json_command = json.loads(json_string)
                print("Json command is " + str(json_string))
                operate_on_bulb_json(idLamp, json_string)
                sleep(self.seconds_to_wait)

                # Il reward dovrebbe essere dato in base a handle_response
                # forse anche l'aggiornamento dello stato dovrebbe essere in handle_response

                # check current state using get_prop method 0
                if json_command["method"] == "set_power" and json_command["params"] and json_command["params"][0] == "on" and state1 == 0:
                    state2 = states.index("on")
                elif json_command["method"] == "set_power" and json_command["params"] and json_command["params"][0] == "on" and state1 == 1:
                    state2 = states.index("on")
                elif json_command["method"] == "set_rgb" and state1 == 1:
                    state2 = states.index("rgb")
                elif json_command["method"] == "set_rgb" and state1 == 2:
                    state2 = states.index("rgb")
                elif json_command["method"] == "set_bright" and state1 == 2:
                    state2 = states.index("brightness")
                elif json_command["method"] == "set_bright" and state1 == 3:
                    state2 = states.index("brightness")
                elif json_command["method"] == "set_power" and json_command["params"][0] == "off" and state1 == 3:  # whatever state
                    state2 = states.index("off_end")
                elif json_command["method"] == "set_power" and json_command["params"][0] == "off":  # whatever state
                    state2 = states.index("off_start")
                else:
                    # state2 = states.index("invalid") # fingiamo che se sbagli comando rimango sempre nello stesso stato
                    state2 = state1

                print("From state", state1, "to state", state2)
                tmp_reward = -1 + tot_reward

                if state1 == 0 and state2 == 1:
                    tmp_reward += 100
                if state1 == 1 and state2 == 2:
                    tmp_reward += 200
                if state1 == 2 and state2 == 3:
                    tmp_reward += 400
                if state1 == 3 and state2 == 4:
                    tmp_reward += 1000
                    done = True

                # Choosing the next action
                action2 = self.choose_action(state2, Q)

                # Learning the Q-value
                self.update(state1, state2, tmp_reward, action1, action2, Q)

                state1 = state2
                current_state = state1
                action1 = action2

                # Updating the respective values
                t += 1
                reward_per_episode += tmp_reward

                with open(log_filename, "a") as write_file:
                    write_file.write("\nTimestep " + str(t - 1) + " finished.")
                    write_file.write(" Temporary reward: " + str(tmp_reward))
                    write_file.write(" Current state: " + str(state1))
                # If at the end of learning process
                if done:
                    break
            cumulative_reward += reward_per_episode
            y_timesteps.append(t - 1)
            y_cum_reward.append(cumulative_reward)
            y_reward.append(reward_per_episode)
            with open(log_filename, "a") as write_file:
                write_file.write("\nEpisode " + str(episode) + " finished.\n")

        # Visualizing the Q-matrix
        if not self.disable_graphs:
            print(Q)
            print("--- %s seconds ---" % (time.time() - start_time))
            plt.plot(x, y_reward)
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.title('Reward per episodes')

            plt.show()

            plt.plot(x, y_cum_reward)
            plt.xlabel('Episodes')
            plt.ylabel('Cumulative reward')
            plt.title('Cumulative reward over episodes')

            plt.show()

            plt.plot(x, y_timesteps)
            plt.xlabel('Episodes')
            plt.ylabel('Timestep to end of the episode')
            plt.title('Timesteps per episode')

            plt.show()

        print("Setting power off")
        operate_on_bulb(idLamp, "set_power", str("\"off\", \"sudden\", 0"))
        sleep(self.seconds_to_wait)
        current_state = 0

        if not self.disable_graphs:
            print("Restarting... returning to state: off")
        t = 0
        final_policy = []
        final_reward = 0
        optimal = [0, 1, 2, 3, 4]
        while t < 10:
            state = current_state
            max_action = np.argmax(Q[state, :])
            final_policy.append(max_action)
            if not self.disable_graphs:
                print("Action to perform is", max_action)

            previous_state = current_state

            json_string = ServeYeelight(idLamp=idLamp, method_chosen_index=max_action).run()
            json_command = json.loads(json_string)
            print("Json command is " + str(json_string))
            operate_on_bulb_json(idLamp, json_string)
            sleep(self.seconds_to_wait)

            state1 = previous_state

            if json_command["method"] == "set_power" and json_command["params"] and json_command["params"][0] == "on" and state1 == 0:
                state2 = states.index("on")
            elif json_command["method"] == "set_power" and json_command["params"] and json_command["params"][0] == "on" and state1 == 1:
                state2 = states.index("on")
            elif json_command["method"] == "set_rgb" and state1 == 1:
                state2 = states.index("rgb")
            elif json_command["method"] == "set_rgb" and state1 == 2:
                state2 = states.index("rgb")
            elif json_command["method"] == "set_bright" and state1 == 2:
                state2 = states.index("brightness")
            elif json_command["method"] == "set_bright" and state1 == 3:
                state2 = states.index("brightness")
            elif json_command["method"] == "set_power" and json_command["params"][0] == "off" and state1 == 3:  # whatever state
                state2 = states.index("off_end")
            elif json_command["method"] == "set_power" and json_command["params"][0] == "off":  # whatever state
                state2 = states.index("off_start")
            else:
                # state2 = states.index("invalid") # fingiamo che se sbagli comando rimango sempre nello stesso stato
                state2 = state1

            tmp_reward = -1 + tot_reward

            if state1 == 0 and state2 == 1:
                tmp_reward += 100
            if state1 == 1 and state2 == 2:
                tmp_reward += 200
            if state1 == 2 and state2 == 3:
                tmp_reward += 400
            if state1 == 3 and state2 == 4:
                tmp_reward += 1000
                done = True

            current_state = state2

            final_reward += tmp_reward
            if not self.disable_graphs:
                print("New state", current_state)
            if previous_state == 2 and current_state == 3:
                break
            t += 1

        print("Length final policy is", len(final_policy))
        print("Final policy is", final_policy)
        if len(final_policy) == 5 and np.array_equal(final_policy, optimal):
            return True, final_reward
        else:
            return False, final_reward


if __name__ == '__main__':
    # MAIN

    # first discover the lamp and Connect to the lamp
    # start the bulb detection thread
    detection_thread = Thread(target=bulbs_detection_loop)
    detection_thread.start()
    # give detection thread some time to collect bulb info
    sleep(10)

    # show discovered lamps
    display_bulbs()

    print(bulb_idx2ip)
    max_wait = 0
    while len(bulb_idx2ip) == 0 and max_wait < 10:
        sleep(1)
        max_wait += 1
    if len(bulb_idx2ip) == 0:
        print("Bulb list is empty.")
    else:
        display_bulbs()
        idLamp = list(bulb_idx2ip.keys())[0]

        print("Waiting 5 seconds before using SARSA")
        sleep(5)

        RUNNING = False
        # Do Sarsa

        optimalPolicy, obtainedReward = SarsaSimplified(max_steps=100, total_episodes=200).run()
        if optimalPolicy:
            print("Optimal policy was found with reward", obtainedReward)
        else:
            print("No optimal policy reached with reward", obtainedReward)

    # goal achieved, tell detection thread to quit and wait
    RUNNING = False
    detection_thread.join()
    # done

    print("Total reward received", tot_reward)

# Set manually timeout for connecting!
