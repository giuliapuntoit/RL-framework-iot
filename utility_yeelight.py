import errno
import json
import sys
from time import sleep
import socket
import re

from learning_yeelight import RUNNING, listen_socket, MCAST_GRP, detected_bulbs, bulb_idx2ip, current_command_id
from learning_yeelight import scan_socket


# The following are utility functions:

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
    if match is not None:
        value = match.group(1)
        return value


def get_support_value(data):
    # match line of 'support = value'
    support_re = re.compile("support" + ":\s*([ -~]*)")  # match all printable characters
    match = support_re.search(data.decode())
    value = ""
    if match is not None:
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
    reward_from_response = 0
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print("Json received is ")
    # print(json_received)
    if 'id' in json_received and json_received['id'] == current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("Result is", json_received['result'])
            reward_from_response = 0
        elif 'error' in json_received and json_received['error'] is not None:
            print("Error is", json_received['error'])
            reward_from_response = -100
        else:
            print("No result or error found in answer.")
            reward_from_response = -1000  # non è colpa di nessuno?
            # TODO verificare quando arrivare qui e riprovare a simluare per tornare di nuovo al crash lampadina
    else:
        print("Bad format response.")
        reward_from_response = -1000  # non è colpa di nessuno?
        # TODO verificare quando arrivare qui e riprovare a simluare per tornare di nuovo al crash lampadina
    return reward_from_response


def handle_response_no_reward(data):
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    print("Json received is ")
    print(json_received)
    if 'id' in json_received and json_received['id'] == current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("Result is", json_received['result'])
        elif 'error' in json_received and json_received['error'] is not None:
            print("Error is", json_received['error'])
        else:
            print("No result or error found in answer.")
    else:
        print("Bad format response.")


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
        msg = "{\"id\":" + str(current_command_id) + ",\"method\":\""
        msg += method + "\",\"params\":[" + params + "]}\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        handle_response_no_reward(data)  # I do not want to compute reward when I manually turn off the lamp
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
        reward_from_response = handle_response(data)
        tcp_socket.close()
        return reward_from_response
    except Exception as e:
        print("Unexpected error:", e)
        return -1000


def compute_next_state(json_command, states, current_state):
    next_state = states.index("invalid")
    if json_command["method"] == "set_power" and json_command["params"] and json_command["params"][
        0] == "on" and current_state == 0:
        next_state = states.index("on")
    elif json_command["method"] == "set_power" and json_command["params"] and json_command["params"][
        0] == "on" and current_state == 1:
        next_state = states.index("on")
    elif json_command["method"] == "set_rgb" and current_state == 1:
        next_state = states.index("rgb")
    elif json_command["method"] == "set_rgb" and current_state == 2:
        next_state = states.index("rgb")
    elif json_command["method"] == "set_bright" and current_state == 2:
        next_state = states.index("brightness")
    elif json_command["method"] == "set_bright" and current_state == 3:
        next_state = states.index("brightness")
    elif json_command["method"] == "set_power" and json_command["params"][
        0] == "off" and current_state == 3:  # whatever state
        next_state = states.index("off_end")
    elif json_command["method"] == "set_power" and json_command["params"][0] == "off":  # whatever state
        next_state = states.index("off_start")
    else:
        # next_state = states.index("invalid") # fingiamo che se sbagli comando rimango sempre nello stesso stato
        next_state = current_state

    return next_state


def compute_reward_from_props(current_state, next_state):
    # This assumes that the path goes from state 0 to state 4 (1 2 3 4)
    reward_from_props = 0
    # Reward from passing through other states:
    if current_state == 0 and next_state == 1:
        reward_from_props = 1
    elif current_state == 1 and next_state == 2:
        reward_from_props = 2
    elif current_state == 2 and next_state == 3:
        reward_from_props = 4
    elif current_state == 3 and next_state == 4:
        reward_from_props = 1000
    return reward_from_props
