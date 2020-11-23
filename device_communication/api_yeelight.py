import errno
import json
import sys
from time import sleep
import socket
import re

from config import GlobalVar


# The following are API functions for sending commands and receiving responses from a Yeelight device

def send_search_broadcast():
    # Multicast search request to all hosts in LAN, do not wait for response
    print("send_search_broadcast running")
    multicase_address = (GlobalVar.MCAST_GRP, 1982)
    msg = "M-SEARCH * HTTP/1.1\r\n"
    msg = msg + "HOST: 239.255.255.250:1982\r\n"
    msg = msg + "MAN: \"ssdp:discover\"\r\n"
    msg = msg + "ST: wifi_bulb"
    GlobalVar.scan_socket.sendto(msg.encode(), multicase_address)


def bulbs_detection_loop():
    # A standalone thread broadcasting search request and listening on all responses
    print("bulbs_detection_loop running")
    search_interval = 30000
    read_interval = 100
    time_elapsed = 0

    while GlobalVar.RUNNING:
        if time_elapsed % search_interval == 0:
            send_search_broadcast()

        # Scanner
        while True:
            try:
                data = GlobalVar.scan_socket.recv(2048)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    break
                else:
                    print(e)
                    sys.exit(1)
            handle_search_response(data)

        # Passive listener
        while True:
            try:
                data, addr = GlobalVar.listen_socket.recvfrom(2048)
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
    GlobalVar.scan_socket.close()
    GlobalVar.listen_socket.close()


def get_param_value(data, param):
    # Match line of 'param = value'
    param_re = re.compile(param + ":\s*([ -~]*)")  # Match all printable characters
    match = param_re.search(data.decode())
    if match is not None:
        value = match.group(1)
        return value


def get_support_value(data):
    # Match line of 'support = value'
    support_re = re.compile("support" + ":\s*([ -~]*)")  # Match all printable characters
    match = support_re.search(data.decode())
    if match is not None:
        value = match.group(1)
        return value


def handle_search_response(data):
    # Parse search response and extract all interested data
    # If new bulb is found, insert it into dictionary of managed bulbs

    location_re = re.compile("Location.*yeelight[^0-9]*([0-9]{1,3}(\.[0-9]{1,3}){3}):([0-9]*)")
    match = location_re.search(data.decode())
    if match is None:
        print("invalid data received: " + data.decode())
        return

    host_ip = match.group(1)
    if host_ip in GlobalVar.detected_bulbs:
        bulb_id = GlobalVar.detected_bulbs[host_ip][0]
    else:
        bulb_id = len(GlobalVar.detected_bulbs) + 1
    host_port = match.group(3)
    model = get_param_value(data, "model")
    power = get_param_value(data, "power")
    bright = get_param_value(data, "bright")
    rgb = get_param_value(data, "rgb")
    supported_methods = get_support_value(data).split(sep=None)
    # print(supported_methods)
    # Use two dictionaries to store index->ip and ip->bulb map
    GlobalVar.detected_bulbs[host_ip] = [bulb_id, model, power, bright, rgb, host_port, supported_methods]
    GlobalVar.bulb_idx2ip[bulb_id] = host_ip


def handle_response(data):
    # Handle the response given by the bulb, assigning some reward based on the correct response

    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print(json_received)
    if 'id' in json_received and json_received['id'] == GlobalVar.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("\t\t\tRESPONSE: result ->", json_received['result'])
            reward_from_response = 0
        elif 'error' in json_received and json_received['error'] is not None:
            print("\t\t\tRESPONSE: error ->", json_received['error'])
            reward_from_response = -10
            if 'message' in json_received['error'] and json_received['error']['message'] == 'client quota exceeded':
                sleep(60)
        else:
            print("\t\t\tRESPONSE: No \'result\' or \'error\' found in answer")
            reward_from_response = -20  # non è colpa di nessuno?
    else:
        print("\t\t\tRESPONSE: Bad format response")
        reward_from_response = -20  # non è colpa di nessuno?
    return reward_from_response


def handle_response_no_reward(data):
    # Handle the response given by the bulb, when returning a reward is not required

    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print(json_received)

    if 'id' in json_received and json_received['id'] == GlobalVar.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("\t\t\tRESPONSE: result ->", json_received['result'])
        elif 'error' in json_received and json_received['error'] is not None:
            print("\t\t\tRESPONSE: error ->", json_received['error'])
            if 'message' in json_received['error'] and json_received['error']['message'] == 'client quota exceeded':
                sleep(60)
        else:
            print("\t\t\tRESPONSE: No \'result\' or \'error\' found in answer")
    else:
        print("\t\t\tRESPONSE: Bad format response")


def handle_response_props(data):
    # Handle the response given by the bulb when asking for properties of the device

    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print("[DEBUG] Json received", json_received)

    if 'id' in json_received and json_received['id'] == GlobalVar.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("\t\t\tRESPONSE: result ->", json_received['result'])
            return json_received['result']  # List of values for properties
        elif 'error' in json_received and json_received['error'] is not None:
            print("\t\t\tRESPONSE: error ->", json_received['error'])
            if 'message' in json_received['error'] and json_received['error']['message'] == 'client quota exceeded':
                sleep(60)
            return json_received['error']
        else:
            print("\t\t\tRESPONSE: No result or error found in answer")
    else:
        print("\t\t\tRESPONSE: Bad format response")
    # If any error was found in the response or there is no response an empty array is returned
    return []


def display_bulb(idx):
    # Display a bulb found in the network
    if idx not in GlobalVar.bulb_idx2ip:
        print("error: invalid bulb idx")
        return
    bulb_ip = GlobalVar.bulb_idx2ip[idx]
    model = GlobalVar.detected_bulbs[bulb_ip][1]
    power = GlobalVar.detected_bulbs[bulb_ip][2]
    bright = GlobalVar.detected_bulbs[bulb_ip][3]
    rgb = GlobalVar.detected_bulbs[bulb_ip][4]
    print(str(idx) + ": ip=" \
          + bulb_ip + ",model=" + model \
          + ",power=" + power + ",bright=" \
          + bright + ",rgb=" + rgb)


def display_bulbs():
    # Display all bulbs found in the network
    print(str(len(GlobalVar.detected_bulbs)) + " managed bulbs")
    for i in range(1, len(GlobalVar.detected_bulbs) + 1):
        display_bulb(i)


def operate_on_bulb(idx, method, params):
    # Operate on bulb; no guarantee of success.
    # Input data 'params' must be a compiled into one string.
    # E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"

    if idx not in GlobalVar.bulb_idx2ip:
        print("error: invalid bulb idx")
        return

    bulb_ip = GlobalVar.bulb_idx2ip[idx]
    port = GlobalVar.detected_bulbs[bulb_ip][5]
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(GlobalVar.timeout)
        # print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = "{\"id\":" + str(GlobalVar.current_command_id) + ",\"method\":\""
        msg += method + "\",\"params\":[" + params + "]}\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        handle_response_no_reward(data)  # I do not want to compute reward when I manually turn off the lamp
        tcp_socket.close()
    except Exception as e:
        print("\t\t\tUnexpected error:", e)


def operate_on_bulb_props(id_lamp, json_string):
    # Send a request for the properties of the bulb, handling the response properly
    # Return the property values of the current state of the bulb
    print("\t\tREQUEST FOR PROPS:", json_string)
    if id_lamp not in GlobalVar.bulb_idx2ip:
        print("error: invalid bulb idx")
        return []

    bulb_ip = GlobalVar.bulb_idx2ip[id_lamp]
    port = GlobalVar.detected_bulbs[bulb_ip][5]
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(GlobalVar.timeout)
        # print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = str(json_string) + "\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        props = handle_response_props(data)
        tcp_socket.close()
        return props
    except Exception as e:
        print("\t\t\tUnexpected error:", e)
        return []


def operate_on_bulb_json(id_lamp, json_string):
    # Operate on bulb; no guarantee of success.
    # Send a command already formatted inside a json
    # Return the reward returned by the response given to this sent command

    if id_lamp not in GlobalVar.bulb_idx2ip:
        print("error: invalid bulb idx")
        return

    bulb_ip = GlobalVar.bulb_idx2ip[id_lamp]
    port = GlobalVar.detected_bulbs[bulb_ip][5]
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(GlobalVar.timeout)
        # print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = str(json_string) + "\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        reward_from_response = handle_response(data)
        tcp_socket.close()
        return reward_from_response
    except Exception as e:
        print("\t\t\tUnexpected error:", e)
        return -10