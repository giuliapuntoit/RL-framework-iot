"""
    This script contains API functions for sending commands and receiving responses from a Yeelight device
"""

import json
from time import sleep
import socket
import logging

from config import FrameworkConfiguration
from formatter_for_output import format_console_output

logger = logging.getLogger(__name__)


def handle_response(data):
    """
    Handle the response given by the bulb, assigning some reward based on the correct response
    """
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    if 'id' in json_received and json_received['id'] == FrameworkConfiguration.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: result -> " + str(json_received['result']))
            reward_from_response = 0
        elif 'error' in json_received and json_received['error'] is not None:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: error -> " + str(json_received['error']))
            reward_from_response = -10
            if 'message' in json_received['error'] and json_received['error']['message'] == 'client quota exceeded':
                sleep(60)
        else:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: No \'result\' or \'error\' found in answer")
            reward_from_response = -20
    else:
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\t\tRESPONSE: Bad format response")
        reward_from_response = -20
    return reward_from_response


def handle_response_no_reward(data):
    """
    Handle the response given by the bulb, when returning a reward is not required
    """
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    if 'id' in json_received and json_received['id'] == FrameworkConfiguration.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: result -> " + str(json_received['result']))
            pass
        elif 'error' in json_received and json_received['error'] is not None:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: error -> " + str(json_received['error']))
            if 'message' in json_received['error'] and json_received['error']['message'] == 'client quota exceeded':
                sleep(60)
        else:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: No \'result\' or \'error\' found in answer")
            pass
    else:
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\t\tRESPONSE: Bad format response")
        pass


def handle_response_props(data):
    """
    Handle the response given by the bulb when asking for properties of the device
    """
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    if 'id' in json_received and json_received['id'] == FrameworkConfiguration.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: result -> " + str(json_received['result']))
            return json_received['result']  # List of values for properties
        elif 'error' in json_received and json_received['error'] is not None:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: error -> " + str(json_received['error']))
            if 'message' in json_received['error'] and json_received['error']['message'] == 'client quota exceeded':
                sleep(60)
            return json_received['error']
        else:
            if FrameworkConfiguration.DEBUG:
                logging.debug("\t\t\tRESPONSE: No result or error found in answer")
            pass
    else:
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\t\tRESPONSE: Bad format response")
        pass
    # If any error was found in the response or there is no response an empty array is returned
    return []


def operate_on_bulb(method, params, discovery_report):
    """
    Operate on bulb; no guarantee of success
    Input data 'params' must be a compiled into one string
                   E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"
    """
    bulb_ip = discovery_report['ip']
    port = discovery_report['port']
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(FrameworkConfiguration.timeout)

        # print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = "{\"id\":" + str(FrameworkConfiguration.current_command_id) + ",\"method\":\""
        msg += method + "\",\"params\":[" + params + "]}\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        handle_response_no_reward(data)  # I do not want to compute reward when I manually turn off the lamp
        tcp_socket.close()
    except Exception as e:
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\t\tUnexpected error:" + str(e))
        pass


def operate_on_bulb_props(json_string, discovery_report):
    """
    Send a request for the properties of the bulb, handling the response properly
    :return Return the property values of the current state of the bulb
    """
    if FrameworkConfiguration.DEBUG:
        logging.debug("\t\tREQUEST FOR PROPS: " + json_string)

    bulb_ip = discovery_report['ip']
    port = discovery_report['port']
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(FrameworkConfiguration.timeout)
        # print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = str(json_string) + "\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        props = handle_response_props(data)
        tcp_socket.close()
        return props
    except Exception as e:
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\t\tUnexpected error:" + str(e))
        return []


def operate_on_bulb_json(json_string, discovery_report):
    """
    Operate on bulb; no guarantee of success
    Send a command already formatted inside a json
    Input json_string: command already formatted inside a json
    :return: Return the reward returned by the response given to this sent command
    """
    bulb_ip = discovery_report['ip']
    port = discovery_report['port']
    try:
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.settimeout(FrameworkConfiguration.timeout)
        # print("connect ", bulb_ip, port, "...")
        tcp_socket.connect((bulb_ip, int(port)))
        msg = str(json_string) + "\r\n"
        tcp_socket.send(msg.encode())
        data = tcp_socket.recv(2048)
        reward_from_response = handle_response(data)
        tcp_socket.close()
        return reward_from_response
    except Exception as e:
        if FrameworkConfiguration.DEBUG:
            logging.debug("\t\t\tUnexpected error:" + str(e))
        return -20
