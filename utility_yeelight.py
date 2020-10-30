import errno
import json
import sys
from time import sleep
import socket
import re

from config import GlobalVar

# The following are utility functions:
from serve_yeelight import ServeYeelight


def send_search_broadcast():
    # multicast search request to all hosts in LAN, do not wait for response
    print("send_search_broadcast running")
    multicase_address = (GlobalVar.MCAST_GRP, 1982)
    msg = "M-SEARCH * HTTP/1.1\r\n"
    msg = msg + "HOST: 239.255.255.250:1982\r\n"
    msg = msg + "MAN: \"ssdp:discover\"\r\n"
    msg = msg + "ST: wifi_bulb"
    GlobalVar.scan_socket.sendto(msg.encode(), multicase_address)


def bulbs_detection_loop():
    # a standalone thread broadcasting search request and listening on all responses
    print("bulbs_detection_loop running")
    search_interval = 30000
    read_interval = 100
    time_elapsed = 0

    while GlobalVar.RUNNING:
        if time_elapsed % search_interval == 0:
            send_search_broadcast()

        # scanner
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

        # passive listener
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
    # use two dictionaries to store index->ip and ip->bulb map
    GlobalVar.detected_bulbs[host_ip] = [bulb_id, model, power, bright, rgb, host_port, supported_methods]
    GlobalVar.bulb_idx2ip[bulb_id] = host_ip


def handle_response(data):
    # This reward should be higher if you are following the desired path. How to enforce it?
    reward_from_response = 0
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print("Json received is ")
    # print(json_received)
    if 'id' in json_received and json_received['id'] == GlobalVar.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("\t\t\tRESPONSE: result ->", json_received['result'])
            reward_from_response = 0
        elif 'error' in json_received and json_received['error'] is not None:
            print("\t\t\tRESPONSE: error ->", json_received['error'])
            reward_from_response = -100
        else:
            print("\t\t\tRESPONSE: No \'result\' or \'error\' found in answer")
            reward_from_response = -1000  # non è colpa di nessuno?
            # TODO verificare quando arrivare qui e riprovare a simluare per tornare di nuovo al crash lampadina
    else:
        print("\t\t\tRESPONSE: Bad format response")
        reward_from_response = -1000  # non è colpa di nessuno?
        # TODO verificare quando arrivare qui e riprovare a simluare per tornare di nuovo al crash lampadina
    return reward_from_response


def handle_response_no_reward(data):
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print("Json received is ")
    # print(json_received)
    if 'id' in json_received and json_received['id'] == GlobalVar.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("\t\t\tRESPONSE: result ->", json_received['result'])
        elif 'error' in json_received and json_received['error'] is not None:
            print("\t\t\tRESPONSE: error ->", json_received['error'])
        else:
            print("\t\t\tRESPONSE: No \'result\' or \'error\' found in answer")
    else:
        print("\t\t\tRESPONSE: Bad format response")


def handle_response_props(data):
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    # print("Json received is", + json_received)
    if 'id' in json_received and json_received['id'] == GlobalVar.current_command_id:
        if 'result' in json_received and json_received['result'] is not None:
            print("\t\t\tRESPONSE: result ->", json_received['result'])
            return json_received['result']  # in teoria qua c'è la lista di valori di tutte le props richieste
        elif 'error' in json_received and json_received['error'] is not None:
            print("\t\t\tRESPONSE: error ->", json_received['error'])
            return json_received['error']
        else:
            print("\t\t\tRESPONSE: No result or error found in answer")
    else:
        print("\t\t\tRESPONSE: Bad format response")
    return []  # se c'è un errore nella risposta o se non c'è risposta torno un array vuoto


def display_bulb(idx):
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
    print(str(len(GlobalVar.detected_bulbs)) + " managed bulbs")
    for i in range(1, len(GlobalVar.detected_bulbs) + 1):
        display_bulb(i)


def operate_on_bulb(idx, method, params):
    '''
  Operate on bulb; no guarantee of success.
  Input data 'params' must be a compiled into one string.
  E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"
  '''
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
    '''
  Operate on bulb; no guarantee of success.
  Input data 'params' must be a compiled into one string.
  E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"
  '''
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
        return -1000


# Useless method, could cancel TODO
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


def compute_next_state_from_props(id_lamp, current_state, old_props_values):
    next_state = current_state
    json_command = ServeYeelight(id_lamp=id_lamp, method_chosen_index=0, select_all_props=True).run()
    # props_names = ServeYeelight(id_lamp=id_lamp).get_all_properties()

    # print("get_prop")
    # print(json_command)
    props_values = operate_on_bulb_props(id_lamp, json_command)  # should contain an array of properties

    # from response compare properties before and after command

    if not props_values:
        print("\t\tSomething went wrong from get_prop: keeping the current state")
        return current_state, old_props_values

    sleep(1)

    power_index = 0
    bright_index = 1
    rgb_index = 2
    ct_index = 3
    name_index = 4
    hue_index = 5
    sat_index = 6

    # PATH 1
    # 0 1 2 4 5
    # if props_values[power_index] == 'off':  # prima colonna e' il power
    #     if current_state == 0 or current_state == 6:
    #         next_state = 0
    #     else:
    #         next_state = 5  # end state
    # elif props_values[power_index] == 'on':
    #     if current_state == 0:  # se precedentemente era spenta passo allo stato acceso
    #         next_state = 1
    #     else:
    #         if current_state == 1 and props_values[bright_index] != old_props_values[bright_index] and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 4
    #         elif current_state == 1 and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 2
    #         elif current_state == 3 and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 4
    #         elif current_state == 1 and props_values[bright_index] != old_props_values[bright_index]:
    #             next_state = 3
    #         elif current_state == 2 and props_values[bright_index] != old_props_values[bright_index]:
    #             next_state = 4
    # PATH 2
    # 0 1 8 10 5
    # if props_values[power_index] == 'off':  # prima colonna e' il power
    #     if old_props_values and props_values[name_index] != old_props_values[name_index] and current_state == 0:
    #         next_state = 7
    #     elif current_state == 0 or current_state == 6:
    #         next_state = 0
    #     else:
    #         next_state = 5  # end state
    # elif props_values[power_index] == 'on':
    #     name_modified = (props_values[name_index] != old_props_values[name_index])
    #     if current_state == 0:  # se precedentemente era spenta passo allo stato acceso
    #         next_state = 1
    #     elif current_state == 7:  # name already modified
    #         next_state = 8
    #     elif current_state == 1 and name_modified:
    #         next_state = 8
    #     else:
    #         if current_state == 1 and props_values[bright_index] != old_props_values[bright_index] and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 4
    #         elif current_state == 1 and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 2
    #         elif current_state == 1 and props_values[bright_index] != old_props_values[bright_index]:
    #             next_state = 3
    #         # case when name modified already
    #         if current_state == 8 and props_values[bright_index] != old_props_values[bright_index] and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 11
    #         elif current_state == 8 and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 9
    #         elif current_state == 8 and props_values[bright_index] != old_props_values[bright_index]:
    #             next_state = 10
    #         elif current_state == 3 and name_modified:
    #             next_state = 10
    #         elif current_state == 3 and props_values[rgb_index] != old_props_values[rgb_index]:
    #             next_state = 4
    #         elif current_state == 2 and name_modified:
    #             next_state = 9
    #         elif current_state == 2 and props_values[bright_index] != old_props_values[bright_index]:
    #             next_state = 4
    #         elif current_state == 4 and name_modified:
    #             next_state = 11

    # PATH 3
    # 0 1 12 13 16 17 18 5
    if props_values[power_index] == 'off':  # prima colonna e' il power
        if current_state == 0 or current_state == 6:
            next_state = 0
        elif current_state in [1, 2, 3, 4, 12, 13, 14, 15]:
            next_state = 16
        else:
            next_state = 5  # end state
    elif props_values[power_index] == 'on':
        bright_modified = (props_values[bright_index] != old_props_values[bright_index])
        rgb_modified = (props_values[rgb_index] != old_props_values[rgb_index])
        ct_modified = (props_values[ct_index] != old_props_values[ct_index])
        if current_state == 0:  # se precedentemente era spenta passo allo stato acceso
            next_state = 1
        elif current_state == 16:
            next_state = 17
        else:
            if current_state == 1 and bright_modified and rgb_modified and ct_modified:
                next_state = 15
            elif current_state == 1 and bright_modified and rgb_modified:
                next_state = 4
            elif current_state == 1 and rgb_modified and ct_modified:
                next_state = 14
            elif current_state == 1 and bright_modified and ct_modified:
                next_state = 13
            elif current_state == 1 and rgb_modified:
                next_state = 2
            elif current_state == 1 and bright_modified:
                next_state = 3
            elif current_state == 1 and ct_modified:
                next_state = 12
            elif current_state == 3 and rgb_modified and ct_modified:
                next_state = 15
            elif current_state == 3 and rgb_modified:
                next_state = 4
            elif current_state == 3 and ct_modified:
                next_state = 13
            elif current_state == 2 and bright_modified and ct_modified:
                next_state = 15
            elif current_state == 2 and bright_modified:
                next_state = 4
            elif current_state == 2 and ct_modified:
                next_state = 14
            elif current_state == 12 and rgb_modified and bright_modified:
                next_state = 15
            elif current_state == 12 and rgb_modified:
                next_state = 14
            elif current_state == 12 and bright_modified:
                next_state = 13
            elif current_state == 13 and rgb_modified:
                next_state = 15
            elif current_state == 17 and (rgb_modified or bright_modified):
                next_state = 19
            elif current_state == 17 and ct_modified:
                next_state = 18

    return next_state, props_values


def compute_reward_from_states(current_state, next_state):
    # This assumes that the path goes 0 1 2 4 5
    reward_from_props = 0

    # PATH 1
    # Reward from passing through other states:
    # if current_state == 0 and next_state == 1:
    #     reward_from_props = 1  # per on
    # elif current_state == 1 and next_state == 3:
    #     reward_from_props = 2  # per bright
    # elif current_state == 1 and next_state == 2:
    #     reward_from_props = 3  # per rgb
    # elif current_state == 3 and next_state == 4:
    #     reward_from_props = 4  # per rgb bright
    # elif current_state == 2 and next_state == 4:
    #     reward_from_props = 5  # per rgb bright
    # elif current_state == 4 and next_state == 5:  # Just the last step
    #     reward_from_props = 2000

    # PATH 2
    # if current_state == 0 and next_state == 1:
    #     reward_from_props = 1
    # elif current_state == 0 and next_state == 7:
    #     reward_from_props = 2
    # elif current_state == 7 and next_state == 8:
    #     reward_from_props = 2
    # elif current_state == 1 and next_state == 8:
    #     reward_from_props = 10
    # elif current_state == 8 and next_state == 10:
    #     reward_from_props = 10
    # elif current_state == 11 and next_state == 5:
    #     reward_from_props = 15
    # elif current_state == 10 and next_state == 5:
    #     reward_from_props = 2000

    # PATH 3
    if current_state == 0 and next_state == 1:
        reward_from_props = 1
    elif current_state == 1 and next_state == 3:
        reward_from_props = 2
    elif current_state == 1 and next_state == 12:
        reward_from_props = 3
    elif current_state == 3 and next_state == 13:
        reward_from_props = 10
    elif current_state == 12 and next_state == 13:
        reward_from_props = 15
    elif current_state == 16 and next_state == 17:
        reward_from_props = 2
    elif current_state == 17 and next_state == 18:
        reward_from_props = 3
    elif current_state == 18 and next_state == 5:
        reward_from_props = 2500
    return reward_from_props
