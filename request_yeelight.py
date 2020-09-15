#!/usr/bin/python
import json
import random
import socket
import sys
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
    current_command_id += 1
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
    print(supported_methods)
    # use two dictionaries to store index->ip and ip->bulb map
    detected_bulbs[host_ip] = [bulb_id, model, power, bright, rgb, host_port, supported_methods]
    bulb_idx2ip[bulb_id] = host_ip


def handle_response(data):
    # This reward should be higher if you are following the desired path. How to enforce it?
    global tot_reward
    # Print response
    json_received = json.loads(data.decode().replace("\r", "").replace("\n", ""))
    print(json_received)
    if json_received['id'] == current_command_id:
        if json_received['result'] is not None:
            print("Result is", json_received['result'])
            tot_reward += 1
        elif json_received['error'] is not None:
            print("Error is", json_received['error'])
            tot_reward -= 10
        else:
            print("No result or error found in answer.")
            tot_reward -= 100  # non è colpa di nessuno?
    else:
        print("Bad format response.")
        tot_reward -= 50  # non è colpa di nessuno?


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


def operate_on_bulb_json(json_string):
    '''
  Operate on bulb; no guarantee of success.
  Input data 'params' must be a compiled into one string.
  E.g. params="1"; params="\"smooth\"", params="1,\"smooth\",80"
  '''
    if json_string["id"] not in bulb_idx2ip:
        print("error: invalid bulb idx")
        return

    bulb_ip = bulb_idx2ip[json_string["id"]]
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

    # Chiami dict_yeelight (probabilmente il comando specifico verra' preso in base ad una matrice)
    # Il dict torna un json tramite serve_yeelight
    # TODO Ci sara' un po' di exploration/exploitation per decidere se usare conformazioni di parametri nuovi
    # o parametri gia' usati (o random? o default?)

    # Choose method
    print("Doing a random method")
    json_command = ServeYeelight(idLamp=idLamp).run()
    operate_on_bulb_json(json_command)

    # Provo a crashare la lampadina eseguendo il compando operate_on_bulb_json con json_command in loop? Potrei provare
    # Se questo metodo operate_on_bulb_json funziona qui dovrei mettere il codice per l'algoritmo di reinforcement learning
    # Potrei veramente fare un rl stupido che impara ad accenderla, e spegnerla, INTANTO

    sleep(2)

    print("Waiting 5 seconds before using default actions")
    sleep(15)

    # Setting power on
    print("Setting power on")
    operate_on_bulb(idLamp, "set_power", str("\"on\", \"sudden\", 500"))
    sleep(2)

    # Set brightness
    print("Changing brightness")
    brightness = random.randint(1, 100)
    operate_on_bulb(idLamp, "set_bright", str(brightness))

    sleep(2)

    # Set rgb
    print("Changing color rgb")
    rgb = random.randint(1, 16777215)
    operate_on_bulb(idLamp, "set_rgb", str(str(rgb) + ", \"smooth\", 500"))

    sleep(2)

    # Toggle
    print("Toggling lamp")
    operate_on_bulb(idLamp, "toggle", "")

# goal achieved, tell detection thread to quit and wait
RUNNING = False
detection_thread.join()
# done

print("Total reward received", tot_reward)
