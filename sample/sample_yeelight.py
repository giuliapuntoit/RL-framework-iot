#!/usr/bin/python

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

detected_bulbs = {}
bulb_idx2ip = {}
DEBUGGING = False
RUNNING = True
current_command_id = 0
MCAST_GRP = '239.255.255.250'

scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fcntl.fcntl(scan_socket, fcntl.F_SETFL, os.O_NONBLOCK)
listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
listen_socket.bind(("", 1982))
fcntl.fcntl(listen_socket, fcntl.F_SETFL, os.O_NONBLOCK)
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)


def debug(msg):
    if DEBUGGING:
        print(msg)


def next_cmd_id():
    global current_command_id
    current_command_id += 1
    return current_command_id


def send_search_broadcast():
    '''
  multicast search request to all hosts in LAN, do not wait for response
  '''
    multicase_address = (MCAST_GRP, 1982)
    debug("send search request")
    msg = "M-SEARCH * HTTP/1.1\r\n"
    msg = msg + "HOST: 239.255.255.250:1982\r\n"
    msg = msg + "MAN: \"ssdp:discover\"\r\n"
    msg = msg + "ST: wifi_bulb"
    scan_socket.sendto(msg.encode(), multicase_address)


def bulbs_detection_loop():
    '''
  a standalone thread broadcasting search request and listening on all responses
  '''
    debug("bulbs_detection_loop running")
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
    '''
  match line of 'param = value'
  '''
    param_re = re.compile(param + ":\s*([ -~]*)")  # match all printable characters
    match = param_re.search(data.decode())
    value = ""
    if match != None:
        value = match.group(1)
        return value


def handle_search_response(data):
    '''
  Parse search response and extract all interested data.
  If new bulb is found, insert it into dictionary of managed bulbs.
  '''
    location_re = re.compile("Location.*yeelight[^0-9]*([0-9]{1,3}(\.[0-9]{1,3}){3}):([0-9]*)")
    match = location_re.search(data.decode())
    if match == None:
        debug("invalid data received: " + data.decode())
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
    # use two dictionaries to store index->ip and ip->bulb map
    detected_bulbs[host_ip] = [bulb_id, model, power, bright, rgb, host_port]
    bulb_idx2ip[bulb_id] = host_ip


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
  Operate on bulb; no gurantee of success.
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
        tcp_socket.close()
    except Exception as e:
        print("Unexpected error:", e)


def toggle_bulb(idx):
    operate_on_bulb(idx, "toggle", "")


def set_bright(idx, bright):
    operate_on_bulb(idx, "set_bright", str(bright))


def print_cli_usage():
    print("Usage:")
    print("  q|quit: quit bulb manager")
    print("  h|help: print this message")
    print("  t|toggle <idx>: toggle bulb indicated by idx")
    print("  b|bright <idx> <bright>: set brightness of bulb with label <idx>")
    print("  r|refresh: refresh bulb list")
    print("  l|list: list all managed bulbs")


def handle_user_input():
    '''
  User interaction loop.
  '''
    while True:
        command_line = input("Enter a command: ")
        # raw_input in Python 2 is just input in Python 3 (input in Python 2 is eval(input())
        valid_cli = True
        debug("command_line=" + command_line)
        command_line.lower()  # convert all user input to lower case, i.e. cli is caseless
        argv = command_line.split()  # i.e. don't allow parameters with space characters
        if len(argv) == 0:
            continue
        if argv[0] == "q" or argv[0] == "quit":
            print("Bye!")
            return
        elif argv[0] == "l" or argv[0] == "list":
            display_bulbs()
        elif argv[0] == "r" or argv[0] == "refresh":
            detected_bulbs.clear()
            bulb_idx2ip.clear()
            send_search_broadcast()
            # sleep(0.5)
            # display_bulbs()
        elif argv[0] == "h" or argv[0] == "help":
            print_cli_usage()
            continue
        elif argv[0] == "t" or argv[0] == "toggle":
            if len(argv) != 2:
                valid_cli = False
            else:
                try:
                    i = int(float(argv[1]))
                    toggle_bulb(i)
                except:
                    valid_cli = False
        elif argv[0] == "b" or argv[0] == "bright":
            if len(argv) != 3:
                print("incorrect argc")
                valid_cli = False
            else:
                try:
                    idx = int(float(argv[1]))
                    print("idx", idx)
                    bright = int(float(argv[2]))
                    print("bright", bright)
                    set_bright(idx, bright)
                except:
                    valid_cli = False
        else:
            valid_cli = False

        if not valid_cli:
            print("error: invalid command line:", command_line)
            print_cli_usage()


# Perform some simple operations on the Yeelight bulb

# MAIN
# print welcome message first
print("Welcome to Yeelight WifiBulb Lan controller")
print_cli_usage()
# start the bulb detection thread
detection_thread = Thread(target=bulbs_detection_loop)
detection_thread.start()
# give detection thread some time to collect bulb info
sleep(0.2)
# user interaction loop
handle_user_input()
# user interaction end, tell detection thread to quit and wait
RUNNING = False
detection_thread.join()
# done
