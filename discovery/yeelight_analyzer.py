import fcntl
import os
import socket
import struct
from threading import Thread
from time import sleep
import time
import sys
import re
import errno

from discovery.discovery_report import DiscoveryReport

RUNNING = True
MCAST_GRP = '239.255.255.250'
detected_bulbs = {}
bulb_idx2ip = {}
scan_socket = None
listen_socket = None

devices = []


def yeelight_discovery():
    print("START SCANNING LAN FOR YEELIGHT DEVICES")
    print("This operation may take a while...")
    global scan_socket, listen_socket
    # Socket setup
    scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fcntl.fcntl(scan_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_socket.bind(("", 1982))
    fcntl.fcntl(listen_socket, fcntl.F_SETFL, os.O_NONBLOCK)
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # Give socket some time to set up
    sleep(2)

    # First discover the lamp and connect to the lamp, with the bulb detection thread
    detection_thread = Thread(target=bulbs_detection_loop)
    detection_thread.start()
    # Give detection thread some time to collect bulb info
    sleep(10)
    max_wait = 0
    while len(bulb_idx2ip) == 0 and max_wait < 10:
        # Wait for 10 seconds to see if some bulb is present
        # The number of seconds could be extended if necessary
        sleep(1)
        max_wait += 1

    if len(bulb_idx2ip) == 0:
        print("Bulb list is empty.")
    else:
        sleep(15)
        global devices
        devices = display_bulbs()

    # Stop bulb detection loop
    global RUNNING
    RUNNING = False
    detection_thread.join()
    print("FINISH SCANNING FOR YEELIGHT DEVICES")
    return


def display_bulb(idx):
    """
    Display a bulb found in the network and returns a Discovery Report object
    """
    if idx not in bulb_idx2ip:
        print("error: invalid bulb idx")
        return None
    bulb_ip = bulb_idx2ip[idx]
    model = detected_bulbs[bulb_ip][1]
    power = detected_bulbs[bulb_ip][2]
    bright = detected_bulbs[bulb_ip][3]
    rgb = detected_bulbs[bulb_ip][4]
    host_port = detected_bulbs[bulb_ip][5]
    print(str(idx) + ": ip=" \
          + bulb_ip + ",model=" + model \
          + ",power=" + power + ",bright=" \
          + bright + ",rgb=" + rgb)
    return DiscoveryReport(result="Yeelight discovery", protocol="yeelight",
                           timestamp=time.time(), ip=bulb_ip, port=host_port)


def display_bulbs():
    """
    Display all bulbs found in the network and returns a list of lamps
    """
    lamps = []
    print(str(len(detected_bulbs)) + " managed bulbs")
    for i in range(1, len(detected_bulbs) + 1):
        lamps.append(display_bulb(i))
    return lamps


def send_search_broadcast():
    """
    Multicast search request to all hosts in LAN, do not wait for response
    """
    print("send_search_broadcast running")
    multicase_address = (MCAST_GRP, 1982)
    msg = "M-SEARCH * HTTP/1.1\r\n"
    msg = msg + "HOST: 239.255.255.250:1982\r\n"
    msg = msg + "MAN: \"ssdp:discover\"\r\n"
    msg = msg + "ST: wifi_bulb"
    scan_socket.sendto(msg.encode(), multicase_address)


def bulbs_detection_loop():
    """
    A standalone thread broadcasting search request and listening on all responses
    """
    print("bulbs_detection_loop running")
    search_interval = 30000
    read_interval = 100
    time_elapsed = 0

    while RUNNING:
        if time_elapsed % search_interval == 0:
            send_search_broadcast()

        # Scanner
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

        # Passive listener
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
    sys.exit()


def get_param_value(data, param):
    """
    Match line of 'param = value'
    """
    param_re = re.compile(param + ":\s*([ -~]*)")  # Match all printable characters
    match = param_re.search(data.decode())
    if match is not None:
        value = match.group(1)
        return value


def get_support_value(data):
    """
    Match line of 'support = value'
    """
    support_re = re.compile("support" + ":\s*([ -~]*)")  # Match all printable characters
    match = support_re.search(data.decode())
    if match is not None:
        value = match.group(1)
        return value


def handle_search_response(data):
    """
    Parse search response and extract all interested data
    If new bulb is found, insert it into dictionary of managed bulbs
    """
    location_re = re.compile("Location.*yeelight[^0-9]*([0-9]{1,3}(\.[0-9]{1,3}){3}):([0-9]*)")
    match = location_re.search(data.decode())
    if match is None:
        # logging.error("invalid data received: " + data.decode())
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
    # Use two dictionaries to store index->ip and ip->bulb map

    detected_bulbs[host_ip] = [bulb_id, model, power, bright, rgb, host_port, supported_methods]
    bulb_idx2ip[bulb_id] = host_ip


def main():
    yeelight_discovery()
    global devices
    return devices


if __name__ == '__main__':
    main()
