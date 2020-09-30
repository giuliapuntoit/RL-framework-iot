class GlobalVar(object):
    RUNNING = True
    MCAST_GRP = '239.255.255.250'
    detected_bulbs = {}
    bulb_idx2ip = {}
    current_command_id = 1
    listen_socket = None
    scan_socket = None
