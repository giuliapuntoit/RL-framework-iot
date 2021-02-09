


# Socket setup
FrameworkConfiguration.scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fcntl.fcntl(FrameworkConfiguration.scan_socket, fcntl.F_SETFL, os.O_NONBLOCK)
FrameworkConfiguration.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
FrameworkConfiguration.listen_socket.bind(("", 1982))
fcntl.fcntl(FrameworkConfiguration.listen_socket, fcntl.F_SETFL, os.O_NONBLOCK)
# GlobalVar.scan_socket.settimeout(GlobalVar.timeout)  # set 2 seconds of timeout -> could be a configurable parameter
# GlobalVar.listen_socket.settimeout(GlobalVar.timeout)
mreq = struct.pack("4sl", socket.inet_aton(FrameworkConfiguration.MCAST_GRP), socket.INADDR_ANY)
FrameworkConfiguration.listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

# Give socket some time to set up
sleep(2)

# First discover the lamp and connect to the lamp, with the bulb detection thread
detection_thread = Thread(target=bulbs_detection_loop)
detection_thread.start()
# Give detection thread some time to collect bulb info
sleep(10)
max_wait = 0
while len(FrameworkConfiguration.bulb_idx2ip) == 0 and max_wait < 10:
    # Wait for 10 seconds to see if some bulb is present
    # The number of seconds could be extended if necessary
    sleep(1)
    max_wait += 1
if len(FrameworkConfiguration.bulb_idx2ip) == 0:
    print("Bulb list is empty.")
else:
    # If some bulb was found, take first bulb or the one specified as argument
    display_bulbs()
    print(FrameworkConfiguration.bulb_idx2ip)
    global idLamp
    if discovery_report is None:
        idLamp = list(FrameworkConfiguration.bulb_idx2ip.keys())[0]
        print("No discovery report: id lamp", idLamp)
    elif discovery_report['ip'] and discovery_report['ip'] in FrameworkConfiguration.bulb_idx2ip.values():
        idLamp = list(FrameworkConfiguration.bulb_idx2ip.keys())[
            list(FrameworkConfiguration.bulb_idx2ip.values()).index(discovery_report['ip'])]
        print("Discovery report found: id lamp", idLamp)
    print("Waiting 5 seconds before using RL algorithm")
    sleep(5)

    # Stop bulb detection loop
    FrameworkConfiguration.RUNNING = False  # TODO should remove this

    print("\n############# Starting RL algorithm path", FrameworkConfiguration.path, "#############")
    print("ALGORITHM", FrameworkConfiguration.algorithm, "- PATH", FrameworkConfiguration.path, " - EPS ALP GAM",
          FrameworkConfiguration.epsilon, FrameworkConfiguration.alpha, FrameworkConfiguration.gamma)
    ReinforcementLearningAlgorithm().run()  # 'sarsa' 'sarsa_lambda' 'qlearning' 'qlearning_lambda'
    print("############# Finish RL algorithm #############")

# Goal achieved, tell detection thread to quit and wait
RUNNING = False  # non credo serva di nuovo, sarebbe global var comunque
detection_thread.join()
# Done
