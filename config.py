class FrameworkConfiguration(object):  # change name, tipo configuration params ecc
    # should be unique and unmodifiable for and by all threads

    # Da spostare:
    RUNNING = True
    MCAST_GRP = '239.255.255.250'
    detected_bulbs = {}
    bulb_idx2ip = {}
    current_command_id = 1
    id_lamp = ""

    # Da togliere:
    listen_socket = None
    scan_socket = None

    # Da riorganizzare
    timeout = 5
    reward = 0
    path = 2
    directory = "../"

    # RL params
    algorithm = 'sarsa'
    epsilon = 0.6
    total_episodes = 10
    max_steps = 100
    alpha = 0.005
    gamma = 0.95
    lam = 0.9
    decay_episode = 20
    decay_value = 0.001

    # Flags
    show_graphs = False
    follow_policy = False
    use_old_matrix = False
    date_old_matrix = 'YY_mm_dd_HH_MM_SS'
    follow_partial_policy = False
    follow_policy_every_tot_episodes = 5

    # General info
    seconds_to_wait = 4.0
    num_actions_to_use = 37

    DEBUG = False
