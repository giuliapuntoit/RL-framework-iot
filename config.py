"""
    Class to configure the framework and parameters for RL algorithms
"""


class FrameworkConfiguration(object):
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

    def as_dict(self):
        return {'RUNNING': self.RUNNING,
                'MCAST_GRP': self.MCAST_GRP,
                'detected_bulbs': self.detected_bulbs,
                'bulb_idx2ip': self.bulb_idx2ip,
                'current_command_id': self.current_command_id,
                'id_lamp': self.id_lamp,
                'listen_socket': self.listen_socket,
                'scan_socket': self.scan_socket,
                'timeout': self.timeout,
                'reward': self.reward,
                'path': self.path,
                'directory': self.directory,
                'algorithm': self.algorithm,
                'epsilon': self.epsilon,
                'total_episodes': self.total_episodes,
                'max_steps': self.max_steps,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'lam': self.lam,
                'decay_episode': self.decay_episode,
                'decay_value': self.decay_value,
                'show_graphs': self.show_graphs,
                'follow_policy': self.follow_policy,
                'use_old_matrix': self.use_old_matrix,
                'date_old_matrix': self.date_old_matrix,
                'follow_partial_policy': self.follow_partial_policy,
                'follow_policy_every_tot_episodes': self.follow_policy_every_tot_episodes,
                'seconds_to_wait': self.seconds_to_wait,
                'num_actions_to_use': self.num_actions_to_use,
                'DEBUG': self.DEBUG}
