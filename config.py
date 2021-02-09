"""
    Class to configure the framework and parameters for RL algorithms
"""


class FrameworkConfiguration(object):
    # should be unique and unmodifiable for and by all threads

    # General
    max_threads = 3
    current_command_id = 1
    timeout = 5
    path = 2
    directory = "../"

    # RL params
    algorithm = 'qlearning'  # 'sarsa' 'sarsa_lambda' 'qlearning' 'qlearning_lambda'
    epsilon = 0.6
    total_episodes = 5
    max_steps = 100
    alpha = 0.05
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
    use_colored_output = False  # Using colored output may delay the write operation on console
    # Put True to use_colored_output flag ONLY IF DEBUG=False

    # General info
    seconds_to_wait = 4.0
    num_actions_to_use = 37

    DEBUG = False

    def as_dict(self):
        return {'max_threads': self.max_threads,
                'current_command_id': self.current_command_id,
                'timeout': self.timeout,
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
                'use_colored_output': self.use_colored_output,
                'seconds_to_wait': self.seconds_to_wait,
                'num_actions_to_use': self.num_actions_to_use,
                'DEBUG': self.DEBUG}
