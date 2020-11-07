import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


from config import GlobalVar


def plot_reward_per_request(date_to_retrieve='YY_mm_dd_HH_MM_SS'):

    directory = GlobalVar.directory + 'output/log/'
    log_file = directory + 'log_' + date_to_retrieve + '.log'

    print(log_file)

    # Each non empty line is a sent command
    # Command of power is substituted by episode finishing line
    # Minus last line that is the total time

    count = 0
    cum_reward = 0
    commands = []
    rewards = []
    cum_rewards = []
    with open(log_file) as f:
        for line in f:
            if len(line.strip()) != 0:  # Not empty lines
                if not line.startswith("Episode") and not line.startswith("Total"):
                    count += 1
                    commands.append(count)
                    tmp_reward = int(line.split()[5])
                    cum_reward += tmp_reward
                    rewards.append(tmp_reward)
                    cum_rewards.append(cum_reward)

    pl.plot(commands, rewards, label='reward')  # single line
    # pl.plot(commands, cum_rewards, label='qlearning_lambda')  # single line

    pl.xlabel('Number of sent commands')
    pl.ylabel('Cumulative reward')
    pl.legend(loc='upper right')
    pl.title('Reward over commands')
    pl.grid(True)
    plt.savefig('all_commands_plot.png')
    plt.show()


if __name__ == '__main__':
    sarsa = ['2020_11_05_03_27_46',
             '2020_11_05_04_07_23',
             '2020_11_05_04_48_59',
             '2020_11_05_05_30_35',
             '2020_11_05_06_10_02', ]

    sarsa_lambda = ['2020_11_05_06_47_59',
                    '2020_11_05_07_33_31',
                    '2020_11_05_08_04_47',
                    '2020_11_05_08_48_46',
                    '2020_11_05_09_35_46', ]

    qlearning = ['2020_11_05_10_24_34',
                 '2020_11_05_11_05_37',
                 '2020_11_05_11_48_23',
                 '2020_11_05_12_33_03',
                 '2020_11_05_13_16_54', ]

    qlearning_lambda = ['2020_11_05_13_54_50',
                        '2020_11_05_14_37_02',
                        '2020_11_05_15_10_00',
                        '2020_11_05_15_49_28',
                        '2020_11_05_16_27_15', ]

    for dat in qlearning_lambda:
        plot_reward_per_request(date_to_retrieve=dat)
