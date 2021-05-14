"""
    Script to generate an automated plot in real time attaching to the execution of the current RL algorithm used
"""

import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation
from plotter.support_plotter import read_reward_timesteps_from_output_file

target_output_dir = './'


def get_reward_for_episode(curr_date="2020_12_15_00_19_48", algorithm="qlearning"):
    """
    Function to save only the reward value per episode
    """
    x, y_reward, y_cum_reward, y_timesteps = read_reward_timesteps_from_output_file(algorithm, curr_date)
    return x, y_reward


def animate(k):
    """
    Generate the animated plot with real time changes
    """
    x, y = get_reward_for_episode()
    plt.cla()
    plt.plot(x, y, 'k')  # single line
    plt.xlabel("Episode $\mathregular{E}$")
    plt.ylabel("Total reward $\mathregular{R(E)}$")
    plt.grid(True, color='gray', linestyle='dashed')
    plt.tight_layout()


def main():
    # Call the animate function, interval is the delay between frames in ms
    ani = FuncAnimation(plt.gcf(), animate, interval=20000)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
