"""
    Script for plotting the heatmap of the Q matrix of a single run of RL-IoT framework
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties
from plotter.support_plotter import print_cute_algo_name, read_reward_timesteps_from_output_file, \
    compute_avg_over_multiple_runs, build_output_dir_from_path, get_font_family_and_size, get_extension


font_family, font_size = get_font_family_and_size()

plt.rcParams["font.family"] = font_family
plt.rcParams['font.size'] = font_size

fontP = FontProperties()
fontP.set_size('x-small')
n_cols = 1

output_dir = './'


def plot_heatmap_single_run(filename, algorithm=None):
    """
    Generate heatmap for the Q matrix of one single run
    """
    target_output_dir = output_dir

    # Retrieve Q matrix into pandas dataframe called "data"
    data = pd.read_csv(filename)

    print(data)

    # Sort data inside the Q matrix to make it more readable
    ds = data.sum()
    temp = pd.DataFrame(ds, columns=['val']) \
        .drop(index=['Q'])
    new_idx = temp.sort_values('val', ascending=True).index
    new_data = data.reindex(columns=new_idx)
    new_data['label'] = data.Q

    # Order rows according to states
    new_data = new_data.set_index('label')
    new_data = new_data.sort_index()

    # Removing digits from states
    new_idx_cut = new_data.index  # .str.replace('\d+', '', regex=True)
    new_data['label'] = new_idx_cut
    new_data = new_data.set_index('label')
    print(new_data)
    print(new_data.shape)

    # Select top 15 columns
    new_data = new_data[new_data.columns[-15:]]

    # Define dimensions of the plot
    plt.figure(figsize=(6, 5))
    # plt.figure(figsize=(6, 2.8))

    # sns.heatmap(new_data, linewidth=0.0, cmap="magma_r", cbar_kws={'label': 'Expected reward $\mathregular{Q(s,a)}$'})
    # sns.heatmap(new_data, linewidth=0.0, cmap="magma_r", cbar_kws={'label': '$\mathregular{Q(s,a)}$'})
    sns.heatmap(new_data, annot=True, fmt='.1f', linewidth=0.0, cmap="magma_r", cbar_kws={'label': '$\mathregular{Q(s,a)}$'})
    sns.set(font_scale=8)
    y = [i + 0.5 for i in range(new_data.shape[0])]
    x = [i + 0.5 for i in range(new_data.shape[1])]
    plt.xticks(x, new_data.columns)
    plt.yticks(y, new_data.index, rotation=0)
    plt.xlabel("Action $\mathregular{a}$")
    plt.ylabel("State $\mathregular{s}$")
    plt.tight_layout()
    plt.savefig(target_output_dir + "heatmap_" + algorithm + get_extension())
    plt.show()


if __name__ == '__main__':
    input_dir = "csv_for_heatmap/"
    file_to_plot = input_dir + "prova_example_of_heatmap.csv"
    # file_to_plot = input_dir + 'cut_output_Q_2021_02_21_22_57_53_13018992640.csv'
    # file_to_plot = input_dir + 'output_Q_2021_02_22_03_16_09_13002203136.csv'
    plot_heatmap_single_run(file_to_plot, "qlearning")

