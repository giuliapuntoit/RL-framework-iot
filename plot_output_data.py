import matplotlib.pyplot as plt
import csv

x = []
y_reward = []
y_cum_reward = []
y_timesteps = []

with open('output_csv/output_sarsa_data.csv', 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    next(reader, None)
    for row in reader:
        x.append(int(row[0]))
        y_reward.append(int(row[1]))
        y_cum_reward.append(int(row[2]))
        y_timesteps.append(int(row[3]))

plt.plot(x, y_reward, 'k--', label='rew')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward per episodes')
plt.legend(loc="center right")

plt.show()

plt.plot(x, y_cum_reward, 'k:', label='cum_rew')
plt.xlabel('Episodes')
plt.ylabel('Cumulative reward')
plt.title('Cumulative reward over episodes')
plt.legend(loc="center right")

plt.show()

plt.plot(x, y_timesteps, 'k', label='timesteps', marker='o')
plt.xlabel('Episodes')
plt.ylabel('Timestep to end of the episode')
plt.title('Timesteps per episode')
plt.legend(loc="center right")

plt.show()
