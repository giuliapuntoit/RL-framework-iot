import json

import numpy as np
import csv
from threading import Thread
from time import sleep

# First, I should call put the transition to states and computation of reward inside external methods
# to call them from here, not to re-write them!

# Q will be read from output_Q_data.csv
# Retrieve actions and state from output_Q_data.csv
# Statistics to compute Q will be read from output_parameters_data.csv

# Identify which RL algorithm was used and use it
from learning_yeelight import bulbs_detection_loop, display_bulbs, operate_on_bulb, operate_on_bulb_json, \
    compute_next_state, compute_reward_from_props
from serve_yeelight import ServeYeelight

directory = 'output_Q_parameters'
date = 'data'  # Date must be in format %H_%M_%S_%d_%m_%Y
file_Q = 'output_Q_' + date + '.csv'
file_parameters = 'output_parameters_' + date + '.csv'

actions = []
states = []
Q = []

parameters = {}

# Retrieving Q matrix, states and actions
with open(directory + '/' + file_Q, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for index_row, row in enumerate(reader):
        if index_row == 0:  # Remember to remove first cell
            for index_col, col in enumerate(row):
                if index_col != 0:
                    actions.append(str(col.strip()))
        else:
            for index_col, col in enumerate(row):
                if index_col == 0:
                    states.append(str(col))

try:
    tmp_matrix = np.genfromtxt(directory + '/' + file_Q, delimiter=',', dtype=np.float32)
    Q = tmp_matrix[1:, 1:]

except Exception as e:
    print("Wrong file format:", e)
    exit(1)

print(states)
print(actions)
print(Q)

# I should check values using tests not prints!!!
# TODO:
# del tipo, runnare lo script con certi valori e verificare che il risultato sia quello scritto nei test
# ad esempio certi valori di parametri, questa esatta lunghezza e questi esatti stati e azioni
# il risultato deve essere coerente con ciò che mi aspetto
# cosicché cambiando parametri posso aspettarmi un risultato giusto

if len(states) != len(Q) or len(actions) != len(Q[0]) or np.isnan(np.sum(Q)):
    print("Wrong file format: wrong Q dimensions or nan values present")
    exit(2)

with open(directory + '/' + file_parameters, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

print(parameters)  # For now the are all strings
# I think the optimal policy that I chose should be inserted into parameters,
# otherwise I do not know how to compare results
# I should check values using tests not prints!!!

optimal_policy = parameters['optimal_policy'].split('-')  # Split policy string and save it into a list

print(optimal_policy)
# I should check values using tests not prints!!!

# I need some REAL generated files in order to be able to test and adjust the following code

detected_bulbs = {}
bulb_idx2ip = {}
RUNNING = True
current_command_id = 0
MCAST_GRP = '239.255.255.250'

# First, connecting to the device, initialization, something like this:

# first discover the lamp and Connect to the lamp
# start the bulb detection thread
detection_thread = Thread(target=bulbs_detection_loop)
detection_thread.start()
# give detection thread some time to collect bulb info
sleep(10)

# show discovered lamps
display_bulbs()

print(bulb_idx2ip)
max_wait = 0
while len(bulb_idx2ip) == 0 and max_wait < 10:
    sleep(1)
    max_wait += 1
if len(bulb_idx2ip) == 0:
    print("Bulb list is empty.")
else:
    display_bulbs()
    idLamp = list(bulb_idx2ip.keys())[0]

    print("Waiting 5 seconds before verifying optimal obtained by SARSA")
    sleep(5)

    seconds_to_wait = int(parameters['seconds_to_wait'])
#
    # Then I can follow the found optimal policy:
    print("Setting power off")
    operate_on_bulb(idLamp, "set_power", str("\"off\", \"sudden\", 0"))
    sleep(seconds_to_wait)
    state1 = 0

    print("Restarting... returning to state: off")
    t = 0
    final_policy = []
    final_reward = 0

    while t < 10:
        max_action = np.argmax(Q[state1, :])
        final_policy.append(max_action)
        print("Action to perform is", max_action)

        json_string = ServeYeelight(idLamp=idLamp, method_chosen_index=max_action).run()
        json_command = json.loads(json_string)
        print("Json command is " + str(json_string))
        reward_from_response = operate_on_bulb_json(idLamp, json_string)
        sleep(seconds_to_wait)

        state2 = compute_next_state(json_command, states, state1)

        print("From state", state1, "to state", state2)

        reward_from_props = compute_reward_from_props(state1, state2)

        tmp_reward = -1 + reward_from_response + reward_from_props  # -1 for using a command more
        final_reward += tmp_reward

        if state1 == 3 and state2 == 4:
            print("Done")
        state1 = state2
        t += 1

    print("Length final policy is", len(final_policy))
    print("Final policy is", final_policy)
    if len(final_policy) == len(optimal_policy) and np.array_equal(final_policy, optimal_policy):
        print("Optimal policy found with reward: " + str(final_reward))
    else:
        print("Not optimal policy found with reward: " + str(final_reward))

# goal achieved, tell detection thread to quit and wait
RUNNING = False
detection_thread.join()
# done
