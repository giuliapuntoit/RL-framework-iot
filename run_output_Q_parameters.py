import numpy as np
import csv

# First, I should call put the transition to states and computation of reward inside external methods
# to call them from here, not to re-write them!


# Q will be read from output_Q_data.csv
# Retrieve actions and state from output_Q_data.csv
# Statistics to compute Q will be read from output_parameters_data.csv

# Identify which RL algorithm was used and use it

directory = 'output_Q_parameters'
file_Q = 'output_Q_data.csv'
file_parameters = 'output_parameters_data.csv'

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

if len(states) != len(Q) or len(actions) != len(Q[0]) or np.isnan(np.sum(Q)):
    print("Wrong file format: wrong Q dimensions or nan values present")
    exit(2)

with open(directory + '/' + file_parameters, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    parameters = {rows[0].strip(): rows[1].strip() for rows in reader}

print(parameters)  # For now the are all strings

# I need some REAL generated files in order to be able to test and adjust the following code

# First, connecting to the device, initialization

# print("Setting power off")
# operate_on_bulb(idLamp, "set_power", str("\"off\", \"sudden\", 0"))
# sleep(self.seconds_to_wait)
# current_state = 0
#
# if not self.disable_graphs:
# print("Restarting... returning to state: off")
# t = 0
# final_policy = []
# final_reward = 0
# optimal = [5, 2, 4, 5]
# while t < 10:
# state = current_state
# max_action = np.argmax(Q[state, :])
# final_policy.append(max_action)
#
# previous_state = current_state
#
# json_string = ServeYeelight(idLamp=idLamp, method_chosen_index=max_action).run()
# json_command = json.loads(json_string)
# print("Json command is " + str(json_string))
# operate_on_bulb_json(idLamp, json_string)
# sleep(self.seconds_to_wait)
#
# state1 = previous_state
#
# if json_command["method"] == "set_power" and json_command["params"] and json_command["params"][
#     0] == "on" and state1 == 0:
#     state2 = states.index("on")
# elif json_command["method"] == "set_power" and json_command["params"] and json_command["params"][
#     0] == "on" and state1 == 1:
#     state2 = states.index("on")
# elif json_command["method"] == "set_rgb" and state1 == 1:
#     state2 = states.index("rgb")
# elif json_command["method"] == "set_rgb" and state1 == 2:
#     state2 = states.index("rgb")
# elif json_command["method"] == "set_bright" and state1 == 2:
#     state2 = states.index("brightness")
# elif json_command["method"] == "set_bright" and state1 == 3:
#     state2 = states.index("brightness")
# elif json_command["method"] == "set_power" and json_command["params"][
#     0] == "off" and state1 == 3:  # whatever state
#     state2 = states.index("off_end")
# elif json_command["method"] == "set_power" and json_command["params"][0] == "off":  # whatever state
#     state2 = states.index("off_start")
# else:
#     # state2 = states.index("invalid") # fingiamo che se sbagli comando rimango sempre nello stesso stato
#     state2 = state1
#
# tmp_reward = -1 + tot_reward
#
# if state1 == 0 and state2 == 1:
#     tmp_reward += 1
# if state1 == 1 and state2 == 2:
#     tmp_reward += 2
# if state1 == 2 and state2 == 3:
#     tmp_reward += 4
# if state1 == 3 and state2 == 4:
#     tmp_reward += 1000
#     done = True
#
# current_state = state2
#
# final_reward += tmp_reward
# if not self.disable_graphs:
#     print("New state", current_state)
# if previous_state == 3 and current_state == 4:
#     print("New state", current_state)
#     break
# t += 1
#
# print("Length final policy is", len(final_policy))
# print("Final policy is", final_policy)
