import numpy as np
import csv

# First, I should call put the transition to states and computation of reward inside external methods
# to call them from here, not to re-write them!

# Q will be read from output_Q_data.csv
# Retrieve actions and state from output_Q_data.csv
# Statistics to compute Q will be read from output_parameters_data.csv

# Identify which RL algorithm was used and use it

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

optimal_policy = parameters['optimal_policy'].split(' ')  # Split values following spaces

print(optimal_policy)
# I should check values using tests not prints!!!

# I need some REAL generated files in order to be able to test and adjust the following code

# First, connecting to the device, initialization, something like this:

# if __name__ == '__main__':
# # MAIN
#
# # first discover the lamp and Connect to the lamp
# # start the bulb detection thread
# detection_thread = Thread(target=bulbs_detection_loop)
# detection_thread.start()
# # give detection thread some time to collect bulb info
# sleep(10)
#
# # show discovered lamps
# display_bulbs()
#
# print(bulb_idx2ip)
# max_wait = 0
# while len(bulb_idx2ip) == 0 and max_wait < 10:
#     sleep(1)
#     max_wait += 1
# if len(bulb_idx2ip) == 0:
#     print("Bulb list is empty.")
# else:
#     display_bulbs()
#     idLamp = list(bulb_idx2ip.keys())[0]
#
#     print("Waiting 5 seconds before using SARSA")
#     sleep(5)
#
#     RUNNING = False
#     # Do Sarsa
#
#     optimalPolicy, obtainedReward = SarsaSimplified(max_steps=100, total_episodes=50).run()
#     if optimalPolicy:
#         print("Optimal policy was found with reward", obtainedReward)
#     else:
#         print("No optimal policy reached with reward", obtainedReward)
#
# # goal achieved, tell detection thread to quit and wait
# RUNNING = False
# detection_thread.join()
# # done

# Then I can follow the found optimal policy:

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
# # Questo dovrebbe essere in un metodo esterno!!!
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
# # La computazione del reward dovrebbe essere in un metodo esterno!!!
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
