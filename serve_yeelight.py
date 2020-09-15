from dict_yeelight import DictYeelight
import random
import pprint
import json

# randomize method selected

# length of methods is 35
n = random.randint(0, 35)
method_selected = DictYeelight(method_requested=n).run()

# print("Method selected is number " + str(n))
print(str(method_selected))

print("Parsing method selected")
method_name = method_selected["name"]
method_min_params = int(method_selected["min_params"])
method_max_params = int(method_selected["max_params"])
method_params_list = method_selected["params_list"]

params = []
cnt = 0
# For now we ignore optional parameters
for (key, value) in method_params_list:
    if cnt >= method_min_params:
        break
    # ci vorrà un controllo sul tipo di dato da mandare (int, string)
    if value == "":
        n = random.randint(0, 20)
        if n % 2 == 0:
            value = "on"
        else:
            value = "false"
    elif value == 0:
        value = random.randint(0, 1000)
    params.append(value)
    cnt += 1

id = 0  # questo dovrebbe essere un parametro che gli passi esternamente

command = {"id": id,  # id_pair
           "method": method_name,  # method_pair
           "params": params,  # params_pair
           }

print("Final command is " + str(command))

pp = pprint.PrettyPrinter(indent=4)
with open("data_file.json", "w") as write_file:
    json.dump(command, write_file)
json_string = json.dumps(command)
pp.pprint(json_string)
