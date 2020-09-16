from dict_yeelight import DictYeelight
import random
import pprint
import json


# randomizing method selected

class ServeYeelight(object):
    def __init__(self, idLamp=0, method_chosen_index=-1):
        self.method_chosen_index = method_chosen_index
        self.id = idLamp
        print("Using serve yeelight.")

    def run(self):

        if self.method_chosen_index == -1:
            # length of methods is 35
            self.method_chosen_index = random.randint(0, 35-1)

        method_selected = DictYeelight(method_requested=self.method_chosen_index).run()

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
        if method_max_params == -1:
            n = random.randint(1, len(method_params_list)-1)
            sample_params_list = random.sample(method_params_list, n)
            # print(sample_params_list)
            for (key, value) in sample_params_list:
                params.append(key)
        else:
            for (key, value) in method_params_list:
                # print("Key " + str(key))
                # print("Value " + str(value))
                if cnt >= method_min_params:
                    break
                n = random.randint(0, 20)
                if key == "power":
                    if n % 2 == 0:
                        value = "on"
                    else:
                        value = "off"
                elif key == "effect":
                    if n % 2 == 0:
                        value = "sudden"
                    else:
                        value = "smooth"
                elif key == "prop":
                    if n % 3 == 0:
                        value = "bright"
                    elif n % 3 == 1:
                        value = "ct"
                    else:
                        value = "color"
                elif key == "class":
                    if n % 5 == 0:
                        value = "color"
                    elif n % 5 == 1:
                        value = "hsv"
                    elif n % 5 == 2:
                        value = "ct"
                    elif n % 5 == 3:
                        value = "cf"
                    else:
                        value = "auto_delay_off"
                elif value == 0:
                    value = random.randint(0, 10000)
                else:
                    value = "Random string"
                params.append(value)
                cnt += 1

        command = {"id": self.id,  # id_pair
                   "method": method_name,  # method_pair
                   "params": params,  # params_pair
                   }

        # Check json command inside file
        with open("data_file.json", "w") as write_file:
           json.dump(command, write_file)

        return json.dumps(command)


if __name__ == '__main__':
    json_command = ServeYeelight().run()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(json_command)
