"""
    Class acting as a broker script that calls the DictYeelight class
    This class asks for information on a command and builds the command with all retrieved info
"""

import string
import random
import pprint
import json

from dictionary.dict_yeelight import DictYeelight


def get_all_properties():
    """
    Method that returns the name of all properties
    """
    method_chosen_index = 0
    method_selected = DictYeelight(method_requested=method_chosen_index).run()
    method_params_list = method_selected["params_list"]

    params = []
    for (key, value) in method_params_list:
        params.append(key)
    return params


def build_command_yeelight(method_chosen_index=-1, select_all_props=False):
    """
    Enable access to the dictionary and constructs command
    Builds the selected Yeelight command and sends it back as a json command
    """

    id_for_command = 1  # id = 1 for all commands since for every command the framework creates a d

    if method_chosen_index == -1:
        # length of methods is 37
        method_chosen_index = random.randint(0, 37 - 1)

    # Retrieve the method_chosen_index-th command from Yeelight dictionary
    method_selected = DictYeelight(method_requested=method_chosen_index).run()

    # Parsing selected method
    method_name = method_selected["name"]
    method_min_params = int(method_selected["min_params"])
    method_max_params = int(method_selected["max_params"])
    method_params_list = method_selected["params_list"]

    params = []
    cnt = 0
    # We ignore optional parameters
    if method_max_params == -1:
        n = random.randint(1, len(method_params_list) - 1)
        if select_all_props:
            # Returns the properties needed to get the current state of the bulb
            sample_params_list = [('power', ""),    # values on off
                                  ('bright', 0),    # range 1 100
                                  ('rgb', 0),       # range 1 16777215
                                  ('ct', 0),        # range 1700 6500 (k)
                                  ('name', ""),     # values set in set_name command
                                  ('hue', 0),       # range 0 359
                                  ('sat', 0),       # range 0 100
                                  ]
        else:
            # Pick a random number of properties from the available ones
            sample_params_list = random.sample(method_params_list, n)
        for (key, value) in sample_params_list:
            params.append(key)
    else:
        for (key, value) in method_params_list:
            # Assign (random) values to parameters
            if cnt >= method_min_params:
                break
            n = random.randint(0, 20)
            if key == "power":
                pass
            elif key == "effect":
                if (2 * n + 1) % 2 == 0:
                    value = "smooth"
                else:
                    value = "sudden"  # For now, to simplify thing the effect is always "sudden"
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
                if key == "rgb_value":
                    value = random.randint(0, 16777215)
                elif key == "brightness":
                    value = random.randint(1, 100)
                elif key == "percentage":
                    value = random.randint(-100, 100)
                elif key == "ct_value":
                    value = random.randint(1700,6500)
                else:
                    value = random.randint(0, 100)  # Random values
            else:
                value = ''.join(random.choice(string.ascii_uppercase) for _ in range(6))  # Random string
            params.append(value)
            cnt += 1

    # Build the command structure
    command = {"id": id_for_command,  # id_pair
               "method": method_name,  # method_pair
               "params": params,  # params_pair
               }

    # Save json command inside external file
    # with open("data_file.json", "w") as write_file:
    #     json.dump(command, write_file)

    return json.dumps(command)


if __name__ == '__main__':
    json_command = build_command_yeelight(method_chosen_index=31)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(json_command)
