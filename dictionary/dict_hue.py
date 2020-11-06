import pprint
import json

bridge_ip_address = "192.168.1.1"

id = 0
base_address = "http://" + bridge_ip_address + "/api/username/lights/" + str(id) + "/"

# username there is always unless:
# exceptions are creating a username and getting basic bridge information – Login Required

get_method = "GET"  # get information
# with get message I can obtain a response with a JSON containing attributes
put_method = "PUT"  # change a setting

# If you’re doing something that isn’t allowed,
# maybe setting a value out of range or typo in the resource name,
# then you’ll get an error message letting you know what’s wrong.

arguments = [
    ("on", True),  # boolean (on off of light)
    ("bri", 0),  # int 1 254
    ("hue", 0),  # int 0 65535
    ("sat", 0),  # int 0 254
    ("xy", [0.0, 0.0]),  # list of float from 0 to 1
    ("ct", 0),  # int 153 500 ?
    ("alert", ""),  # string "none" "select" "lselect"
    ("effect", ""),  # string "none" "colorloop"
    ("transitiontime", 0),  # int s (1 is 100ms)
    ("bri_inc", 0),  # int -254 +254
    ("sat_inc", 0),  # int -254 +254
    ("hue_inc", 0),  # int -65534 +65534
    ("ct_inc", 0),  # int -65534 +65534
    ("xy_inc", [0.0, 0.0]),  # list of float from -0.5 +0.5 ?
]

# Commands missing
settings = []
# Consider only light for Hue
settings.extend(({"name": "state",
                  "min_params": 1,
                  "max_params": 14,
                  "params_list": arguments,
                  },
                 ))
'''
config
lights
groups
config
schedules
scenes
sensors
rules'''

command = {"base_address": base_address + settings[0]["name"], }
for (key, value) in settings[0]["params_list"]:
    # Basing on the values of min_params and max_params we will decide how many parameters to send
    # and which value to assign to them
    command[key] = value

pp = pprint.PrettyPrinter(indent=4)

json_string = json.dumps(command)

pp.pprint(command)
