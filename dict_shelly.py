import pprint
import json

# Focusing only on Shelly Bulb commands in site https://shelly-api-docs.shelly.cloud/#shelly-bulb

# In Color mode, each output channel: Red, Green, Blue and White is individually controllable.
# In Warm/Cold White mode the device accepts color temperature and brightness
# and adjusts power output of all channels to produce the desired white light.

predefined_effects = [
    (0, "Off"),
    (1, "Meteor Shower"),
    (2, "Gradual Change"),
    (3, "Breath"),
    (4, "Flash"),
    (5, "On/Off Gradual"),
    (6, "Red/Green Change"),
]

# on and off commands, setting the color, brightness and effects using a JSON payload

# to control simply on-off
# shellies/shellybulb-<deviceid>/color/0/command with command on or off
# shellies/shellybulb-<deviceid>/color/0 to know the current on off state

# to control other parameters of the LED channel
# shellies/shellybulb<deviceid>/color/0/set


attributes_settings_light = [
    ("ison", ""),              # string "on" or "off"
    ("red", 0),                 # int 0 255 in mode "color"
    ("green", 0),               # int 0 255 in mode "color"
    ("blue", 0),                # int 0 255 in mode "color"
    ("gain", 0),                # int 0 100 in mode "color"
    ("brightness", 0),          # int 0 100 in mode "white"
    ("white", 0),               # int 0 255 in mode "color"
    ("temp", 0),                # int 3000 6500 in mode "white"
    ("effect", 0),              # int 0 6
    ("default_state", ""),      # string "on" "off" or "last"
    ("auto_on", 0),             # int seconds
    ("auto_off", 0),            # int seconds
    ("power", 0),               # int ?
    ("schedule", True),         # boolean True or False
    ("schedule_rules", []),     # array of strings, rules for schedule activation
]

# differenza tra attributes and parameters???
parameters_settings_light = [
    ("reset", ""),              # any non-zero-length value
    ("effect", 0),              # int 0 6
    ("default_state", ""),      # string "on" "off" or "last"
    ("auto_on", 0),             # int seconds
    ("auto_off", 0),            # int seconds
    ("schedule", True),         # boolean True or False
    ("schedule_rules", []),     # array of strings, rules for schedule activation
]

attributes_light_0 = [
    ("ison", ""),               # string "on" or "off"
    ("has_timer", True),        # boolean true or false
    ("timer_remaining", 0),     # int seconds
    ("mode", ""),               # string "color" or "white
    ("red", 0),                 # int 0 255 in mode "color"
    ("green", 0),               # int 0 255 in mode "color"
    ("blue", 0),                # int 0 255 in mode "color"
    ("gain", 0),                # int 0 100 in mode "color"
    ("brightness", 0),          # int 0 100 in mode "white"
    ("white", 0),               # int 0 255 in mode "color"
    ("temp", 0),                # int 3000 6500 in mode "white"
    ("effect", 0),              # int 0 6
]

# parameters allow to send commands
# while attributes are responds to just see the current state?
parameters_light_0 = [
    ("mode", ""),               # string "color" or "white
    ("timer", 0),               # int seconds
    ("turn", ""),               # string "on" "off" or "toggle"
    ("red", 0),                 # int 0 255 in mode "color"
    ("green", 0),               # int 0 255 in mode "color"
    ("blue", 0),                # int 0 255 in mode "color"
    ("gain", 0),                # int 0 100 in mode "color"
    ("brightness", 0),          # int 0 100 in mode "white"
    ("white", 0),               # int 0 255 in mode "color"
    ("temp", 0),                # int 3000 6500 in mode "white"
    ("effect", 0),              # int 0 6
]

# to obtain latest device state
# shellies/shellybulb-<deviceid>/color/0/status

# Settings
# shellies/shellybulb-<deviceid>/settings
# esistono un sacco di altre settings

# attributes (equal to method of yeelight?)
settings = []
settings.extend(({"name": "settings",
                  "min_params": 2,
                  "max_params": 2,
                  "params_list": [("mode", ["white", "color"]),
                                  ("lights", attributes_settings_light)],
                  },
                 {"name": "settings/light/0",
                  "min_params": 1,                          # are they all mandatory???
                  "max_params": 15,
                  "params_list": attributes_settings_light, # devo mettere attributes o parameters?
                  },
                 {"name": "settings/color/0",  # same as lights
                  "min_params": 1,
                  "max_params": 15,
                  "params_list": attributes_settings_light,
                  },
                 {"name": "settings/white/0",  # same as lights
                  "min_params": 1,
                  "max_params": 15,
                  "params_list": attributes_settings_light, # devo mettere attributes o parameters?
                  },
                 {"name": "settings/power/0",
                  "min_params": 1,
                  "max_params": 1,
                  "params_list": [0],   # 0 4000 watt?
                  },
                 {"name": "status", # current list of light channels (see lights/0)
                  "min_params": 2,
                  "max_params": 2,
                  "params_list": [("lights", attributes_light_0),   # or parameters?
                                  ("meters", [("power", 0),         # same value as settings/power/0
                                            ("is_valid", True)])    # always true (minuscolo)
                                  ],
                  },
                 {"name": "light/0",
                  "min_params": 1,
                  "max_params": 11,
                  "params_list": parameters_light_0,       # parameters or attributes?
                  },
                {"name": "color/0",
                  "min_params": 1,
                  "max_params": 11,
                  "params_list": parameters_light_0,       # paramters or attributes?
                  },
                {"name": "white/0",
                  "min_params": 1,
                  "max_params": 11,
                  "params_list": parameters_light_0,       # paramters or attributes?
                  },
))

device_id = 0

min_p = settings[3]["min_params"]
max_p = settings[3]["max_params"]


params = []
for (key,value) in settings[3]["params_list"]:
    # in base ai valori di min_params e max_params si deciderà quanti parametri mettere
    # e che valori assegnargli
    # ci vorrà un controllo sul tipo di dato da mandare (int, string)
    params.append(key + ":" + str(value))
    # non mi piace per ora come li formatta
    # le stringe dovrebbero avere le virgolette

command = {"get": "shellies/shellybulb"+ str(device_id) + "/" + settings[3]["name"],                                # id_pair
           "params": params,
           }

pp = pprint.PrettyPrinter(indent=4)

json_string = json.dumps(command)

pp.pprint(json_string)
