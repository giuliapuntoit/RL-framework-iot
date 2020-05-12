import pprint
import json

bridge_ip_address="192.168.1.1"

base_address = "http://"+bridge_ip_address+"/api/username"

# username there is always unless:
# exceptions are creating a username and getting basic bridge information – Login Required

get_method = "GET" # get information
# with get message I can obtain a response with a JSON containing attributes
put_method = "PUT" # change a setting

# If you’re doing something that isn’t allowed,
# maybe setting a value out of range or typo in the resource name,
# then you’ll get an error message letting you know what’s wrong.

settings = []
settings.extend(({"name": "config",
                  "min_params": 1,
                  "max_params": 1,
                  "params_list": [("name", ""),         # string
                                 ],
                  },
                 ))

'''lights
groups
config
schedules
scenes
sensors
rules'''
