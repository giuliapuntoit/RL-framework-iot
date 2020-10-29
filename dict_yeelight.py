import pprint
import json

# Control Protocol

# COMMAND message {id_pair, method_pair, params_pair}
# id_pair is        "id":<number>
# method_pair is    "method":"<method>"
# params_pair is    "params":["<param1>","<param2>", <numeric_param3>]
# <param1> is       "property":<property_value>
import sys


class DictYeelight(object):

    def __init__(self, method_requested=2):
        self.method_requested = method_requested
        # print("Using dictionary yeelight.")

    def run(self):
        properties = [('power', ""),  # values on off
                      ('bright', 0),  # range 1 100
                      ('ct', 0),  # range 1700 6500 (k)
                      ('rgb', 0),  # range 1 16777215
                      ('hue', 0),  # range 0 359
                      ('sat', 0),  # range 0 100
                      ('color_mode', 0),  # values 1 2 3
                      ('flowing', 0),  # values 0 1
                      ('delayoff', 0),  # range 1 60
                      ('flow_params', 0),  # ?
                      ('music_on', 0),  # values 0 1
                      ('name', ""),  # values set in set_name command
                      ('bg_power', ""),  # values on off
                      ('bg_flowing', 0),  # values 0 1
                      ('bg_flow_params', ""),  # ?
                      ('bg_ct', 0),  # range 1700 6500 (k?)
                      ('bg_lmode', 0),  # values 1 2 3
                      ('bg_bright', 0),  # range 0 100 (percentage)
                      ('bg_rgb', 0),  # range 1 16777215
                      ('bg_hue', 0),  # range 0 359
                      ('bg_sat', 0),  # range 0 100
                      ('nl_br', 0),  # range 1 100
                      ('active_mode', 0),  # values 0 1
                      ]

        # how to define values for each property? Maybe not necessary
        # sarebbe bello definire per ogni property il tipo (int string ecc)
        # how can i enforce a type?
        # for now:      0   ->  int
        #               ""  ->  string

        methods = []
        methods.extend(({"name": "get_prop",
                         "min_params": 1,
                         "max_params": -1,
                         "params_list": properties,
                         },
                        {"name": "set_ct_abx",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('ct_value', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ],
                         },
                        {"name": "set_rgb",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('rgb_value', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ],
                         },
                        {"name": "set_hsv",
                         "min_params": 4,
                         "max_params": 4,
                         "params_list": [
                             ('hue', 0),  # int
                             ('sat', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ],
                         },
                        {"name": "set_bright",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('brightness', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ],
                         },
                        {"name": "set_power",  # ON
                         "min_params": 3,
                         "max_params": 4,  # mode is optional
                         "params_list": [
                             ('power', "on"),  # string
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                             ('mode', 0),  # int
                         ],
                         },
                        {"name": "set_power",  # OFF TODO non il miglior modo per distinguerli ma vediamo se funziona
                         "min_params": 3,
                         "max_params": 4,  # mode is optional
                         "params_list": [
                             ('power', "off"),  # string
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                             ('mode', 0),  # int
                         ],
                         },
                        {"name": "toggle",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": [],
                         },
                        {"name": "set_default",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": [],
                         },
                        {"name": "start_cf",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('count', 0),  # int
                             ('action', 0),  # int
                             ('flow_expression', "")  # string
                         ],
                         },
                        {"name": "stop_cf",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": [],
                         },
                        {"name": "set_scene",
                         "min_params": 3,
                         "max_params": 4,
                         "params_list": [
                             ('class', ""),  # string
                             ('val1', 0),  # int
                             ('val2', 0),  # int
                             ('val3', 0)  # int, optional
                         ],
                         },
                        {"name": "cron_add",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('type', 0),  # int
                             ('value', 0),  # int
                         ]
                         },
                        {"name": "cron_get",
                         "min_params": 1,
                         "max_params": 1,
                         "params_list": [
                             ('type', 0),  # int
                         ]
                         },
                        {"name": "cron_del",
                         "min_params": 1,
                         "max_params": 1,
                         "params_list": [
                             ('type', 0),  # int
                         ]
                         },
                        {"name": "set_adjust",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('action', ""),  # string
                             ('prop', ""),  # string
                         ]},
                        {"name": "set_music",
                         "min_params": 1,
                         "max_params": 3,
                         "params_list": [
                             ('action', 0),  # int
                             ('host', ""),  # string
                             ('port', 0),  # int
                         ]},
                        {"name": "set_name",
                         "min_params": 1,
                         "max_params": 1,
                         "params_list": [
                             ('name', ""),  # string
                         ]},
                        {"name": "bg_set_rgb",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('rgb_value', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_set_hsv",
                         "min_params": 4,
                         "max_params": 4,
                         "params_list": [
                             ('hue', 0),  # int
                             ('sat', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_set_ct_abx",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('ct_value', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_start_cf",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('count', 0),  # int
                             ('action', 0),  # int
                             ('flow_expression', ""),  # string
                         ]},
                        {"name": "bg_stop_cf",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": [],
                         },
                        {"name": "bg_set_scene",
                         "min_params": 3,
                         "max_params": 4,
                         "params_list": [
                             ('class', ""),  # string
                             ('val1', 0),  # int
                             ('val2', 0),  # int
                             ('val3', 0),  # int optional
                         ]},
                        {"name": "bg_set_default",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": []
                         },
                        {"name": "bg_set_power",  # ON
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('power', "on"),  # string
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                             ('mode', 0),  # int
                         ]},
                        {"name": "bg_set_power",  # OFF
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('power', "off"),  # string
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                             ('mode', 0),  # int
                         ]},
                        {"name": "bg_set_bright",
                         "min_params": 3,
                         "max_params": 3,
                         "params_list": [
                             ('brightness', 0),  # int
                             ('effect', ""),  # string
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_set_adjust",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('action', ""),  # string
                             ('prop', ""),  # string
                         ]},
                        {"name": "bg_toggle",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": []
                         },
                        {"name": "dev_toggle",
                         "min_params": 0,
                         "max_params": 0,
                         "params_list": []
                         },
                        {"name": "adjust_bright",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('percentage', 0),  # int
                             ('duration', 0),  # int
                         ]},
                        {"name": "adjust_ct",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('percentage', 0),  # int
                             ('duration', 0),  # int
                         ]},
                        {"name": "adjust_color",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('percentage', 0),  # int
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_adjust_bright",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('percentage', 0),  # int
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_adjust_ct",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('percentage', 0),  # int
                             ('duration', 0),  # int
                         ]},
                        {"name": "bg_adjust_color",
                         "min_params": 2,
                         "max_params": 2,
                         "params_list": [
                             ('percentage', 0),  # int
                             ('duration', 0),  # int
                         ]},
                        ))

        # max_params = -1 means N -> non mi piace molto come scelta

        # i nomi dei parametri non coincidono con i nomi delle proprietà
        # anzi ce ne sono anche di piu (vedi duration)

        method_selected = 2

        if 0 <= self.method_requested < len(methods):
            method_selected = self.method_requested

        # if number set as first parameter take that, otherwise default is 2 (?)

        # method_selected = 2
        #
        # if len(sys.argv) == 2:
        #     if 0 <= int(sys.argv[1]) < len(methods):
        #         method_selected = int(sys.argv[1])

        # print("Method selected is number: " + str(method_selected))
        # print("Total methods is " + str(len(methods)))
        return methods[method_selected]


if __name__ == '__main__':
    method_returned = DictYeelight().run()

    # Useful information
    # print("Method is " + str(method_returned))

# serve_yeelight e' uno script intermedio che chiama dict_yeelight,
# elabora la risposta e poi la ritorna a request_yeelight/learning_yeelight
