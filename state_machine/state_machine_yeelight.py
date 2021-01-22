"""
    Script to define all information about state-machines, defined states, paths and reward definition
"""

from device_communication.api_yeelight import operate_on_bulb_props
from config import FrameworkConfiguration
from request_builder.builder_yeelight import BuilderYeelight
from time import sleep


def get_states():
    # Returns an array containing all the states for a specified path

    states = []
    if FrameworkConfiguration.path == 1:
        # PATH 1
        states = ["0_off_start", "1_on", "2_rgb", "3_bright", "4_rgb_bright", "5_off_end",
                  "6_invalid"]

    elif FrameworkConfiguration.path == 2:
        # PATH 2
        states = ["0_off_start", "1_on", "2_rgb", "3_bright", "4_rgb_bright", "5_off_end",
                  "6_invalid", "7_name", "8_on_name", "9_on_name_rgb", "10_on_name_bright", "11_on_name_rgb_bright"
                  ]

    elif FrameworkConfiguration.path == 3:
        # PATH 3
        # If you want to visually check the optimal path related to this path,
        # start from 0_off_start and from rgb value set to 255 (blue)
        # in order to check that the lamp does not change the rgb value
        states = ["0_off_start", "1_on", "2_rgb", "3_bright", "4_rgb_bright", "5_off_end",
                  "6_invalid", "7_name", "8_on_name", "9_on_name_rgb", "10_on_name_bright", "11_on_name_rgb_bright",
                  "12_ct", "13_ct_bright", "14_rgb_ct", "15_rgb_bright_ct", "16_off_middle", "17_on_middle",
                  "18_ct_middle",
                  "19_rgb_middle", "20_bright_middle", "21_ct_rgb_middle", "22_ct_bright_middle",
                  "23_rgb_bright_middle", "24_ct_rgb_bright_middle",
                  ]
    return states


def get_optimal_policy():
    # Returns an array containing an optimal policy for a specified path

    optimal_policy = []
    if FrameworkConfiguration.path == 1:
        # PATH 1
        optimal_policy = [5, 2, 4, 6]

    elif FrameworkConfiguration.path == 2:
        # PATH 2
        optimal_policy = [5, 17, 4, 6]

    elif FrameworkConfiguration.path == 3:
        # PATH 3
        optimal_policy = [5, 1, 4, 6, 5, 1, 6]

    return optimal_policy


def get_optimal_path():
    # Return an array containing the optimal path

    optimal_path = []
    if FrameworkConfiguration.path == 1:
        # PATH 1
        optimal_path = [0, 1, 2, 4, 5]

    elif FrameworkConfiguration.path == 2:
        # PATH 2
        optimal_path = [0, 1, 8, 10, 5]

    elif FrameworkConfiguration.path == 3:
        # PATH 3
        optimal_path = [0, 1, 12, 13, 16, 17, 18, 5]

    return optimal_path


def compute_next_state_from_props(current_state, old_props_values, discovery_report):
    # Given the current state, send a request to the yeelight device
    # to get the current property values. From that information, based on the selected
    # path, it computes and returns the next state

    next_state = current_state

    # Get the json command for asking the desired properties
    json_command = BuilderYeelight(method_chosen_index=0, select_all_props=True).run()
    # props_names = BuilderYeelight().get_all_properties()

    # Send the json command to the yeelight device
    props_values = operate_on_bulb_props(json_command, discovery_report)

    if not props_values or len(props_values) < 7:
        # print("\t\tSomething went wrong from get_prop: keeping the current state") TODO
        return current_state, old_props_values

    sleep(0.5)

    # Indexes of properties:
    power_index = 0
    bright_index = 1
    rgb_index = 2
    ct_index = 3
    name_index = 4
    hue_index = 5
    sat_index = 6

    if FrameworkConfiguration.path == 1:
        # PATH 1
        # 0 1 2 4 5
        if props_values[power_index] == 'off':
            if current_state == 0 or current_state == 6:
                next_state = 0
            else:
                next_state = 5  # end state
        elif props_values[power_index] == 'on':
            if current_state == 0:  # if the device was previously off it turns to on
                next_state = 1
            else:
                if old_props_values:
                    if current_state == 1 and props_values[bright_index] != old_props_values[bright_index] and \
                            props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 4
                    elif current_state == 1 and props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 2
                    elif current_state == 3 and props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 4
                    elif current_state == 1 and props_values[bright_index] != old_props_values[bright_index]:
                        next_state = 3
                    elif current_state == 2 and props_values[bright_index] != old_props_values[bright_index]:
                        next_state = 4
    elif FrameworkConfiguration.path == 2:
        # PATH 2
        # 0 1 8 10 5
        if props_values[power_index] == 'off':
            if old_props_values and props_values[name_index] != old_props_values[name_index] and current_state == 0:
                next_state = 7
            elif current_state == 0 or current_state == 6:
                next_state = 0
            else:
                next_state = 5  # end state
        elif props_values[power_index] == 'on':
            name_modified = (old_props_values and props_values[name_index] != old_props_values[name_index])
            if current_state == 0:
                next_state = 1
            elif current_state == 7:  # name already modified
                next_state = 8
            elif current_state == 1 and name_modified:
                next_state = 8
            else:
                if old_props_values:
                    if current_state == 1 and props_values[bright_index] != old_props_values[bright_index] and \
                            props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 4
                    elif current_state == 1 and props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 2
                    elif current_state == 1 and props_values[bright_index] != old_props_values[bright_index]:
                        next_state = 3
                    # case when name modified already
                    if current_state == 8 and props_values[bright_index] != old_props_values[bright_index] and \
                            props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 11
                    elif current_state == 8 and props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 9
                    elif current_state == 8 and props_values[bright_index] != old_props_values[bright_index]:
                        next_state = 10
                    elif current_state == 3 and name_modified:
                        next_state = 10
                    elif current_state == 3 and props_values[rgb_index] != old_props_values[rgb_index]:
                        next_state = 4
                    elif current_state == 2 and name_modified:
                        next_state = 9
                    elif current_state == 2 and props_values[bright_index] != old_props_values[bright_index]:
                        next_state = 4
                    elif current_state == 4 and name_modified:
                        next_state = 11

    elif FrameworkConfiguration.path == 3:
        # PATH 3
        # 0 1 12 13 16 17 18 5
        if props_values[power_index] == 'off':
            if current_state == 0 or current_state == 6:
                next_state = 0  # Turn to initial off state
            elif current_state in [1, 2, 3, 4, 12, 13, 14, 15, 16]:  # Turn to intermediate off state
                next_state = 16
            else:
                next_state = 5  # Turn to final off state
        elif props_values[power_index] == 'on':
            if old_props_values:
                bright_modified = (props_values[bright_index] != old_props_values[bright_index])
                rgb_modified = (props_values[rgb_index] != old_props_values[rgb_index])
                ct_modified = (props_values[ct_index] != old_props_values[ct_index])
                if current_state == 0:
                    next_state = 1
                elif current_state == 16:
                    next_state = 17
                else:
                    if current_state == 1 and bright_modified and rgb_modified and ct_modified:
                        next_state = 15
                    elif current_state == 1 and bright_modified and rgb_modified:
                        next_state = 4
                    elif current_state == 1 and rgb_modified and ct_modified:
                        next_state = 14
                    elif current_state == 1 and bright_modified and ct_modified:
                        next_state = 13
                    elif current_state == 1 and rgb_modified:
                        next_state = 2
                    elif current_state == 1 and bright_modified:
                        next_state = 3
                    elif current_state == 1 and ct_modified:
                        next_state = 12
                    elif current_state == 3 and rgb_modified and ct_modified:
                        next_state = 15
                    elif current_state == 3 and rgb_modified:
                        next_state = 4
                    elif current_state == 3 and ct_modified:
                        next_state = 13
                    elif current_state == 2 and bright_modified and ct_modified:
                        next_state = 15
                    elif current_state == 2 and bright_modified:
                        next_state = 4
                    elif current_state == 2 and ct_modified:
                        next_state = 14
                    elif current_state == 12 and rgb_modified and bright_modified:
                        next_state = 15
                    elif current_state == 12 and rgb_modified:
                        next_state = 14
                    elif current_state == 12 and bright_modified:
                        next_state = 13
                    elif current_state == 13 and rgb_modified:
                        next_state = 15
                    elif current_state == 17 and (rgb_modified or bright_modified):
                        next_state = 19
                    elif current_state == 17 and ct_modified:
                        next_state = 18

    return next_state, props_values


def compute_reward_from_states(current_state, next_state):
    # Compute the reward given by the transition from current_state to next_state
    # The reward depends on the path that the learning process is trying to learn

    reward_from_props = 0

    # Reward is given only to the last step, based on the path followed from the start to the end

    if FrameworkConfiguration.path == 1:
        # PATH 1
        if current_state == 0 and next_state == 1:
            reward_from_props = 1  # per on
        elif current_state == 1 and next_state == 3:
            reward_from_props = 2  # per bright
        elif current_state == 1 and next_state == 2:
            reward_from_props = 3  # per rgb
        elif current_state == 3 and next_state == 4:
            reward_from_props = 4  # per rgb bright
        elif current_state == 2 and next_state == 4:
            reward_from_props = 5  # per rgb bright
        elif current_state == 4 and next_state == 5:  # Last step
            reward_from_props = 200
            FrameworkConfiguration.reward += reward_from_props
            tmp = FrameworkConfiguration.reward
            FrameworkConfiguration.reward = 0
            return tmp
        FrameworkConfiguration.reward += reward_from_props
        return 0

    elif FrameworkConfiguration.path == 2:
        # PATH 2
        if current_state == 0 and next_state == 1:
            reward_from_props = 2
        elif current_state == 0 and next_state == 7:
            reward_from_props = 1
        elif current_state == 7 and next_state == 8:
            reward_from_props = 2
        elif current_state == 1 and next_state == 8:
            reward_from_props = 10
        elif current_state == 8 and next_state == 10:
            reward_from_props = 10
        elif current_state == 11 and next_state == 5:
            reward_from_props = 15
            FrameworkConfiguration.reward += reward_from_props
            tmp = FrameworkConfiguration.reward
            FrameworkConfiguration.reward = 0
            return tmp
        elif current_state == 10 and next_state == 5:
            reward_from_props = 200
            FrameworkConfiguration.reward += reward_from_props
            tmp = FrameworkConfiguration.reward
            FrameworkConfiguration.reward = 0
            return tmp
        FrameworkConfiguration.reward += reward_from_props
        return 0

    elif FrameworkConfiguration.path == 3:
        # PATH 3
        if current_state == 0 and next_state == 1:
            reward_from_props = 1
        elif current_state == 1 and next_state == 3:
            reward_from_props = 2
        elif current_state == 1 and next_state == 12:
            reward_from_props = 10
        elif current_state == 3 and next_state == 13:
            reward_from_props = 2
        elif current_state == 12 and next_state == 13:
            reward_from_props = 5
        elif current_state == 13 and next_state == 16:
            reward_from_props = 20
        elif current_state == 16 and next_state == 17:
            reward_from_props = 2
        elif current_state == 17 and next_state == 18:
            reward_from_props = 10
        elif current_state == 18 and next_state == 5:
            reward_from_props = 200
            FrameworkConfiguration.reward += reward_from_props
            tmp = FrameworkConfiguration.reward
            FrameworkConfiguration.reward = 0
            return tmp
        FrameworkConfiguration.reward += reward_from_props
        return 0
    return reward_from_props
