"""
    General interface to access each protocol-specific builder script
    Each method receives needed parameters and the target protocol
"""

from request_builder import builder_yeelight


def get_all_properties(protocol):
    """
    Method that returns the name of all properties of a device for a specified protocol
    """
    if protocol == "yeelight":
        return builder_yeelight.get_all_properties()


def build_command(method_chosen_index, select_all_props, protocol):
    """
    Method to access the dictionary and constructs commands
    """
    if protocol == "yeelight":
        return builder_yeelight.build_command_yeelight(method_chosen_index, select_all_props)
