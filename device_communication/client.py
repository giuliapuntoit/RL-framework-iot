"""
    General interface methods to access protocol-specific API script
    Each method receives needed parameters and the target protocol
"""

from device_communication import api_yeelight


def operate_on_bulb(method, params, discovery_report, protocol):
    if protocol == "yeelight":
        return api_yeelight.operate_on_bulb(method, params, discovery_report)


def operate_on_bulb_props(json_string, discovery_report, protocol):
    if protocol == "yeelight":
        return api_yeelight.operate_on_bulb_props(json_string, discovery_report)


def operate_on_bulb_json(json_string, discovery_report, protocol):
    if protocol == "yeelight":
        return api_yeelight.operate_on_bulb_json(json_string, discovery_report)
