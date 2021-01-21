"""
    Class to save and pass the information about found IoT devices
"""


class DiscoveryReport(object):
    # To facilitate passing information about results of the discovery process inside the network
    def __init__(self, result="", protocol="", timestamp="", ip="", port=""):
        self.result = result
        self.protocol = protocol
        self.timestamp = timestamp
        self.ip = ip
        self.port = port

    def as_dict(self):
        return {'result': self.result,
                'protocol': self.protocol,
                'timestamp': self.timestamp,
                'ip': self.ip,
                'port': self.port}
