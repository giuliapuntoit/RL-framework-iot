class DiscoveryReport(object):
    # To facilitate passing information about results of the discovery process inside the network
    def __init__(self, result="", protocol="", timestamp="", ip="", port=""):
        self.result = result
        self.protocol = protocol
        self.timestamp = timestamp
        self.ip = ip
        self.port = port
