import pprint
import nmap
import ipaddress
import time
from discovery.discovery_report import DiscoveryReport


def analyze_lan():
    """
    Found IoT Shelly and Yeelight devices and creates 1 Discovery Report for each IoT device
    :return: list of Discovery Reports
    """

    # Instantiate a PortScanner object
    scanner = nmap.PortScanner()

    print("START SCANNING LAN")
    scan_range = scanner.scan(hosts="192.168.1.1/24", arguments='-sL')

    # print(scan_range['scan'])

    # I am assuming I already know protocol ports
    yeelight_port = 1982
    shelly_port = 80

    devices = []

    # Look for Yeelight and Shelly devices
    for ip in ipaddress.IPv4Network('192.168.1.0/24'):
        hostname = scan_range['scan'][str(ip)]['hostnames'][0]['name']
        if "yeelink" in hostname or "yeelight" in hostname:
            target_yeelight = str(ip)
            print("\tDEVICE: Found yeelight at", target_yeelight)
            devices.append(DiscoveryReport(result=scan_range['scan'][target_yeelight], protocol="yeelight",
                                           timestamp=time.time(), ip=target_yeelight, port=yeelight_port))
        elif "shelly" in hostname:
            target_shelly = str(ip)
            print("\tDEVICE: Found shelly at", target_shelly)
            devices.append(DiscoveryReport(result=scan_range['scan'][target_shelly], protocol="shelly",
                                           timestamp=time.time(), ip=target_shelly, port=shelly_port))

    pp = pprint.PrettyPrinter(indent=4)
    print("FINISH SCANNING LAN\nALL IOT DEVICES:")
    for dev in devices:
        pp.pprint(dev.__dict__)
    return devices

    # TODO try not to use already known ports!
    # Scan ports:
    # if yeelight
    # res1 = scanner.scan(target_yeelight, str(yeelight_port))
    # print(res1['scan'])

    # if shelly
    # res2 = scanner.scan(target_shelly, str(shelly_port))
    # print(res2['scan'])


if __name__ == '__main__':
    analyze_lan()
