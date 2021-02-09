"""
    Script for analyzing LAN to find IoT devices
"""

import pickle
import pprint
import nmap
import ipaddress
import time
from discovery.discovery_report import DiscoveryReport


def save_report_to_file(report, filename):
    with open(filename, 'wb') as report_file:
        pickle.dump(report, report_file)


def load_report_to_file(filename):
    with open(filename, 'rb') as report_file:
        report = pickle.load(report_file)
        return report


def analyze_lan():
    """
    Found IoT Shelly and Yeelight devices and creates 1 Discovery Report for each IoT device
    :return: list of Discovery Reports
    """

    # Instantiate a PortScanner object
    scanner = nmap.PortScanner()

    ip_to_scan = "192.168.1.0/24"  # you may want to change this last number if no devices are found
    print("START SCANNING LAN", ip_to_scan)
    print("This operation may take a while...")
    scan_range = scanner.scan(hosts=ip_to_scan, arguments='-sL')

    # print(scan_range['scan'])

    # I am assuming I already know protocol ports
    # yeelight_port = 1982
    yeelight_port = 55443
    shelly_port = 80

    devices = []

    # Look for Yeelight and Shelly devices
    for ip in ipaddress.IPv4Network(ip_to_scan):
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
    if len(devices) == 0:
        print("No found devices.\nPlease be sure devices are connected to LAN with an IP address in", ip_to_scan)
        print("If not, you can change the range of IPs.")

    else:
        print("FINISH SCANNING LAN\nALL IOT DEVICES:")
        cnt = 0
        for dev in devices:
            pp.pprint(dev.__dict__)
            save_report_to_file(dev, "reports" + str(cnt) + ".dictionary")
            cnt += 1
    return devices

    # Scan ports:
    # if yeelight
    # res1 = scanner.scan(target_yeelight, str(yeelight_port))
    # print(res1['scan'])

    # if shelly
    # res2 = scanner.scan(target_shelly, str(shelly_port))
    # print(res2['scan'])


if __name__ == '__main__':
    analyze_lan()
