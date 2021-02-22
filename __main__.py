"""
    Main script that starts the framework
"""

from threading import Thread
from config import FrameworkConfiguration
from discovery import network_analyzer, yeelight_analyzer
from learning import learning_yeelight
from discovery.network_analyzer import load_report_to_file

if __name__ == '__main__':
    # Discovery: find devices in the local network

    # 2 possible ways:
    # print("Analyzing LAN using nmap")
    # devices = network_analyzer.analyze_lan()

    print("Analyzing LAN using Yeelight discovery")
    devices = yeelight_analyzer.main()

    th = []
    print("\nSTART FRAMEWORK EXECUTION")
    cnt = 0

    if len(devices) == 0:
        print("No device found. Try to use a previously detected device.")
        discovery_report = load_report_to_file("reports0.dictionary").as_dict()
        tmp_th = Thread(target=learning_yeelight.main, args=(discovery_report,))
        cnt += 1
        th.append(tmp_th)

    for dev in devices:
        # Learning: starting the learning process
        # Should launch a thread for each device, with a maximum number of threads available
        # Until now, this is done only for yeelight devices

        if dev.protocol == "yeelight":
            FrameworkConfiguration.directory = "./"
            if cnt <= FrameworkConfiguration.max_threads:
                tmp_th = Thread(target=learning_yeelight.main, args=(dev.as_dict(), ))
                cnt += 1
                th.append(tmp_th)
        elif dev.protocol == "shelly":
            # pass for now
            pass

    if len(th) > 0:
        print("STARTING THREADS")
        for t in th:
            t.start()

        for t in th:
            t.join()
        print("JOINED THREADS")
    else:
        print("No suitable device found.")
    print("FINISH FRAMEWORK EXECUTION")
