"""
    Main script that starts the framework
"""

from threading import Thread
from config import FrameworkConfiguration
from discovery import network_analyzer
from learning import learning_yeelight


if __name__ == '__main__':
    # Discovery: find devices in the local network
    devices = network_analyzer.analyze_lan()

    # For implementing a thread safe queue with list of devices and ip:
    # https://stackoverflow.com/questions/19369724/the-right-way-to-limit-maximum-number-of-threads-running-at-once

    th = []
    print("\nSTART LEARNING PROCESS")
    for dev in devices:
        # Learning: starting the learning process
        # Should launch a thread for each device, with a maximum number of threads available
        # Until now, this is done only for yeelight devices
        cnt = 0
        if dev.protocol == "yeelight":
            FrameworkConfiguration.directory = "./"
            if cnt <= FrameworkConfiguration.max_threads:
                tmp_th = Thread(target=learning_yeelight.main, args=(dev.as_dict(), ))
                cnt += 1
                # tmp_th.start()
                # tmp_th.join()  # Useless this thread for now
                th.append(tmp_th)

    for t in th:
        print("STARTING THREADS")
        t.start()

    for t in th:
        t.join()
        print("JOINT THREADS")
    print("FINISH LEARNING PROCESS")


# TODO qui dovrei poter scegliere tra runnare solo il run o anche il learning, se runnare il run mettere una data
#  nel config dovrei scrivere la data se scelgo il run
#  dovrei salvare i device found dei discovery report in un file chiamato devices
# I do not need run script to be multithread write?
