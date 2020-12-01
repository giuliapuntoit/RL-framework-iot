
from threading import Thread

from config import GlobalVar
from discovery import network_analyzer
from learning import learning_yeelight

if __name__ == '__main__':
    # Discovery: find devices in the local network
    devices = network_analyzer.analyze_lan()

    # For implementing a thread safe queue with list of devices and ip:
    # https://stackoverflow.com/questions/19369724/the-right-way-to-limit-maximum-number-of-threads-running-at-once

    # th = []
    print("\nSTART LEARNING PROCESS")
    for dev in devices:
        # Learning: starting the learning process
        # Should launch a thread for each device, with a maximum number of threads available
        # Until now, this is done only for yeelight devices
        if dev.protocol == "yeelight":
            GlobalVar.directory = "./"
            tmp_th = Thread(target=learning_yeelight.main, args=(dev.as_dict(), ))
            tmp_th.start()
            tmp_th.join()  # Useless this thread for now
            # th.append(tmp_th)

    # for t in th:
    #     print("STARTING T")
    #     t.start()
    # for t in th:
    #     t.join()
    print("FINISH LEARNING PROCESS")
