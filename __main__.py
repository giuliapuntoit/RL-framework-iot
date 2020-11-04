from threading import Thread

from config import GlobalVar
from discovery import network_analyzer
from learning import learning_yeelight

if __name__ == '__main__':
    devices = network_analyzer.analyze_lan()

    # For implementing a thread safe queue with list of devices and ip:
    # https://stackoverflow.com/questions/19369724/the-right-way-to-limit-maximum-number-of-threads-running-at-once
    print("\nSTART LEARNING PROCESS")
    for dev in devices:
        # Should launch a thread for each device, with a maximum number of threads available
        if dev.protocol == "yeelight":
            GlobalVar.directory = "./"
            protocol_thread = Thread(target=learning_yeelight.main())
            protocol_thread.start()
            protocol_thread.join()  # For now this thread is useless
    print("FINISH LEARNING PROCESS")
