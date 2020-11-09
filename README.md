# iot


# THIS README IS WORK IN PROGRESS

General structure of directories:

* learning directory contains a learning module, with RL algorithms, and a run script to follow the best policy found by algorithms
* discovery contains script for finding IoT devices in LAN
* dictionary contains dictionaries for IoT protocols
* request\_builder accesses dictionaries and build requests to be sent to IoT devices
* device\_communication contains api for directly communicate with a specific IoT device
* state\_machine contains methods with state machine for RL and for computation of reward
* plotter contains script for plotting results
* sample contains some toy scripts to communicate to individual devices

The project can be run from the \_\_main\_\_ to have also a discovery part or from the learning\_yeelight script.


