# RL framework for IoT protocols

In this work the goal is to automatically learn the protocol of a generic IoT device in the shortest possible time, using Reinforcement Learning (RL) techniques.

This RL framework implements 4 RL algorithms:

* SARSA
* Q-learning
* SARSA(λ) 
* Q(λ) (Watkin's version)

These are used to automatize the interaction with the IoT devices present in a local network.
For these algorithms we assume there exists a dataset with valid protocol messages of different IoT devices. However, we have no further knowledge on the semantics of such command messages, nor on whether particular devices would accept the commands. This dataset will be stored into a dictionary inside our framework.

We implemented a first component based on the Yeelight protocol.

## Introduction to the project

### Motivation

In IT systems, the presence of Internet of Things (IoT) devices is exponentially growing. Most of them are custom devices, and they rely on proprietary protocols, often closed or poorly documented. Here we want to interact with such devices, by learning their protocols in an autonomous manner.

### Definitions

* We define the **state-machine** of a protocol as multiple series of states linked by one or more sequences of commands. These commands can be exchanged through that protocol to complete a predefined task.
* A **task** can be identified as a **path** inside the state-machine. The sequence of commands could change the **state** of the targeted IoT device, that is represented by some properties specific to that device, following a certain path - i.e., completing a task - inside the state-machine.

### Current work

We mimic the behaviour of an attacker, which tries to explore the state-machine of the IoT device it is trying to communicate with.

We start implementing our framework targeting an actual IoT protocol: the **Yeelight** protocol. Moreover, in order to evaluate multiple techniques and compare their performance, we implement four RL algorithms: SARSA, Q-learning, SARSA(λ) and Q(λ).

**Note**: the Yeelight protocol defines a maximum rate on the commands to be sent to Yeelight devices, hence our framework takes at least 50 minutes to complete the entire learning process, for 200 episodes.

## How to use?

After installing all needed dependencies, The project can be run from the ``__main__.py`` script or from the ``learning_yeelight`` script.

### Structure

General structure of directories:

* ``learning`` directory contains a learning module, with RL algorithms, and a run script to follow the best policy found by algorithms
* ``discovery`` contains script for finding IoT devices in LAN
* ``dictionary`` contains dictionaries for IoT protocols
* ``request_builder`` accesses dictionaries and build requests to be sent to IoT devices
* ``device_communication`` contains api for directly communicate with a specific IoT device
* ``state_machine`` contains methods with state machine for RL and for computation of reward
* ``plotter`` contains script for plotting results
* ``sample`` contains some toy scripts to communicate to individual devices (Yeelight and Hue devices)

The project can be run from the ``__main__.py`` to have also a discovery part or from the ``learning_yeelight`` script to used Yeelight devices inside the network.

#### Output

Throughout the entire learning process, the Learning module collects data into external files, inside the ``output`` directory.

All files for one execution of the learning process are identified by the current date in the format ``%Y_%m_%d_%H_%M_%S``.

Inside ``output`` directory:

* ``output_Q_parameters``: this directory contains info collected before and after the learning process. Before the process starts, all values for the configurable parameters are saved into file ``output_parameters_<date>.csv``:.information about the path to learn, the optimal policy, the algorithm chosen, the number of episodes, the values for α, γ, λ and ε. If one wants to reproduce an execution of the learning process, all the parameters saved inside this file allow for repeating the learning process using the exact same configuration. Then, at the end of each iteration of the outer loop, the Q matrix is written and updated inside a file ``output_Q_<date>.csv``. The E matrix if present is written into ``output_E_<date>.csv``.
* ``output_csv``: this directory contains ``output_<algorithm>_<date>.csv`` and ``partial_output_<algorithm>_<date>.csv``. The first file contains for each episode the reward obtained, the number of time steps and the cumulative reward. The latter contains these values when the learning process is stopped and the best policy found by the Q matrix is followed. So it contains reward and time steps when following the Q matrix suggested policy. The latter file is present only if a proper flag is activated inside the ``learning_yeelight.py`` script, and the number of episodes at which stopping the learning process should be specified.
* ``log``: this directory contains log data for each execution. After the learning process has started, for each step t performed by the RL agent a ``log_<date>.log`` file is updated, with information about the current state st, the performed action at, the new state st+1 and the reward rt+1.
* ``log_dates.log`` is a file appending date and algorithm for each execution. It can be used to collect all ids for all executions and put inside the scripts inside the Plotter module.

### Workflow

Complete structure of the project is modelled like this:




1.  Before starting, in the ``config.py`` some general information is present, for example the root directory in which saving output files or the state-machine and the goal that the RL agent should learn.
2. When the framework starts, it is managed by the ``__main__.py`` script. This script first activates the Discoverer.
3. After analyzing the local network, the Discoverer returns to the main script all reports created after the found devices.
4. The main script receives these reports and calls the Learning module, passing to it also the Discovery Report for the Yeelight bulb found inside the LAN.
5. When the Learning module starts, it receives in input multiple parameters. Among these, the most important are the RL algorithm to use, with values for ε, α, γ and λ if needed, the total number of episodes, the number of maximum time steps per episode and some flags. These flags decide whether after the learning process the user wants to plot some results or to run the RL agent following the best policy found, using respectively the Plotter module or the Run Policy Found script.
6. The Learning module is our RL agent, which iterates over episodes. For each episode it iterates until a terminal state is reached or when the number of time steps for that episode is above a certain threshold. 
7. During each episode, the agent asks for commands to the Request Builder, which access data from the Dictionary - the Dictionary for the Yeelight in our case - and returns to the agent a JSON string with the complete command to send to the Yeelight device.
8. The Learning module then passes this string to the Device Communication module, more specifically to the API script for the Yeelight protocol, which sends commands to the Yeelight bulb and handles its responses.
9. Moreover, at each time step the Learning module retrieves data about reward and current state from the State Machine module, more specifically from the State Machine Yeelight script. The State Machine module, in order to retrieve information about the state of the bulb, asks to the Dictionary module the command to retrieve all necessary information from the bulb and sends this command to the API script, which actually sends the command to the bulb and returns the response to the State Machine Yeelight module.
10. At the end of the learning process, the Learning module generates some output files, for storing the Q matrix, intermediate reward and number of time steps values during the episodes and some log files, which can be used for error checking.
11. Output files can be used by the Run Policy Found script, which retrieves data from these files and follows the best policy found, through the Q-matrix. While following the policy, the script retrieves complete commands from the Dictionary module and sends them to the Yeelight device passing through the API Yeelight script. Also, output files can be used by the Plotter module to present graphically the results obtained in the learning process.
12. Recall that for the Yeelight component the Learning process can also be started directly, without passing through the main and Discoverer parts. When this is the case, the Learning module performs a Yeelight specific discovery phase, accessing methods inside the API Yeelight script specific for this purpose. After information about the Yeelight bulb are retrieved, the Learning process works exactly as explained previously.

More information can be found in my master's degree thesis.

 
## Screenshots

(TODO screenshot of output and video demo)

## Features

Main features include:

* Support to 4 RL algortihms, that can selected inside the ``learning_yeelight.py``script
* Block the learning process and restart it from the Q matrix computed before, giving as id the date of the previous execution

## Tests

No tests present for now.


## Contribute

Pull Requests are always welcome.

Ensure the PR description clearly describes the problem and solution. It should include:

* Name of the module modified
* Reasons for modification


## Authors

* **Giulia Milan** - *Initial work* - [giuliapuntoit](https://github.com/giuliapuntoit)

See also the list of [contributors](https://github.com/giuliapuntoit/RL-framework-iot/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Previous implementation of RL algorithms in TCP toycase scenario: [RL-for-TCP](https://github.com/giuliapuntoit/RL-tcp-toycase)

* SARSA implementation example: [SARSA-example](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/?ref=rp)

* How to evaluate RL algorithms: [RL-examples](https://towardsdatascience.com/reinforcement-learning-temporal-difference-sarsa-q-learning-expected-sarsa-on-python-9fecfda7467e)




