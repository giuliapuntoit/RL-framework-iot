# RL framework for IoT protocols

The goal of this project is to automatically learn the protocol of a generic IoT device in the shortest possible time, using Reinforcement Learning (RL) techniques.

This RL framework implements 4 RL algorithms:

* SARSA
* Q-learning
* SARSA(λ) 
* Q(λ) (Watkin's version)

These are used to automatize the interaction with the IoT devices present in the local network.
For these algorithms we assume there exists a dataset with valid protocol messages of different IoT devices. However, we have no further knowledge on the semantics of such command messages, nor on whether particular devices would accept the commands. This dataset will be stored into a dictionary inside our framework.

Here there is a first component based on the [Yeelight](https://www.yeelight.com/) protocol.


## Introduction to the project

### Motivation

In IT systems, the presence of IoT devices is exponentially growing and most of them are custom devices: they rely on proprietary protocols, often closed or poorly documented.
Here we want to interact with such devices, by learning their protocols in an autonomous manner.

### Definitions

* **state** of an IoT device: represented by some properties specific to that device.
* **state-machine** of a protocol: multiple series of states linked by one or more sequences of commands. These commands can be exchanged through that protocol to complete a predefined task.
* **task**: identified as a **path** inside the state-machine. The sequence of commands could change the **state** of the IoT device following a certain path - i.e., completing a task - inside the state-machine.
* RL algorithms iterate over 2 nested loops: the outer loop iterating over **episodes** and the inner loop iterating over **time steps** t.

### Current work

We mimic the behaviour of an attacker, which tries to explore the state-machine of the IoT device it is trying to communicate with.

We start developing our framework:

*  Targeting an actual IoT protocol: **Yeelight** protocol.
*  Implementing 4 RL algorithms: SARSA, Q-learning, SARSA(λ) and Q(λ).

**Note**: the Yeelight protocol defines a maximum rate on the commands to be sent to Yeelight devices, hence our framework can take about 50 minutes to complete the 1 learning process of 200 episodes for a single RL algorithm.

## How to use?

This project has been developed with Python 3.7. 
To use it, you need to first install all necessary Python packages with command:

```
pip install .
```

After installing all needed dependencies, the project can be executed directly running the ``__main__.py`` script or the ``learning_yeelight`` script.

### Structure

General structure of directories:

* ``learning`` directory contains a learning module, with RL algorithms, and a run script to follow the best policy found by algorithms.
* ``discovery`` contains scripts for finding IoT devices in LAN.
* ``dictionary`` contains dictionaries for IoT protocols.
* ``request_builder`` accesses dictionaries and builds requests to be sent to IoT devices.
* ``device_communication`` contains api for directly communicating with a specific IoT device.
* ``state_machine`` contains methods defining state machines for protocols and methods for the computation of the reward.
* ``plotter`` contains scripts for plotting results.
* ``sample`` contains some toy scripts to communicate to individual devices (Yeelight and Hue devices).
* ``images`` contains images for readme purposes.

The project can be run in 2 ways:

* from the ``__main__.py``, to have also a discovery part of all IoT devices in local network.
* from the ``learning_yeelight`` script to target only Yeelight devices inside the network.

#### Output

Throughout the entire learning process, the Learning module collects data into external files, inside the ``output`` directory.

All files for 1 execution of the learning process are identified by the current date in the format ``%Y_%m_%d_%H_%M_%S``.

The structure of the ``output`` directory is the following:

```
output
|
|__ log
|   |__ log_<date1>.log
|   |__ log_<date2>.log
|
|__ output_csv
|   |__ output_<algorithm1>_<date1>.csv
|   |__ output_<algorithm2>_<date2>.csv
|   |__ partial_output_<algorithm1>_<date1>.csv
|   |__ partial_output_<algorithm2>_<date2>.csv
|
|__ output_Q_parameters
|   |__ output_parameters_<date1>.csv
|   |__ output_parameters_<date2>.csv
|   |__ output_Q_<date1>.csv
|   |__ output_E_<date1>.csv
|
|__ log_date.log

```

More in details, inside ``output`` directory:

* ``output_Q_parameters``: contains data collected before and after the learning process. Before the process starts, all values for the configurable parameters are saved into file ``output_parameters_<date>.csv``: information about the path to learn, the optimal policy, the chosen algorithm, the number of episodes, the values of α, γ, λ and ε. If one wants to reproduce an execution of the learning process, all the parameters saved inside this file allow for repeating the learning process using the exact same configuration. Then, at the end of each episode, the Q matrix is written and updated inside a file ``output_Q_<date>.csv``. The E matrix, if required by the chosen RL algorithm, is written into ``output_E_<date>.csv``.
* ``output_csv``: contains ``output_<algorithm>_<date>.csv`` and ``partial_output_<algorithm>_<date>.csv`` files. The first file contains, for each episode, the obtained reward, the number of time steps and the cumulative reward. The latter contains the same values obtained while stopping the learning process at a certain episode and following the best policy found until that episode. ``partial_output_<algorithm>_<date>.csv`` files are present only if a proper flag is activated inside the ``learning_yeelight.py`` script, specifying the number of episodes at which the learning process should be stopped.
* ``log``: contains log data for each execution. After the learning process has started, for each step t performed by the RL agent, ``log_<date>.log`` is updated with information about the current state s<sub>t</sub>, the performed action a<sub>t</sub>, the new state s<sub>t+1</sub> and the reward r<sub>t+1</sub>.
* ``log_dates.log``: file saving the id of each execution. It can be used to collect all ids for all executions and use them inside the Plotter module.

### Workflow

The complete workflow is modelled in the following way:

<p align="center"><img src="./images/workflow-yeelight.png" height="800"></p>


Here there is an in-depth description of the previous figure:

1.  Normally, the framework starts through the ``__main__.py`` script, that first activates the Discoverer. Before starting, the ``config.py`` file provides some general information about the root directory in which saving output files, the state-machine and the goal that the RL agent should learn. Possible paths arbitrarily defined for the Yeelight protocol are shown inside the ``images`` directory.
2. The Discoverer analyzes the local network and returns to the main script all Discovery Reports describing found IoT devices of 2 protocols: Yeelight and Shelly. *Support for multiple protocols needs to be done.*
3. The main script receives these reports and generates a thread running the Learning module, passing to it the Discovery Report for 1 single Yeelight device found inside the LAN. *Support for concurrent threads is work in progress.*
4. The Learning module is the RL agent, iterating over episodes. 
	1. It receives multiple parameters as input: the chosen RL algorithm, values of ε, α, γ and λ if needed, total number of episodes, etc. Also some flags are present to decide whether after the learning process the user wants to directly plot some results, or wants to run the RL agent following the best policy found, using respectively the Plotter module or the Run Policy Found script.
	2. During each episode, the agent asks for commands to the Request Builder, which accesses data of the Yeelight Dictionary and returns a JSON string with the built command requested by the agent. This string can be sent to the Yeelight device.
	3. The JSON string is passed to the API Yeelight script inside the Device Communication module, that sends commands to the Yeelight bulb and handles its responses.
	4. Moreover, at each time step t the Learning module retrieves the reward r<sub>t</sub> and the current state s<sub>t</sub> from the State Machine module. In order to retrieve information about the state of the Yeelight device, this module asks to the Dictionary module the command to retrieve all necessary information from the bulb and sends this command to the API script, which actually sends the command to the bulb and returns the response to the State Machine Yeelight module.
	5. At the end of the learning process, the Learning module generates some output files, described in <a href="#output">Output</a> section.
5. The main thread waits until the thread running the Learning module ends. 


Generated output files can then be used by the **Run Policy Found** script, which retrieves data from these files and follows the best policy found, through the Q-matrix. While following the policy, the script retrieves complete commands from the Dictionary module and sends them to the Yeelight device passing through the API Yeelight script.
Also, output files can be used by the **Plotter** module to present graphically the results obtained in the learning process.


**Note**: for now the Learning process for Yeelight can also be executed directly, without using the main script. When this is the case, the Learning module performs a Yeelight specific discovery phase, accessing methods inside the API Yeelight script specific for this purpose. After information about Yeelight devices are retrieved, the Learning process works exactly as explained previously. *This feature will be removed soon.*

#### Plots (screenshots)

Since a lot of different plots can be generated, here there is a quick explanation on what graphs can be generated by scripts of the Plotter module.

* ``get_training_time_traffic.py`` and ``plot_training_time_traffic.py`` retrieve values of time of execution and traffic generated by each execution of the algorithm and generates these bar graphs:
  <p align="center"><img src="./images/training_times.png" height="170">       <img src="./images/training_traffic.png" height="170"></p>
* ``plot_moving_avg.py`` and ``plot_moving_avg_for_params.py`` show the following results respectively for different algorithms and for different values of parameters:
  <p align="center"><img src="./images/all_reward_plot_qlearning.png" height="170">       <img src="./images/all_timesteps_plot_qlearning.png" height="170"></p>
  <p align="center"><img src="./images/mavg_reward_plot.png" height="170">       <img src="./images/mavg_timesteps_plot.png" height="170"></p>
  <p align="center"><img src="./images/avg_rewards_for_algos.png" height="170">       <img src="./images/avg_steps_for_algos.png" height="170"></p>
* ``plot_cdf_reward.py``plots the CDF (Cumulative Distribution Function) of the reward:
  <p align="center"><img src="./images/cdf_rewards_multiple_algo.png" height="170"></p>
* ``plot_reward_per_request.py`` shows the cumulative reward over the number of commands sent:
  <p align="center"><img src="./images/all_commands_all_algos.png" height="170">
* ``plot_output_data.py`` shows reward and time step results for 1 single execution (It can be used for check the correct working:
<p align="center"><img src="./images/single_output.png" height="270">
* ``plot_animation.py`` generates an animated plot in real time while the algorithm is working. Once the algorithm has started, the current date can be retrieved from the ``log_date.log`` file and copied into the ``plot_animation.py`` script. Once this script has started, it will generate a real time plot as the one showed in <a href="#demo">Demo</a> section.


**Note**:

- all scripts use arrays of dates in format ``%Y_%m_%d_%H_%M_%S`` to identify executions of RL algorithms. 
- most of the scripts save plots inside subdirectories of the Plot directory. The target directory can be manually chosen inside each script.


## Demo

A short demo of the working of the Learning process, showed through the console and an animated plot can be seen in [demo](https://drive.google.com/file/d/1vNQbgy6AtDedNQ9U6nRPkNA36Z8X2tYQ/view?usp=sharing).

Recall that this demo was done using the previously described ``plot_animation.py``script, in order to create an animated plot.

## Features

Main features include:

* Support to 4 RL algorithms, that can selected inside the ``learning_yeelight.py`` script.
* Collect all necessary data to generate plots for comparing performance among different configurations.
* Block the learning process and restart it from the Q matrix computed before, giving as id the date of the previous execution.

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




