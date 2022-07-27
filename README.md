# Safety Margin Instructions

Docker setup for Ahalya's project : learning user-specific safety margin.

Contains a dockerized ROS package which works with the Franka Emika Panda robot to :
- Record kinesthetic demonstrations
- Process recorded data
- Learn a user-specific safety margin
- Replay demonstrations
- Play trajectories from learned user-specific model

## Table of contents:

* [Prerequisites](#prerequisites)
* [Connect to robot](#connect-to-robot)
* [Recording Demonstrations](#Recording)
* [Process Data](#Data-processing)
* [Running Controller](#run-controllers)
* [Development](#development)

## Prerequisites

This package contains control loops in a ROS environment. The folder is a fully functional ROS package that can
be directly copied in a ROS workspace.

There is also a `Dockerfile` provided that encapsulates the whole package in a containerized ROS environment. If you
want to run demo with the dockerized environment, you need to do the following first:

```console
cd path/to/desired/location
git clone https://github.com/aica-technology/docker-images.git
cd docker-images/scripts
./install-aica-docker.sh
```

Visit the [docker-images](https://github.com/aica-technology/docker-images) repository for more information on this.


## Connect to robot
### To connect to real robot
Go to franka-lightweight-interface and run docker with following commands :
```console
cd Workspace/franka-lightweight-interface
bash run-rt.sh
franka_lightweight_interface 17 panda_ --sensitivity low --joint-damping off
```

### To connect to simulator robot
Go to simulator-backend and run docker with following commands :
```console
cd Workspace/simulator-backend/pybullet_zmq
bash build-server.sh
aica-docker interactive aica-technology/zmq-simulator -u ros2 --net host --no-hostname
python3 pybullet_zmq/bin/zmq-simulator
```
Once the simulator is running, you can start your controllers as shown [below](#run-controllers)
For development, follow the instructions [below](#development)

To run controllers in the simulator, you must add 'robot_name:=franka' to the roslaunch command :
```console
roslaunch learning_safety_margin demo.launch demo:=joint_torque_traj_follow_control robot_name:=franka
```
Note that the robot name has to be the same as specified in the simulator, otherwise the topics won't be in the same
namespace and the demos don't work.

## Recording
Launch idle controller, use rosbags to record

### Terminal #1
```console
./build-server.sh
aica-docker interactive learning-safety-margin:noetic -u ros --net host --no-hostname 
aica-docker interactive learning-safety-margin:noetic -u ros --net host --no-hostname -v data_vol:/home/ros/ros_ws/src/learning_safety_margin/data
roslaunch learning_safety_margin demo.launch demo:=idle_control
```

### Terminal #2
```console 
aica-docker connect learning-safety-margin-noetic-runtime 
cd src/learning_safety_margin/data
rosbag record /joint_states
```

### some useful command for data handling
```console
docker cp learning-safety-margin-noetic-ssh:/home/ros/ros_ws/src/learning_safety_margin/data/User_0/csv /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data/User_0/csv
docker cp learning-safety-margin-noetic-ssh:/home/ros/ros_ws/src/learning_safety_margin/data/example_traj_to_replay/csv /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data/example_traj_to_replay
sudo chown -R lasa /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data/
mv filename ../new_folder
rm filename
```


## Data processing

Run and connect to learning-safety-margin docker container
Run process_and_learn.py script inside docker container with the following arguments :
-u <user_number>
-l <learning_algo> ['pos', 'vel']

```console 
aica-docker connect learning-safety-margin-noetic-runtime 
cd src/learning_safety_margin/process_data_and_learning
python3 process_and_learn.py -u <user_number> -l <learning_algo> 
```

## Run Controllers

### Terminal 1

Connect to either the real robot or simulator as shown [above](#connect-to-robot)
```console
cd Workspace/franka-lightweight-interface
bash run-rt.sh
franka_lightweight_interface 17 panda_ --sensitivity low --joint-damping off
```

### Terminal #2
Start docker container and launch controller from there

If running an ssh container, you can connect to it using :
```console
aica-docker connect learning-safety-margin-noetic-ssh -u ros
```
Note: the argument -u ros lets docker write files with local user permission

The trajectory replaying and following take in as argument the user_number and the number_of_traj per safety category to be played
Default is "0 2"

```console
bash build-server.sh
aica-docker interactive learning-safety-margin:noetic -u ros --net host --no-hostname -v data_vol:/home/ros/ros_ws/src/learning_safety_margin/data
roslaunch learning_safety_margin demo.launch demo:=joint_space_traj_replay_control args_for_control:="0 2"
roslaunch learning_safety_margin demo.launch demo:=cartesian_space_traj_follow_control args_for_control:="1 2"
roslaunch learning_safety_margin demo.launch demo:=joint_space_velocity_control
roslaunch learning_safety_margin mpc_control.launch robot_name:=franka args_for_planner:=0
```

## Development

To run the Docker image as SSH server:

```console
bash build-server.sh -s
```

You can then connect to it using
```console
aica-docker connect learning-safety-margin-noetic-ssh
```

Code here is based on this [example](https://github.com/domire8/control-libraries-ros-demos/tree/main/rospy_zmq)

# Notes 
to add to pycharm environment variable PYTHONPATH : 
$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages:/home/ros/ros_ws/devel/lib/python3/dist-packages:/home/ros/.local/lib/python3.8/site-packages

current working scripts : 
- idle_control (recording) 
- joint_space_traj_replay_control (successive replays)
- cartesian_space_single_traj_follow_control (follows single trajectory from CBF planner)
- cartesian_space_traj_follow_control (follow several trajectories from CBF planner)
- cartesian twist control (demo script)
- joint space velocity control (demo script)


# TODO 
- add option to start/stop integrator ? start thresh should be small but always reachable ( np.any instead? or start integrator if in same pos for too long )
- add 'show_plots' option ?
- make offline plot functions for mpc trajectories
- improve random positions of cbf traj planner -> must stay in reachable space



# Instructions for MPC-simulator pipeline

## Terminal 1 - run simulator
```console 
cd Workspace/simulator-backend/pybullet_zmq
bash build-server.sh
aica-docker interactive aica-technology/zmq-simulator -u ros2 --net host --no-hostname
python3 pybullet_zmq/bin/zmq-simulator
```

## Terminal 2 - run controller
```console
bash build-server.sh -s
aica-docker connect learning-safety-margin-noetic-ssh
roslaunch learning_safety_margin mpc_control.launch
```

Can edit code directly in pycharm without the need to rebuild, just ctrl+c in terminal 2 and do roslaunch again.
The mpc_control.launch file runs MPC_velocity_control and MPC_velocity_planner as two communicating nodes


### Instructions for MPC-simulator pipeline

# Terminal 1 - run simulator
```console 
cd Workspace/simulator-backend/pybullet_zmq
bash build-server.sh
aica-docker interactive aica-technology/zmq-simulator -u ros2 --net host --no-hostname
python3 pybullet_zmq/bin/zmq-simulator
```

# Terminal 2 - run controller
```console
bash build-server.sh -s -u ros
aica-docker connect learning-safety-margin-noetic-ssh -u ros
roslaunch learning_safety_margin mpc_control.launch
```

Can edit code directly in pycharm without the need to rebuild, just ctrl+c in terminal 2 and do roslaunch again
launch file runs MPC_velocity_control and MPC_velocity_planner