# Safety Margin Instructions

Docker setup for Ahalya's project : learning user-specific safety margin

## Table of contents:

* [Prerequisites](#prerequisites)
* [Connect to robot](#connect-to-robot)
* [Recording Demonstrations](#Recording)
* [Process Data](#Data-processing)
* [Running Controller](#run-controllers)
* [Development](#development)

## Prerequisites

This package contains control loop examples in a ROS environment. The folder is a fully functional ROS package that can
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
franka_lightweight_interface 17 panda_ --sensitivity low --damping off
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
NOTE : To run controllers in the simulator, IP ports of RobotInterface should be 'robot_interface = RobotInterface("*:1601", "*:1602")'
Similarly, binded ports in aica-server command in build-server.sh script should be '-p1601:1601 -p1602:1602'

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
aica-docker connect learning-safety-margin-noetic-ssh
```

TODO : add arguments to chose user_number and safety of replayed traj
currently
```console
bash build-server.sh
aica-docker interactive learning-safety-margin:noetic -u ros --net host --no-hostname -v data_vol:/home/ros/ros_ws/src/learning_safety_margin/data
roslaunch learning_safety_margin demo.launch demo:=cartesian_impedance_MPC_control user_number:=1
roslaunch learning_safety_margin demo.launch demo:=joint_torque_control
roslaunch learning_safety_margin demo.launch demo:=joint_torque_traj_follow_control args_for_control:="test 1"
roslaunch learning_safety_margin demo.launch demo:=joint_torque_one_traj_MPC_control
 robot_name:=franka
roslaunch learning_safety_margin demo.launch demo:=cartesian_twist_traj_follow_control
roslaunch learning_safety_margin demo.launch demo:=joint_space_velocity_control
```
future 
```console
bash build-server.sh
aica-docker interactive learning-safety-margin:noetic -u ros --net host --no-hostname -v data_vol:/home/ros/ros_ws/src/learning_safety_margin/data
roslaunch learning_safety_margin demo.launch demo:=joint_torque_traj_follow_control user_number:=1 safety:=safe
roslaunch learning_safety_margin mpc_control.launch robot_name:=franka args_for_planner:=0
```



# Notes 
to add to pycharm environment variable PYTHONPATH : 
$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages:/home/ros/ros_ws/devel/lib/python3/dist-packages:/home/ros/.local/lib/python3.8/site-packages

current working scripts : 
- idle_control ( recording) 
- joint_torque_traj_follow (replays)
- cartesian twist control (demo script)
- joint space velocity control (demo script)



# TODO 
- 
- add logic to play several trajectories of varying safety in planner - > check ahalya' scode, make somethi nto match that
- add integrator to replays
- replays must choose and output safety and traj_number
- add arguments for user_specific and safety 
- Tune gains
- add data_recording in mpc controller
- add script to handle several MPC trajectories and pause between each until next input 
- save reference traj to plot against recorded later 

tried updating using state instead of time -> no improvment
tried using 5 steps of online mpc-> jittery traj

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

