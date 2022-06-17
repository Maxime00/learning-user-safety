# Safety margin Instructions

Docker setup for Ahalya's project : learning user-specific safety margin

Available:
- Record kinesthetic demonstrations from Panda
- process data for learning
- learn trajectories with modulable safety 
- Command Panda to follow learned trajectories


## To connect to robot
Fwi command :
```console
franka_lightweight_interface 17 panda_ --sensitivity low --damping off
```

## Recording
Launch idle controller, use rosbags to record

### Terminal #1
```console
./build-server.sh
aica-docker interactive learning-safety-margin:noetic -u ros --net host --no-hostname -v /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data:/home/ros/ros_ws/src/learning_safety_margin/data
roslaunch learning_safety_margin demo.launch demo:=idle_control
```

### Terminal #2
```console 
aica-docker connect learning-safety-margin-noetic-runtime 
cd src/learning_safety_margin/data
rosbag record /joint_states
```

docker cp learning-safety-margin-noetic-ssh:/home/ros/ros_ws/src/learning_safety_margin/data/User_0/csv /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data/User_0/csv
sudo chown -R lasa /home/lasa/Workspace/learning_safety_DS/learning_safety_margin/data/
mv filename ../new_folder
rm filename

## Data processing

Currently will only process User_1 data -> TODO : make user specific
just make script that takes in user_number, checks for directories and adds them then calls processing and learning
should have second argumetn to decide if it learns vel or acc-> need to add other learnign script - > give dsitinct names to data_dict
```console 
aica-docker connect learning-safety-margin-noetic-runtime 
cd src/learning_safety_margin/franka_env
python3 bag2csv.py <user_number>
python3 learning_cbf_vel.py
```



## Run demos 
 
Currently runs with mpc controller
TODO : add controller to replay demos 
compute planned command less often
###Terminal #1
```console
roslaunch learning_safety_margin demo.launch demo:=cartesian_impedance_MPC_control user_number:=1
roslaunch learning_safety_margin demo.launch demo:=joint_torque_control
roslaunch learning_safety_margin demo.launch demo:=joint_space_velocity_control

```

# Notes 
to add to pycharm PYTHONPATH : 
$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages:/home/ros/ros_ws/devel/lib/python3/dist-packages:/home/ros/.local/lib/python3.8/site-packages

# Questions
How to pass argument with demo.launch file 

# TODO 
- remove transpose from learning 
- update bag2csv in process and data_learning
- test controller that uses mpc 
- get joitn states as csv from rosbag
- need logic to get target ? fom demos? or hardset it because all demos end in same point ?
- add rostopic to save cartesian EEF state + velocity
- add rostopic to controller (must be able to rosbag when commanding with cbfmpc)
- logic to replay demonstrations from rosbag /joint_states -> joitn controller ?

- make sure processing and controller code can run inside docker 


Code here is based on this [example](https://github.com/domire8/control-libraries-ros-demos/tree/main/rospy_zmq)


# `ros_examples` demonstration scripts

## Table of contents:

* [Prerequisites](#prerequisites)
* [Running a demo script](#running-demonstration-scripts)
* [Running a demo with the simulator](#running-the-simulator-simultaneously)
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

## Running demonstration scripts

If working with Docker, build and run the image with

```console
./build-server.sh
aica-docker interactive control-libraries-rospy-zmq-demos:noetic -u ros --net host --no-hostname
```

Running the scripts uses ROS commands, e.g. to run a script:

```console
roslaunch rospy_zmq_examples demo.launch demo:=<demo>
roslaunch rospy_zmq_examples demo.launch demo:=cartesian_twist_control
```

Available demos are:

- cartesian_twist_control
- joint_space_velocity_control

## Running the simulator simultaneously

The scripts require a simulator (or real robot with the same interface) to be running. Start the simulator with:

```console
cd path/to/desired/location
git clone -b develop git@github.com:epfl-lasa/simulator-backend.git
cd simulator-backend/pybullet_zmq
./build-server.sh
aica-docker interactive aica-technology/zmq-simulator -u ros2 --net host --no-hostname
python3 pybullet_zmq/bin/zmq-simulator
```

Once the simulator is running (in this case the franka simulator), do

```console
roslaunch rospy_zmq_examples demo.launch demo:=<demo> robot_name:=franka
```

Note that the robot name has to be the same as specified in the simulator, otherwise the topics won't be in the same
namespace and the demos don't work.

## Development

To run the Docker image as SSH server:

```console
bash build-server.sh -s
```
