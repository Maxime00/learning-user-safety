cmake_minimum_required(VERSION 3.9)
project(learning_safety_margin)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
)

catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS
    std_msgs
    sensor_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

set(CONTROL_SCRIPTS
    idle_control
    joint_space_traj_replay_control
    cartesian_space_traj_follow_control
    user_eval_joint_control
)

foreach(SCRIPT ${CONTROL_SCRIPTS})
    catkin_install_python(PROGRAMS scripts/${SCRIPT} DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endforeach()
