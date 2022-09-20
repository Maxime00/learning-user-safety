cmake_minimum_required(VERSION 3.9)
project(learning_safety_margin)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  mpcCommand.msg
  mpcTraj.msg
)

catkin_python_setup()

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
    CATKIN_DEPENDS
    std_msgs
    sensor_msgs
    message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

set(CONTROL_SCRIPTS
    joint_space_velocity_control
    cartesian_twist_control
    idle_control
    joint_space_traj_replay_control
    cartesian_space_single_traj_follow_control
    cartesian_space_traj_follow_control
    user_evaluation_control
    MPC_velocity_planner
    MPC_velocity_control
)

foreach(SCRIPT ${CONTROL_SCRIPTS})
    catkin_install_python(PROGRAMS scripts/${SCRIPT} DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endforeach()


