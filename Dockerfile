ARG BASE_IMAGE_TAG=noetic
FROM ghcr.io/aica-technology/ros-control-libraries:${BASE_IMAGE_TAG}

WORKDIR /tmp
RUN git clone -b release/v1.1 --depth 1 https://github.com/aica-technology/network-interfaces.git && cd network-interfaces && \
  sudo bash install.sh --auto --no-cpp
RUN rm -rf /tmp/network-interfaces


WORKDIR /home/${USER}/ros_ws
COPY --chown=${USER} ./learning_safety_margin ./src/learning_safety_margin
# set up python env
RUN pip3 install -r ./src/learning_safety_margin/requirements.txt
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash; catkin_make"



