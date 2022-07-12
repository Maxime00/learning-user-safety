ARG BASE_IMAGE_TAG=noetic
FROM ghcr.io/aica-technology/ros-control-libraries:${BASE_IMAGE_TAG}

WORKDIR /tmp
RUN git clone -b v1.1.0 --depth 1 https://github.com/aica-technology/network-interfaces.git && \
    cd network-interfaces && sudo bash install.sh --auto --no-cpp
RUN rm -rf /tmp/network-interfaces

# set up python env, separate copy to avoid re-installing packages everytime
WORKDIR /home/${USER}/ros_ws
ADD ./learning_safety_margin/requirements.txt ./src/learning_safety_margin/requirements.txt
RUN pip3 install -r ./src/learning_safety_margin/requirements.txt

# set up MOSEK license
WORKDIR /home/${USER}
ADD ./learning_safety_margin/mosek.lic ./mosek/mosek.lic

WORKDIR /home/${USER}/ros_ws
COPY --chown=${USER} ./learning_safety_margin ./src/learning_safety_margin
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash; catkin_make"



