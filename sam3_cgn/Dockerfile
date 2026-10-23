FROM osrf/ros:jazzy-desktop

# Install Python package manager
RUN apt-get update && apt-get install -y python3-pip

# Install PyTorch with CUDA support
RUN pip3 install --break-system-packages \
    torch==2.10.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install additional SAM3 dependencies
RUN pip3 install --break-system-packages \
    einops \
    pycocotools
    
# RealSense / ROS image dependencies
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-realsense2-camera \
    ros-$ROS_DISTRO-realsense2-description \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-image-geometry \
    ros-$ROS_DISTRO-camera-info-manager \
    ros-$ROS_DISTRO-rqt-image-view \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /workspace/sam3_cgn/ros2_ws

CMD ["/bin/bash"]
