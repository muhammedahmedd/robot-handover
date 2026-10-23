FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create Python 3.9 environment
RUN conda create -n cgn_tf210 python=3.9 -y

ENV PATH=/opt/conda/envs/cgn_tf210/bin:/opt/conda/bin:$PATH

ENV LD_LIBRARY_PATH=/opt/conda/envs/cgn_tf210/lib/python3.9/site-packages/tensorflow:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tested TensorFlow and NumPy versions
RUN python -m pip install \
    "tensorflow==2.10.1" \
    "numpy==1.23.5"

# Additional Contact-GraspNet dependencies
RUN python -m pip install \
    "scipy==1.10.1" \
    "opencv-python-headless==4.8.1.78" \
    "trimesh==3.23.5" \
    "Pillow==10.4.0" \
    "PyYAML==5.4.1" \
    "tqdm==4.67.1" \
    "pyrender==0.1.45" \
    "pyglet==1.5.27" \
    "PyOpenGL==3.1.0"
    
# VTK and Mayavi dependencies
RUN python -m pip install \
    "vtk==9.2.6" \
    "traits==7.1.0" \
    "traitsui==8.0.0" \
    "pyface==8.0.0"

# For visualization and plotting
RUN python -m pip install --no-build-isolation "mayavi==4.8.2"

# Qt support for Matplotlib visualization
RUN python -m pip install "PyQt5==5.15.11"

# Verify package dependencies
RUN python -m pip check

# Contact-GraspNet repository mounted at runtime
WORKDIR /workspace/contact_graspnet

CMD ["/bin/bash"]
