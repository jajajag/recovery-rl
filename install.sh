#!/bin/bash
sudo apt install -y python3-pip ffmpeg zip unzip libsm6 libxext6 libgl1-mesa-dev libosmesa6-dev libgl1-mesa-glx patchelf

# create conda environment for recovery-rl
#conda create --name recovery-rl
#conda activate recovery-rl

# install Python dependencies in a virtualenv
pip3 install numpy scipy dotmap matplotlib tqdm opencv-python tensorboardX moviepy plotly gdown
pip3 install torch==1.4.0
pip3 install torchvision==0.5.0
#conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#pip3 install numpy --upgrade
#pip3 install mujoco_py==1.50.1.68
pip3 install 0.15.7
pip3 install mujoco_py==2.0.2.13
