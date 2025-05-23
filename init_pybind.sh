#!/bin/bash

# pybind依赖安装脚本，基于notebook 910B云环境，python3.9，cann8.0.0.alpha003
# 910B云环境可直接执行脚本，基础赛道根据自身的设备参考安装。

pip3 install pip==21.0.1
pip3 install pybind11==2.13.1 numpy==1.24 expecttest
pip3 install tensorflow==2.16.1
pip3 install torch==2.1.0
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.1.0/torch_npu-2.1.0.post10-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.1.0.post10-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
python3 -m pip install protobuf==3.20.0 -i https://mirrors.huaweicloud.com/repository/pypi/simple