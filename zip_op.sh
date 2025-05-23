#!/bin/bash

op_name=$1
# 算子工程目录,自行设定路径
op_dir="./${op_name}"
echo $op_dirs
rm -rf ${op_name}_zip ${op_name}.zip
mkdir ${op_name}_zip

# 复制op_host/ op_kernel/ build_out/custom_*.run文件到指定目录
cp -r ${op_dir}/op_host ${op_name}_zip
cp -r ${op_dir}/op_kernel ${op_name}_zip
cp -r ${op_dir}/build_out/custom_*.run ${op_name}_zip

# 打包文件
zip -r ${op_name}.zip ${op_name}_zip

