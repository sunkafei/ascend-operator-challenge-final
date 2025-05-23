/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


at::Tensor my_op_impl_npu(const at::Tensor& self, const at::Tensor& x2, int64_t caseNum) {
    at::Tensor result = at::empty_like(self);
    auto ruond = 1;
    if (caseNum == 5)
    {
        ruond = 50;
    }
    for (size_t i = 0; i < ruond; i++)
    {
        EXEC_NPU_CMD(aclnnGcd, self, x2, result);
    }
    
    
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor self, Tensor x2, int caseNum) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.gcd");
}
