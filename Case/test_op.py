import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

from typing import Optional, Tuple
case_data = {
    'case1': {
        'x1':np.random.randint(-100, 100, [32]).astype(np.int16),
        'x2':np.random.randint(-100, 100, [32]).astype(np.int16)
    },
    'case2': {
        'x1':np.random.randint(-100, 100, [32]).astype(np.int16),
        'x2':np.random.randint(-100, 100, [1]).astype(np.int16)
    },
    'case3': {
        'x1':np.random.randint(-100, 100, [77, 31, 9]).astype(np.int16),
        'x2':np.random.randint(-100, 100, [9]).astype(np.int16)
    },
    'case5': {
        'x1':np.random.randint(-1e9, 1e9, [512, 64, 1, 2, 1]).astype(np.int64),
        'x2':np.random.randint(-1e9, 1e9, [512, 64, 1, 2, 1]).astype(np.int64)
    }
}

def verify_result(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4
    
    minimum = 10e-10
    result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss)  # 计算绝对误差
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)  # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseNmae='case'+str(num)
        x1 = torch.from_numpy(case_data[caseNmae]["x1"])
        x2 = torch.from_numpy(case_data[caseNmae]["x2"])
        golden = torch.gcd(x1, x2).numpy()
        # 修改输入
        output = custom_ops_lib.custom_op(x1.npu(), x2.npu(), int(num))
        if output is None:
            print(f"{caseNmae} execution timed out!")
        else:
            output = output.cpu().numpy()
            if verify_result(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
