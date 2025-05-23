#include "kernel_operator.h"
using namespace AscendC;
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t n1, uint32_t n2) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->n1 = n1;
        this->n2 = n2;

        x1Gm.SetGlobalBuffer((__gm__ T*)x1, n1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2, n2);
        yGm.SetGlobalBuffer((__gm__ T*)y, n1);
    }
    __aicore__ inline void Process() {
        for (int i = 0; i < n1; ++i) {
            int64_t a = x1Gm.GetValue(i);
            int64_t b = x2Gm.GetValue(i);
            if (a < 0) {
                a = -a;
            }
            if (b < 0) {
                b = -b;
            }
            while (b) {
                int64_t A = b;
                int64_t B = a % b;
                a = A;
                b = B;
            }
            yGm.SetValue(i, a);
        }
    }

private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t n1, n2;
};
extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    BruteForce<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.n1, tiling_data.n2);
    op.Process();
}