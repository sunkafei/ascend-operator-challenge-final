#include "kernel_operator.h"
using namespace AscendC;
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t n1[5], uint32_t n2[5], uint32_t ny[5], uint32_t bsize) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t total1 = 1, total2 = 1, totaly = 1;
        for (int i = 0; i < 5; ++i) {
            this->n1[i] = n1[i];
            this->n2[i] = n2[i];
            this->ny[i] = ny[i];
            total1 *= n1[i];
            total2 *= n2[i];
            totaly *= ny[i];
        }
        this->m1[4] = this->m2[4] = this->my[4] = 1;
        for(int i = 3; i >= 0; i--){
            this->m1[i] = this->m1[i + 1] * this->n1[i + 1];
            this->m2[i] = this->m2[i + 1] * this->n2[i + 1];
            this->my[i] = this->my[i + 1] * this->ny[i + 1];
        }
        this->st = bsize * GetBlockIdx();
        this->ed = this->st + bsize;
        this->ed = this->ed > totaly ? totaly : this->ed;

        x1Gm.SetGlobalBuffer((__gm__ T*)x1, total1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2, total2);
        yGm.SetGlobalBuffer((__gm__ T*)y, totaly);
    }
    __aicore__ inline void Process() {
        // for (uint32_t i0 = 0; i0 < ny[0]; ++i0) {
        //     for (uint32_t i1 = 0; i1 < ny[1]; ++i1) {
        //         for (uint32_t i2 = 0; i2 < ny[2]; ++i2) {
        //             for (uint32_t i3 = 0; i3 < ny[3]; ++i3) {
        //                 for (uint32_t i4 = 0; i4 < ny[4]; ++i4) {
        //                     uint32_t indices[5] = {i0, i1, i2, i3, i4};
        //                     uint32_t idx1 = 0, idx2 = 0, idxy = 0;
        //                     for (int j = 0; j < 5; ++j) {
        //                         idx1 = idx1 * n1[j] + indices[j] % n1[j];
        //                         idx2 = idx2 * n2[j] + indices[j] % n2[j];
        //                         idxy = idxy * ny[j] + indices[j];
        //                     }
        //                     int64_t a = x1Gm.GetValue(idx1);
        //                     int64_t b = x2Gm.GetValue(idx2);
        //                     if (a < 0) {
        //                         a = -a;
        //                     }
        //                     if (b < 0) {
        //                         b = -b;
        //                     }
        //                     while (b) {
        //                         int64_t A = b;
        //                         int64_t B = a % b;
        //                         a = A;
        //                         b = B;
        //                     }
        //                     yGm.SetValue(idxy, a);
        //                 }
        //             }
        //         }
        //     }
        // }
        for(uint32_t i = this->st; i < this->ed; i++){
            uint32_t indices[5] = {i / my[0], i / my[1] % ny[1], i / my[2] % ny[2], i / my[3] % ny[3], i % ny[4]};
            uint32_t idx1 = 0, idx2 = 0, idxy = 0;
            for (int j = 0; j < 5; ++j) {
                idx1 = idx1 * n1[j] + indices[j] % n1[j];
                idx2 = idx2 * n2[j] + indices[j] % n2[j];
                idxy = idxy * ny[j] + indices[j];
            }
            T a = x1Gm.GetValue(idx1);
            T b = x2Gm.GetValue(idx2);
            if (a < 0) {
                a = -a;
            }
            if (b < 0) {
                b = -b;
            }
            while (b) {
                T A = b;
                T B = a % b;
                a = A;
                b = B;
            }
            yGm.SetValue(idxy, a);
        }
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm);
    }

private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t n1[5], n2[5], ny[5];
    uint32_t m1[5], m2[5], my[5];
    uint32_t st, ed;
};
extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    BruteForce<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.n1, tiling_data.n2, tiling_data.ny, tiling_data.bsize);
    op.Process();
}