#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t tile_length = 512;
template<typename T> class BruteForce {
public:
    __aicore__ inline BruteForce() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t n1[5], uint32_t n2[5], uint32_t ny[5]) {
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

        x1Gm.SetGlobalBuffer((__gm__ T*)x1, total1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2, total2);
        yGm.SetGlobalBuffer((__gm__ T*)y, totaly);
    }
    __aicore__ inline void Process() {
        for (uint32_t i0 = 0; i0 < ny[0]; ++i0) {
            for (uint32_t i1 = 0; i1 < ny[1]; ++i1) {
                for (uint32_t i2 = 0; i2 < ny[2]; ++i2) {
                    for (uint32_t i3 = 0; i3 < ny[3]; ++i3) {
                        for (uint32_t i4 = 0; i4 < ny[4]; ++i4) {
                            uint32_t indices[5] = {i0, i1, i2, i3, i4};
                            uint32_t idx1 = 0, idx2 = 0, idxy = 0;
                            for (int j = 0; j < 5; ++j) {
                                idx1 = idx1 * n1[j] + indices[j] % n1[j];
                                idx2 = idx2 * n2[j] + indices[j] % n2[j];
                                idxy = idxy * ny[j] + indices[j];
                            }
                            int64_t a = x1Gm.GetValue(idx1);
                            int64_t b = x2Gm.GetValue(idx2);
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
                            yGm.SetValue(idxy, a);
                        }
                    }
                }
            }
        }
    }

private:
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t n1[5], n2[5], ny[5];
};
template<typename T> class GCDKernalFast {
    public:
        __aicore__ inline GCDKernalFast() {}
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t size, uint32_t total) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            const unsigned num_cores = GetBlockNum();
            unsigned L = GetBlockIdx() * total;
            unsigned R = (GetBlockIdx() + 1) * total;
            if (R > size) {
                R = size;
            }
            this->L = L;
            this->R = R;
            x1Gm.SetGlobalBuffer((__gm__ T*)x1, total);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2, total);
            yGm.SetGlobalBuffer((__gm__ T*)y, total);
            pipe.InitBuffer(Q_x1, BUFFER_NUM, tile_length * sizeof(T));
            pipe.InitBuffer(Q_x2, BUFFER_NUM, tile_length * sizeof(T));
            pipe.InitBuffer(Q_y, BUFFER_NUM, tile_length * sizeof(T));
            pipe.InitBuffer(B_tmp, tile_length * sizeof(T));
            pipe.InitBuffer(B_zero, tile_length * sizeof(T));
            pipe.InitBuffer(B_bits, tile_length * sizeof(uint8_t));
            zero = B_zero.Get<T>();
            Duplicate(zero, T(0), tile_length);
        }
        __aicore__ inline void CopyIn(int32_t pos) {
            LocalTensor<T> x1 = Q_x1.AllocTensor<T>();
            LocalTensor<T> x2 = Q_x2.AllocTensor<T>();
            DataCopy(x1, x1Gm[pos], tile_length);
            DataCopy(x2, x2Gm[pos], tile_length);
            Q_x1.EnQue(x1);
            Q_x2.EnQue(x2);
        }
        __aicore__ inline void CopyOut(int32_t pos) {
            LocalTensor<T> y = Q_y.DeQue<T>();
            DataCopy(yGm[pos], y, tile_length);
            Q_y.FreeTensor(y);
        }
        __aicore__ inline void Process() {
            for (int i = L; i < R; i += tile_length) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
        __aicore__ inline void Compute(int32_t pos) {
            LocalTensor<T> a = Q_x1.DeQue<T>();
            LocalTensor<T> b = Q_x2.DeQue<T>();
            LocalTensor<T> y = Q_y.AllocTensor<T>();
            // for (int i = 0; i < tile_length; ++i) {
            //     T a = x1.GetValue(i);
            //     T b = x2.GetValue(i);
            //     while (b) {
            //         // T A = b;
            //         // T B = a % b;
            //         // a = A;
            //         // b = B;
            //         a = a % b;
            //         a ^= b ^= a ^= b;
            //     }
            //     y.SetValue(i, a > 0 ? a : -a);
            // }
            auto tmp = B_tmp.Get<T>();
            auto bits = B_bits.Get<uint8_t>();
            for (int i = 0; i < 64; ++i) {
                // Div(tmp, a, b, tile_length);
                // Mul(tmp, tmp, b, tile_length);
                // Sub(tmp, a, tmp, tile_length);

                // Compare(bits, tmp, zero, CMPMODE::GT, length);
            }

            Q_x1.FreeTensor(a);
            Q_x2.FreeTensor(b);
            Q_y.EnQue<T>(y);
        }
    private:
        TPipe pipe;
        TBuf<QuePosition::VECCALC> B_bits, B_tmp, B_zero;
        TQue<QuePosition::VECIN, BUFFER_NUM> Q_x1, Q_x2;
        TQue<QuePosition::VECOUT, BUFFER_NUM> Q_y;
        GlobalTensor<T> x1Gm, x2Gm, yGm;
        LocalTensor<T> zero;
        uint32_t L, R;
};
extern "C" __global__ __aicore__ void gcd(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.status == 0) {
        BruteForce<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.n1, tiling_data.n2, tiling_data.ny);
        op.Process();
    }
    else {
        GCDKernalFast<DTYPE_X1> op;
        op.Init(x1, x2, y, tiling_data.size, tiling_data.length);
        op.Process();
    }
}