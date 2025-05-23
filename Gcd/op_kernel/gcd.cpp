#include "kernel_operator.h"
using namespace AscendC;

constexpr uint64_t pre[65] = {0, 1ll,3ll,5ll,11ll,17ll,39ll,65ll,139ll,261ll,531ll,1025ll,2095ll,4097ll,8259ll,16405ll,32907ll,65537ll,131367ll,262145ll,524827ll,1048645ll,2098179ll,4194305ll,8390831ll,16777233ll,33558531ll,67109125ll,134225995ll,268435457ll,536887863ll,1073741825ll,2147516555ll,4294968325ll,8590000131ll,17179869265ll,34359871791ll,68719476737ll,137439215619ll,274877911045ll,549756338843ll,1099511627777ll,2199024312423ll,4398046511105ll,8796095120395ll,17592186061077ll,35184376283139ll,70368744177665ll,140737496778927ll,281474976710721ll,562949970199059ll,1125899906908165ll,2251799847243787ll,4503599627370497ll,9007199321981223ll,18014398509483025ll,36028797153190091ll,72057594038190085ll,144115188344291331ll,288230376151711745ll,576460752840837695ll,1152921504606846977ll,2305843010287435779ll,4611686018428436805ll,9223372039002292363};

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
        __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t size, uint32_t length) {
            ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
            const unsigned num_cores = GetBlockNum();
            unsigned L = GetBlockIdx() * length;
            unsigned R = (GetBlockIdx() + 1) * length;
            if (R > size) {
                R = size;
            }
            this->L = 0;
            this->R = R - L;
            x1Gm.SetGlobalBuffer((__gm__ T*)x1 + L, length);
            x2Gm.SetGlobalBuffer((__gm__ T*)x2 + L, length);
            yGm.SetGlobalBuffer((__gm__ T*)y + L, length);
        }
        __aicore__ inline void Process() {
            for (int i = L; i < R; ++i) {
                T a = x1Gm.GetValue(i);
                T b = x2Gm.GetValue(i);
                a = (a > 0 ? a : -a);
                b = (b > 0 ? b : -b);
                if (a == 0){
                    yGm.SetValue(i, b);
                }
                else if (b == 0) {
                    yGm.SetValue(i, a);
                }
                else {
                    T shift = ScalarGetSFFValue<1>(a | b);
                    a >>= ScalarGetSFFValue<1>(a);
                    do {
                        b >>= ScalarGetSFFValue<1>(b);
                        if(a <= 64 && b <= 64){
                            a = 64 - ScalarCountLeadingZero(pre[a] & pre[b]);
                            break;
                        }
                        if (a > b) {
                            a ^= b ^= a ^= b;
                        }
                        b -= a;
                    } while (b);
                    yGm.SetValue(i, a << shift);
                }
            }
            //AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(yGm);
            //DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(yGm);
        }
    
    private:
        GlobalTensor<T> x1Gm, x2Gm, yGm;
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