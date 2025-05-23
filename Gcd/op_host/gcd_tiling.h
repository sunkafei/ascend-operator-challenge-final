
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GcdTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, n1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, n2);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, ny);
  TILING_DATA_FIELD_DEF(uint32_t, status);
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, length);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd, GcdTilingData)
}
