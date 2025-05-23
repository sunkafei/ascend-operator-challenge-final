
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GcdTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n1);
  TILING_DATA_FIELD_DEF(uint32_t, n2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd, GcdTilingData)
}
