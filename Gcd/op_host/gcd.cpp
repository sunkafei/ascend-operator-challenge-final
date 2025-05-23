
#include "gcd_tiling.h"
#include "register/op_def_registry.h"
#include <iostream>
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    GcdTilingData tiling;
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t dim = x1_shape->GetStorageShape().GetDimNum();
    uint32_t n1[5] = {1, 1, 1, 1, 1};
    for (int i = 0; i < dim; ++i) {
        n1[i] = x1_shape->GetStorageShape().GetDim(i);
    }
    const gert::StorageShape* x2_shape = context->GetInputShape(1);
    dim = x2_shape->GetStorageShape().GetDimNum();
    uint32_t n2[5] = {1, 1, 1, 1, 1};
    for (int i = 0; i < dim; ++i) {
        n2[i] = x2_shape->GetStorageShape().GetDim(i);
    }
    uint32_t ny[5] = {1, 1, 1, 1, 1};
    for (int i = 0; i < dim; ++i) {
        ny[i] = std::max(n1[i], n2[i]);
    }
    tiling.set_n1(n1);
    tiling.set_n2(n2);
    tiling.set_ny(ny);
    std::cout << "n1: " << n1[0] << " " << n1[1] << " " << n1[2] << " " << n1[3] << " " << n1[4] << " " << std::endl;
    std::cout << "n2: " << n2[0] << " " << n2[1] << " " << n2[2] << " " << n2[3] << " " << n2[4] << " " << std::endl;
    std::cout << "ny: " << ny[0] << " " << ny[1] << " " << ny[2] << " " << ny[3] << " " << ny[4] << " " << std::endl;
    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class Gcd : public OpDef {
public:
    explicit Gcd(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Gcd);
}
