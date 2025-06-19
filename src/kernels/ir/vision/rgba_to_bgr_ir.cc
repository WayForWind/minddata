
#include "OURSdata/dataset/kernels/ir/vision/rgba_to_bgr_ir.h"

#include "OURSdata/dataset/kernels/image/rgba_to_bgr_op.h"

namespace ours {
namespace dataset {
namespace vision {
// RgbaToBgrOperation.
RgbaToBgrOperation::RgbaToBgrOperation() = default;

RgbaToBgrOperation::~RgbaToBgrOperation() = default;

std::string RgbaToBgrOperation::Name() const { return kRgbaToBgrOperation; }

Status RgbaToBgrOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToBgrOperation::Build() {
  std::shared_ptr<RgbaToBgrOp> tensor_op = std::make_shared<RgbaToBgrOp>();
  return tensor_op;
}

Status RgbaToBgrOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::RgbaToBgrOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
