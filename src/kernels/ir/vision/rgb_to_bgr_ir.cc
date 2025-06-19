
#include "OURSdata/dataset/kernels/ir/vision/rgb_to_bgr_ir.h"

#include "OURSdata/dataset/kernels/image/rgb_to_bgr_op.h"

namespace ours {
namespace dataset {
namespace vision {
RgbToBgrOperation::RgbToBgrOperation() = default;

// RGB2BGROperation
RgbToBgrOperation::~RgbToBgrOperation() = default;

std::string RgbToBgrOperation::Name() const { return kRgbToBgrOperation; }

Status RgbToBgrOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbToBgrOperation::Build() { return std::make_shared<RgbToBgrOp>(); }

Status RgbToBgrOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::RgbToBgrOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
