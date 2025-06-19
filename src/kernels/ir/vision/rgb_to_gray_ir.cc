
#include "OURSdata/dataset/kernels/ir/vision/rgb_to_gray_ir.h"

#include "OURSdata/dataset/kernels/image/rgb_to_gray_op.h"

namespace ours {
namespace dataset {
namespace vision {
RgbToGrayOperation::RgbToGrayOperation() = default;

// RGB2GRAYOperation
RgbToGrayOperation::~RgbToGrayOperation() = default;

std::string RgbToGrayOperation::Name() const { return kRgbToGrayOperation; }

Status RgbToGrayOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbToGrayOperation::Build() { return std::make_shared<RgbToGrayOp>(); }

Status RgbToGrayOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::RgbToGrayOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
