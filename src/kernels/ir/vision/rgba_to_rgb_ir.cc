
#include "OURSdata/dataset/kernels/ir/vision/rgba_to_rgb_ir.h"

#include "OURSdata/dataset/kernels/image/rgba_to_rgb_op.h"

namespace ours {
namespace dataset {
namespace vision {
// RgbaToRgbOperation.
RgbaToRgbOperation::RgbaToRgbOperation() = default;

RgbaToRgbOperation::~RgbaToRgbOperation() = default;

std::string RgbaToRgbOperation::Name() const { return kRgbaToRgbOperation; }

Status RgbaToRgbOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToRgbOperation::Build() {
  std::shared_ptr<RgbaToRgbOp> tensor_op = std::make_shared<RgbaToRgbOp>();
  return tensor_op;
}

Status RgbaToRgbOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  *operation = std::make_shared<vision::RgbaToRgbOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
