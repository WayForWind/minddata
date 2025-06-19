

#include "OURSdata/dataset/audio/ir/kernels/mask_along_axis_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/mask_along_axis_op.h"

namespace ours {
namespace dataset {
namespace audio {
MaskAlongAxisOperation::MaskAlongAxisOperation(int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis)
    : mask_start_(mask_start), mask_width_(mask_width), mask_value_(mask_value), axis_(axis) {}

MaskAlongAxisOperation::~MaskAlongAxisOperation() = default;

Status MaskAlongAxisOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MaskAlongAxis", "mask_start", mask_start_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MaskAlongAxis", "mask_width", mask_width_));
  RETURN_IF_NOT_OK(ValidateScalarValue("MaskAlongAxis", "axis", axis_, {1, 2}));
  return Status::OK();
}

std::string MaskAlongAxisOperation::Name() const { return kMaskAlongAxisOperation; }

std::shared_ptr<TensorOp> MaskAlongAxisOperation::Build() {
  std::shared_ptr<MaskAlongAxisOp> tensor_op =
    std::make_shared<MaskAlongAxisOp>(mask_start_, mask_width_, mask_value_, axis_);
  return tensor_op;
}

Status MaskAlongAxisOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["mask_start"] = mask_start_;
  args["mask_width"] = mask_width_;
  args["mask_value"] = mask_value_;
  args["axis"] = axis_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
