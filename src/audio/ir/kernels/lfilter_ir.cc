

#include "OURSdata/dataset/audio/ir/kernels/lfilter_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/lfilter_op.h"

namespace ours {
namespace dataset {
namespace audio {
// LFilterOperation
LFilterOperation::LFilterOperation(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}

Status LFilterOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("lfilter", "a_coeffs", a_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("lfilter", "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorSameSize("lfilter", "a_coeffs", a_coeffs_, "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("lfilter", "a_coeffs[0]", a_coeffs_[0]));
  return Status::OK();
}

std::shared_ptr<TensorOp> LFilterOperation::Build() {
  std::shared_ptr<LFilterOp> tensor_op = std::make_shared<LFilterOp>(a_coeffs_, b_coeffs_, clamp_);
  return tensor_op;
}

Status LFilterOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["a_coeffs"] = a_coeffs_;
  args["b_coeffs"] = b_coeffs_;
  args["clamp"] = clamp_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
