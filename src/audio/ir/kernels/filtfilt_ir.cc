

#include "OURSdata/dataset/audio/ir/kernels/filtfilt_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/filtfilt_op.h"

namespace ours {
namespace dataset {
namespace audio {
// FiltfiltOperation
FiltfiltOperation::FiltfiltOperation(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}

Status FiltfiltOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("filtfilt", "a_coeffs", a_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("filtfilt", "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorSameSize("filtfilt", "a_coeffs", a_coeffs_, "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("filtfilt", "a_coeffs[0]", a_coeffs_[0]));
  return Status::OK();
}

std::shared_ptr<TensorOp> FiltfiltOperation::Build() {
  std::shared_ptr<FiltfiltOp> tensor_op = std::make_shared<FiltfiltOp>(a_coeffs_, b_coeffs_, clamp_);
  return tensor_op;
}

Status FiltfiltOperation::to_json(nlohmann::json *out_json) {
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
