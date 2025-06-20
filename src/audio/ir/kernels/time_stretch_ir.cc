
#include "OURSdata/dataset/audio/ir/kernels/time_stretch_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/time_stretch_op.h"

namespace ours {
namespace dataset {
namespace audio {
// TimeStretchOperation
TimeStretchOperation::TimeStretchOperation(float hop_length, int n_freq, float fixed_rate)
    : hop_length_(hop_length), n_freq_(n_freq), fixed_rate_(fixed_rate) {}

TimeStretchOperation::~TimeStretchOperation() = default;

std::string TimeStretchOperation::Name() const { return kTimeStretchOperation; }

Status TimeStretchOperation::ValidateParams() {
  //  param check
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("TimeStretch", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("TimeStretch", "n_freq", n_freq_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("TimeStretch", "fixed_rate", fixed_rate_));
  return Status::OK();
}

std::shared_ptr<TensorOp> TimeStretchOperation::Build() {
  std::shared_ptr<TimeStretchOp> tensor_op = std::make_shared<TimeStretchOp>(hop_length_, n_freq_, fixed_rate_);
  return tensor_op;
}

Status TimeStretchOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["hop_length"] = hop_length_;
  args["n_freq"] = n_freq_;
  args["fixed_rate"] = fixed_rate_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
