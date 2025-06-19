

#include "OURSdata/dataset/audio/ir/kernels/highpass_biquad_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/highpass_biquad_op.h"

namespace ours {
namespace dataset {
namespace audio {
// HighpassBiquadOperation
HighpassBiquadOperation::HighpassBiquadOperation(int32_t sample_rate, float cutoff_freq, float Q)
    : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}

Status HighpassBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarNotZero("HighpassBiquad", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateScalar("HighpassBiquad", "Q", Q_, {0, 1.0}, true, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> HighpassBiquadOperation::Build() {
  std::shared_ptr<HighpassBiquadOp> tensor_op = std::make_shared<HighpassBiquadOp>(sample_rate_, cutoff_freq_, Q_);
  return tensor_op;
}

Status HighpassBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["cutoff_freq"] = cutoff_freq_;
  args["Q"] = Q_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
