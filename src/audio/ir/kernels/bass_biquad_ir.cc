

#include "OURSdata/dataset/audio/ir/kernels/bass_biquad_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/bass_biquad_op.h"

namespace ours {
namespace dataset {
namespace audio {
// BassBiquadOperation
BassBiquadOperation::BassBiquadOperation(int32_t sample_rate, float gain, float central_freq, float Q)
    : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}

Status BassBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("BassBiquad", "Q", Q_, {0, 1.0}, true, false));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("BassBiquad", "sample_rate", sample_rate_));
  return Status::OK();
}

std::shared_ptr<TensorOp> BassBiquadOperation::Build() {
  std::shared_ptr<BassBiquadOp> tensor_op = std::make_shared<BassBiquadOp>(sample_rate_, gain_, central_freq_, Q_);
  return tensor_op;
}

Status BassBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["gain"] = gain_;
  args["central_freq"] = central_freq_;
  args["Q"] = Q_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
