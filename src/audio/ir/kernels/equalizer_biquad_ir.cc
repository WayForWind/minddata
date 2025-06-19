

#include "OURSdata/dataset/audio/ir/kernels/equalizer_biquad_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/equalizer_biquad_op.h"

namespace ours {
namespace dataset {
namespace audio {
EqualizerBiquadOperation::EqualizerBiquadOperation(int32_t sample_rate, float center_freq, float gain, float Q)
    : sample_rate_(sample_rate), center_freq_(center_freq), gain_(gain), Q_(Q) {}

Status EqualizerBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarNotZero("EqualizerBiquad", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateScalar("EqualizerBiquad", "Q", Q_, {0, 1.0}, true, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> EqualizerBiquadOperation::Build() {
  std::shared_ptr<EqualizerBiquadOp> tensor_op =
    std::make_shared<EqualizerBiquadOp>(sample_rate_, center_freq_, gain_, Q_);
  return tensor_op;
}

Status EqualizerBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["center_freq"] = center_freq_;
  args["gain"] = gain_;
  args["Q"] = Q_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
