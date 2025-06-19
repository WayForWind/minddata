/

#include "OURSdata/dataset/audio/ir/kernels/band_biquad_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/band_biquad_op.h"

namespace ours {
namespace dataset {
namespace audio {
// BandBiquadOperation
BandBiquadOperation::BandBiquadOperation(int32_t sample_rate, float central_freq, float Q, bool noise)
    : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), noise_(noise) {}

Status BandBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("BandBiquad", "Q", Q_, {0, 1.0}, true, false));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("BandBiquad", "sample_rate", sample_rate_));
  return Status::OK();
}

std::shared_ptr<TensorOp> BandBiquadOperation::Build() {
  std::shared_ptr<BandBiquadOp> tensor_op = std::make_shared<BandBiquadOp>(sample_rate_, central_freq_, Q_, noise_);
  return tensor_op;
}

Status BandBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["central_freq"] = central_freq_;
  args["Q"] = Q_;
  args["noise"] = noise_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
