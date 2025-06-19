

#include "OURSdata/dataset/audio/ir/kernels/riaa_biquad_ir.h"

#include "OURSdata/dataset/audio/kernels/riaa_biquad_op.h"
#include "OURSdata/dataset/audio/ir/validators.h"

namespace ours {
namespace dataset {
namespace audio {
RiaaBiquadOperation::RiaaBiquadOperation(int32_t sample_rate) : sample_rate_(sample_rate) {}

Status RiaaBiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarValue("RiaaBiquad", "sample_rate", sample_rate_, {44100, 48000, 88200, 96000}));
  return Status::OK();
}

std::shared_ptr<TensorOp> RiaaBiquadOperation::Build() {
  std::shared_ptr<RiaaBiquadOp> tensor_op = std::make_shared<RiaaBiquadOp>(sample_rate_);
  return tensor_op;
}

Status RiaaBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
