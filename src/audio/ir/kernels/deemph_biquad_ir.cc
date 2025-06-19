

#include "OURSdata/dataset/audio/ir/kernels/deemph_biquad_ir.h"

#include "OURSdata/dataset/audio/kernels/deemph_biquad_op.h"

namespace ours {
namespace dataset {
namespace audio {
// DeemphBiquadOperation
DeemphBiquadOperation::DeemphBiquadOperation(int32_t sample_rate) : sample_rate_(sample_rate) {}

Status DeemphBiquadOperation::ValidateParams() {
  if ((sample_rate_ != 44100 && sample_rate_ != 48000)) {
    std::string err_msg =
      "DeemphBiquad: sample_rate can only be 44100 or 48000, but got: " + std::to_string(sample_rate_);
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> DeemphBiquadOperation::Build() {
  std::shared_ptr<DeemphBiquadOp> tensor_op = std::make_shared<DeemphBiquadOp>(sample_rate_);
  return tensor_op;
}

Status DeemphBiquadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
