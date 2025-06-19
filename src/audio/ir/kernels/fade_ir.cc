
#include "OURSdata/dataset/audio/ir/kernels/fade_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/fade_op.h"

namespace ours {
namespace dataset {
namespace audio {
FadeOperation::FadeOperation(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape)
    : fade_in_len_(fade_in_len), fade_out_len_(fade_out_len), fade_shape_(fade_shape) {}

Status FadeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Fade", "fade_in_len", fade_in_len_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Fade", "fade_out_len", fade_out_len_));
  return Status::OK();
}

std::shared_ptr<TensorOp> FadeOperation::Build() {
  std::shared_ptr<FadeOp> tensor_op = std::make_shared<FadeOp>(fade_in_len_, fade_out_len_, fade_shape_);
  return tensor_op;
}

Status FadeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["fade_in_len"] = fade_in_len_;
  args["fade_out_len"] = fade_out_len_;
  args["fade_shape"] = fade_shape_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
