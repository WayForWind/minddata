

#include "OURSdata/dataset/audio/ir/kernels/dc_shift_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/dc_shift_op.h"

namespace ours {
namespace dataset {
namespace audio {
// DCShiftOperation
DCShiftOperation::DCShiftOperation(float shift, float limiter_gain) : shift_(shift), limiter_gain_(limiter_gain) {}

Status DCShiftOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("DCShift", "shift", shift_, {-2.0, 2.0}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> DCShiftOperation::Build() {
  std::shared_ptr<DCShiftOp> tensor_op = std::make_shared<DCShiftOp>(shift_, limiter_gain_);
  return tensor_op;
}

Status DCShiftOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["shift"] = shift_;
  args["limiter_gain"] = limiter_gain_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
