
#include "OURSdata/dataset/audio/ir/kernels/sliding_window_cmn_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/sliding_window_cmn_op.h"

namespace ours {
namespace dataset {
namespace audio {
SlidingWindowCmnOperation::SlidingWindowCmnOperation(int32_t cmn_window, int32_t min_cmn_window, bool center,
                                                     bool norm_vars)
    : cmn_window_(cmn_window), min_cmn_window_(min_cmn_window), center_(center), norm_vars_(norm_vars) {}

SlidingWindowCmnOperation::~SlidingWindowCmnOperation() = default;

Status SlidingWindowCmnOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("SlidingWindowCmn", "cmn_window", cmn_window_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("SlidingWindowCmn", "min_cmn_window", min_cmn_window_));

  return Status::OK();
}

Status SlidingWindowCmnOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["cmn_window"] = cmn_window_;
  args["min_cmn_window"] = min_cmn_window_;
  args["center"] = center_;
  args["norm_vars"] = norm_vars_;
  *out_json = args;
  return Status::OK();
}

std::shared_ptr<TensorOp> SlidingWindowCmnOperation::Build() {
  std::shared_ptr<SlidingWindowCmnOp> tensor_op =
    std::make_shared<SlidingWindowCmnOp>(cmn_window_, min_cmn_window_, center_, norm_vars_);
  return tensor_op;
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
