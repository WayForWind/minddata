

#include "OURSdata/dataset/audio/ir/kernels/compute_deltas_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/compute_deltas_op.h"

namespace ours {
namespace dataset {
namespace audio {
ComputeDeltasOperation::ComputeDeltasOperation(int32_t win_length, BorderType pad_mode)
    : win_length_(win_length), pad_mode_(pad_mode) {}

std::shared_ptr<TensorOp> ComputeDeltasOperation::Build() {
  return std::make_shared<ComputeDeltasOp>(win_length_, pad_mode_);
}

Status ComputeDeltasOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["win_length"] = win_length_;
  args["pad_mode"] = pad_mode_;
  *out_json = args;
  return Status::OK();
}

Status ComputeDeltasOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("ComputeDeltas", "win_length", win_length_, {3}, false));
  if (pad_mode_ != BorderType::kConstant && pad_mode_ != BorderType::kEdge && pad_mode_ != BorderType::kReflect &&
      pad_mode_ != BorderType::kSymmetric) {
    std::string err_msg = "ComputeDeltas: invalid pad_mode value, check the optional value of BorderType.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
