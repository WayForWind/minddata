

#include "OURSdata/dataset/audio/ir/kernels/overdrive_ir.h"

#include "OURSdata/dataset/audio/kernels/overdrive_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"

namespace ours {
namespace dataset {
namespace audio {
OverdriveOperation::OverdriveOperation(float gain, float color) : gain_(gain), color_(color) {}

Status OverdriveOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("Overdrive", "gain", gain_, {0.0f, 100.0f}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Overdrive", "color", color_, {0.0f, 100.0f}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> OverdriveOperation::Build() {
  std::shared_ptr<OverdriveOp> tensor_op = std::make_shared<OverdriveOp>(gain_, color_);
  return tensor_op;
}

Status OverdriveOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["gain"] = gain_;
  args["color"] = color_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
