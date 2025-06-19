

#include "OURSdata/dataset/audio/ir/kernels/contrast_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/contrast_op.h"

namespace ours {
namespace dataset {
namespace audio {
// ContrastOperation
ContrastOperation::ContrastOperation(float enhancement_amount) : enhancement_amount_(enhancement_amount) {}

Status ContrastOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("Contrast", "enhancement_amount", enhancement_amount_, {0, 100.0}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> ContrastOperation::Build() {
  std::shared_ptr<ContrastOp> tensor_op = std::make_shared<ContrastOp>(enhancement_amount_);
  return tensor_op;
}

Status ContrastOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["enhancement_amount"] = enhancement_amount_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
