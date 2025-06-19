

#include "OURSdata/dataset/audio/ir/kernels/gain_ir.h"

#include "OURSdata/dataset/audio/kernels/gain_op.h"

namespace ours {
namespace dataset {
namespace audio {
// GainOperation
GainOperation::GainOperation(float gain_db) : gain_db_(gain_db) {}

Status GainOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> GainOperation::Build() {
  std::shared_ptr<GainOp> tensor_op = std::make_shared<GainOp>(gain_db_);
  return tensor_op;
}

Status GainOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["gain_db"] = gain_db_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
