

#include "OURSdata/dataset/audio/ir/kernels/db_to_amplitude_ir.h"

#include "OURSdata/dataset/audio/kernels/db_to_amplitude_op.h"

namespace ours {
namespace dataset {
namespace audio {
// DBToAmplitudeOperation
DBToAmplitudeOperation::DBToAmplitudeOperation(float ref, float power) : ref_(ref), power_(power) {}

Status DBToAmplitudeOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DBToAmplitudeOperation::Build() {
  std::shared_ptr<DBToAmplitudeOp> tensor_op = std::make_shared<DBToAmplitudeOp>(ref_, power_);
  return tensor_op;
}

Status DBToAmplitudeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["ref"] = ref_;
  args["power"] = power_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
