

#include "OURSdata/dataset/audio/ir/kernels/magphase_ir.h"

#include "OURSdata/dataset/audio/ir/validators.h"
#include "OURSdata/dataset/audio/kernels/magphase_op.h"

namespace ours {
namespace dataset {
namespace audio {
MagphaseOperation::MagphaseOperation(float power) : power_(power) {}

Status MagphaseOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Magphase", "power", power_));
  return Status::OK();
}

std::shared_ptr<TensorOp> MagphaseOperation::Build() { return std::make_shared<MagphaseOp>(power_); }

Status MagphaseOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["power"] = power_;
  *out_json = args;
  return Status::OK();
}

Status MagphaseOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "power", kMagphaseOperation));
  float power = op_params["power"];
  *operation = std::make_shared<audio::MagphaseOperation>(power);
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace ours
