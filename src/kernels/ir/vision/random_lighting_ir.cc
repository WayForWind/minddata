
#include "OURSdata/dataset/kernels/ir/vision/random_lighting_ir.h"

#include "OURSdata/dataset/kernels/image/random_lighting_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomLightingOperation.
RandomLightingOperation::RandomLightingOperation(float alpha) : TensorOperation(true), alpha_(alpha) {}

RandomLightingOperation::~RandomLightingOperation() = default;

std::string RandomLightingOperation::Name() const { return kRandomLightingOperation; }

Status RandomLightingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("RandomLighting", "alpha", alpha_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomLightingOperation::Build() {
  std::shared_ptr<RandomLightingOp> tensor_op = std::make_shared<RandomLightingOp>(alpha_);
  return tensor_op;
}

Status RandomLightingOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["alpha"] = alpha_;
  *out_json = args;
  return Status::OK();
}

Status RandomLightingOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "alpha", kRandomLightingOperation));
  float alpha = op_params["alpha"];
  *operation = std::make_shared<vision::RandomLightingOperation>(alpha);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
