
#include "OURSdata/dataset/kernels/ir/vision/adjust_gamma_ir.h"

#include "OURSdata/dataset/kernels/image/adjust_gamma_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// AdjustGammaOperation
AdjustGammaOperation::AdjustGammaOperation(float gamma, float gain) : gamma_(gamma), gain_(gain) {}

Status AdjustGammaOperation::ValidateParams() {
  // gamma
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustGamma", "gamma", gamma_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustGammaOperation::Build() {
  std::shared_ptr<AdjustGammaOp> tensor_op = std::make_shared<AdjustGammaOp>(gamma_, gain_);
  return tensor_op;
}

Status AdjustGammaOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["gamma"] = gamma_;
  args["gain"] = gain_;
  *out_json = args;
  return Status::OK();
}

Status AdjustGammaOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "gamma", kAdjustGammaOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "gain", kAdjustGammaOperation));
  float gamma = op_params["gamma"];
  float gain = op_params["gain"];
  *operation = std::make_shared<vision::AdjustGammaOperation>(gamma, gain);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
