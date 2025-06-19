
#include "OURSdata/dataset/kernels/ir/vision/random_adjust_sharpness_ir.h"

#include "OURSdata/dataset/kernels/image/random_adjust_sharpness_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomAdjustSharpnessOperation
RandomAdjustSharpnessOperation::RandomAdjustSharpnessOperation(float degree, float prob)
    : degree_(degree), probability_(prob) {}

RandomAdjustSharpnessOperation::~RandomAdjustSharpnessOperation() = default;

std::string RandomAdjustSharpnessOperation::Name() const { return kRandomAdjustSharpnessOperation; }

Status RandomAdjustSharpnessOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("RandomAdjustSharpness", "degree", degree_));
  RETURN_IF_NOT_OK(ValidateProbability("RandomAdjustSharpness", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomAdjustSharpnessOperation::Build() {
  std::shared_ptr<RandomAdjustSharpnessOp> tensor_op = std::make_shared<RandomAdjustSharpnessOp>(degree_, probability_);
  return tensor_op;
}

Status RandomAdjustSharpnessOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["degree"] = degree_;
  args["prob"] = probability_;
  *out_json = args;
  return Status::OK();
}

Status RandomAdjustSharpnessOperation::from_json(nlohmann::json op_params,
                                                 std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "degree", kRandomAdjustSharpnessOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomAdjustSharpnessOperation));
  float degree = op_params["degree"];
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomAdjustSharpnessOperation>(degree, prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
