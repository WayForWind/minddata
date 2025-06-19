
#include "OURSdata/dataset/kernels/ir/vision/random_invert_ir.h"

#include "OURSdata/dataset/kernels/image/random_invert_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomInvertOperation
RandomInvertOperation::RandomInvertOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomInvertOperation::~RandomInvertOperation() = default;

std::string RandomInvertOperation::Name() const { return kRandomInvertOperation; }

Status RandomInvertOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomInvert", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomInvertOperation::Build() {
  std::shared_ptr<RandomInvertOp> tensor_op = std::make_shared<RandomInvertOp>(probability_);
  return tensor_op;
}

Status RandomInvertOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomInvertOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomInvertOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomInvertOperation>(prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
