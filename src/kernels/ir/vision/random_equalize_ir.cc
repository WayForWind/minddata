

#include "OURSdata/dataset/kernels/ir/vision/random_equalize_ir.h"

#include "OURSdata/dataset/kernels/image/random_equalize_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomEqualizeOperation
RandomEqualizeOperation::RandomEqualizeOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomEqualizeOperation::~RandomEqualizeOperation() = default;

std::string RandomEqualizeOperation::Name() const { return kRandomEqualizeOperation; }

Status RandomEqualizeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomEqualize", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomEqualizeOperation::Build() {
  std::shared_ptr<RandomEqualizeOp> tensor_op = std::make_shared<RandomEqualizeOp>(probability_);
  return tensor_op;
}

Status RandomEqualizeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomEqualizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomEqualizeOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomEqualizeOperation>(prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
