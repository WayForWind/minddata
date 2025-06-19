
#include "OURSdata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"

#include "OURSdata/dataset/kernels/image/random_horizontal_flip_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomHorizontalFlipOperation
RandomHorizontalFlipOperation::RandomHorizontalFlipOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomHorizontalFlipOperation::~RandomHorizontalFlipOperation() = default;

std::string RandomHorizontalFlipOperation::Name() const { return kRandomHorizontalFlipOperation; }

Status RandomHorizontalFlipOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomHorizontalFlip", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomHorizontalFlipOperation::Build() {
  std::shared_ptr<RandomHorizontalFlipOp> tensor_op = std::make_shared<RandomHorizontalFlipOp>(probability_);
  return tensor_op;
}

Status RandomHorizontalFlipOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomHorizontalFlipOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomHorizontalFlipOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomHorizontalFlipOperation>(prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
