
#include "OURSdata/dataset/kernels/ir/vision/random_vertical_flip_ir.h"

#include "OURSdata/dataset/kernels/image/random_vertical_flip_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomVerticalFlipOperation
RandomVerticalFlipOperation::RandomVerticalFlipOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomVerticalFlipOperation::~RandomVerticalFlipOperation() = default;

std::string RandomVerticalFlipOperation::Name() const { return kRandomVerticalFlipOperation; }

Status RandomVerticalFlipOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomVerticalFlip", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomVerticalFlipOperation::Build() {
  std::shared_ptr<RandomVerticalFlipOp> tensor_op = std::make_shared<RandomVerticalFlipOp>(probability_);
  return tensor_op;
}

Status RandomVerticalFlipOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomVerticalFlipOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomVerticalFlipOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomVerticalFlipOperation>(prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
