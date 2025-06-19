
#include "OURSdata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"

#include "OURSdata/dataset/kernels/image/random_horizontal_flip_with_bbox_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomHorizontalFlipWithBBoxOperation
RandomHorizontalFlipWithBBoxOperation::RandomHorizontalFlipWithBBoxOperation(float probability)
    : TensorOperation(true), probability_(probability) {}

RandomHorizontalFlipWithBBoxOperation::~RandomHorizontalFlipWithBBoxOperation() = default;

std::string RandomHorizontalFlipWithBBoxOperation::Name() const { return kRandomHorizontalFlipWithBBoxOperation; }

Status RandomHorizontalFlipWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomHorizontalFlipWithBBox", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomHorizontalFlipWithBBoxOperation::Build() {
  std::shared_ptr<RandomHorizontalFlipWithBBoxOp> tensor_op =
    std::make_shared<RandomHorizontalFlipWithBBoxOp>(probability_);
  return tensor_op;
}

Status RandomHorizontalFlipWithBBoxOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomHorizontalFlipWithBBoxOperation::from_json(nlohmann::json op_params,
                                                        std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomHorizontalFlipWithBBoxOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
