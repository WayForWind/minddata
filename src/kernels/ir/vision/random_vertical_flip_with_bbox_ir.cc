
#include "OURSdata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"

#include "OURSdata/dataset/kernels/image/random_vertical_flip_with_bbox_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomVerticalFlipWithBBoxOperation
RandomVerticalFlipWithBBoxOperation::RandomVerticalFlipWithBBoxOperation(float prob)
    : TensorOperation(true), probability_(prob) {}

RandomVerticalFlipWithBBoxOperation::~RandomVerticalFlipWithBBoxOperation() = default;

std::string RandomVerticalFlipWithBBoxOperation::Name() const { return kRandomVerticalFlipWithBBoxOperation; }

Status RandomVerticalFlipWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomVerticalFlipWithBBox", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomVerticalFlipWithBBoxOperation::Build() {
  std::shared_ptr<RandomVerticalFlipWithBBoxOp> tensor_op =
    std::make_shared<RandomVerticalFlipWithBBoxOp>(probability_);
  return tensor_op;
}

Status RandomVerticalFlipWithBBoxOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomVerticalFlipWithBBoxOperation::from_json(nlohmann::json op_params,
                                                      std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomVerticalFlipWithBBoxOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
