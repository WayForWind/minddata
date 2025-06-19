
#include "OURSdata/dataset/kernels/ir/vision/bounding_box_augment_ir.h"

#include "OURSdata/dataset/engine/serdes.h"
#include "OURSdata/dataset/kernels/image/bounding_box_augment_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
BoundingBoxAugmentOperation::BoundingBoxAugmentOperation(const std::shared_ptr<TensorOperation> &transform, float ratio)
    : transform_(transform), ratio_(ratio) {}

BoundingBoxAugmentOperation::~BoundingBoxAugmentOperation() = default;

std::string BoundingBoxAugmentOperation::Name() const { return kBoundingBoxAugmentOperation; }

Status BoundingBoxAugmentOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorTransforms("BoundingBoxAugment", {transform_}));
  RETURN_IF_NOT_OK(ValidateScalar("BoundingBoxAugment", "ratio", ratio_, {0.0, 1.0}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> BoundingBoxAugmentOperation::Build() {
  std::shared_ptr<BoundingBoxAugmentOp> tensor_op = std::make_shared<BoundingBoxAugmentOp>(transform_->Build(), ratio_);
  return tensor_op;
}

Status BoundingBoxAugmentOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args, transform_args;
  nlohmann::json op_item;
  RETURN_IF_NOT_OK(transform_->to_json(&transform_args));
  op_item["tensor_op_params"] = transform_args;
  op_item["tensor_op_name"] = transform_->Name();
  args["transform"] = op_item;
  args["ratio"] = ratio_;
  *out_json = args;
  return Status::OK();
}

Status BoundingBoxAugmentOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "transform", kBoundingBoxAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "ratio", kBoundingBoxAugmentOperation));
  std::vector<std::shared_ptr<TensorOperation>> transforms;
  std::vector<nlohmann::json> json_operations = {};
  json_operations.push_back(op_params["transform"]);
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(json_operations, &transforms));
  float ratio = op_params["ratio"];
  CHECK_FAIL_RETURN_UNEXPECTED(transforms.size() == 1,
                               "Expect size one of transforms parameter, but got:" + std::to_string(transforms.size()));
  *operation = std::make_shared<vision::BoundingBoxAugmentOperation>(transforms[0], ratio);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
