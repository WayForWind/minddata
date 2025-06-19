

#include "OURSdata/dataset/kernels/ir/vision/uniform_aug_ir.h"

#include <algorithm>

#include "OURSdata/dataset/engine/serdes.h"
#include "OURSdata/dataset/kernels/image/uniform_aug_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// UniformAugOperation
UniformAugOperation::UniformAugOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms,
                                         int32_t num_ops)
    : transforms_(transforms), num_ops_(num_ops) {}

UniformAugOperation::~UniformAugOperation() = default;

std::string UniformAugOperation::Name() const { return kUniformAugOperation; }

Status UniformAugOperation::ValidateParams() {
  // transforms
  RETURN_IF_NOT_OK(ValidateVectorTransforms("UniformAugment", transforms_));
  // num_ops
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("UniformAugment", "num_ops", num_ops_));
  if (num_ops_ > transforms_.size()) {
    std::string err_msg =
      "UniformAugment: num_ops must be less than or equal to transforms size, but got: " + std::to_string(num_ops_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> UniformAugOperation::Build() {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;
  (void)std::transform(
    transforms_.begin(), transforms_.end(), std::back_inserter(tensor_ops),
    [](const std::shared_ptr<TensorOperation> &op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  std::shared_ptr<UniformAugOp> tensor_op = std::make_shared<UniformAugOp>(tensor_ops, num_ops_);
  return tensor_op;
}

Status UniformAugOperation::to_json(nlohmann::json *out_json) {
  CHECK_FAIL_RETURN_UNEXPECTED(out_json != nullptr, "parameter out_json is nullptr");
  nlohmann::json args;
  std::vector<nlohmann::json> transforms;
  for (const auto &op : transforms_) {
    nlohmann::json op_item, op_args;
    RETURN_IF_NOT_OK(op->to_json(&op_args));
    op_item["tensor_op_params"] = op_args;
    op_item["tensor_op_name"] = op->Name();
    transforms.push_back(op_item);
  }
  args["transforms"] = transforms;
  args["num_ops"] = num_ops_;
  *out_json = args;
  return Status::OK();
}

Status UniformAugOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "transforms", kUniformAugOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "num_ops", kUniformAugOperation));
  std::vector<std::shared_ptr<TensorOperation>> transforms = {};
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(op_params["transforms"], &transforms));
  int32_t num_ops = op_params["num_ops"];
  *operation = std::make_shared<vision::UniformAugOperation>(transforms, num_ops);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
