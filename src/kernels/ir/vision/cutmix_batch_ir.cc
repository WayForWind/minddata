
#include "OURSdata/dataset/kernels/ir/vision/cutmix_batch_ir.h"

#include "OURSdata/dataset/kernels/image/cutmix_batch_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// CutMixBatchOperation
CutMixBatchOperation::CutMixBatchOperation(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {}

CutMixBatchOperation::~CutMixBatchOperation() = default;

std::string CutMixBatchOperation::Name() const { return kCutMixBatchOperation; }

Status CutMixBatchOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("CutMixBatch", "alpha", alpha_));
  RETURN_IF_NOT_OK(ValidateProbability("CutMixBatch", prob_));
  if (image_batch_format_ != ImageBatchFormat::kNHWC && image_batch_format_ != ImageBatchFormat::kNCHW) {
    std::string err_msg = "CutMixBatch: Invalid ImageBatchFormat, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CutMixBatchOperation::Build() {
  std::shared_ptr<CutMixBatchOp> tensor_op = std::make_shared<CutMixBatchOp>(image_batch_format_, alpha_, prob_);
  return tensor_op;
}

Status CutMixBatchOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["image_batch_format"] = image_batch_format_;
  args["alpha"] = alpha_;
  args["prob"] = prob_;
  *out_json = args;
  return Status::OK();
}

Status CutMixBatchOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "image_batch_format", kCutMixBatchOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "alpha", kCutMixBatchOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kCutMixBatchOperation));
  auto image_batch = static_cast<ImageBatchFormat>(op_params["image_batch_format"]);
  float alpha = op_params["alpha"];
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::CutMixBatchOperation>(image_batch, alpha, prob);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
