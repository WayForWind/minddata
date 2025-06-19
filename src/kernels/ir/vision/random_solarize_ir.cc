

#include "OURSdata/dataset/kernels/ir/vision/random_solarize_ir.h"

#include <algorithm>

#include "OURSdata/dataset/kernels/image/random_solarize_op.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomSolarizeOperation.
RandomSolarizeOperation::RandomSolarizeOperation(const std::vector<uint8_t> &threshold)
    : TensorOperation(true), threshold_(threshold) {}

RandomSolarizeOperation::~RandomSolarizeOperation() = default;

std::string RandomSolarizeOperation::Name() const { return kRandomSolarizeOperation; }

Status RandomSolarizeOperation::ValidateParams() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;
  constexpr uint8_t kThresholdMax = 255;

  if (threshold_.size() != size_two) {
    std::string err_msg =
      "RandomSolarize: threshold must be a vector of two values, got: " + std::to_string(threshold_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (auto threshold_value : threshold_) {
    if (threshold_value < 0 || threshold_value > kThresholdMax) {
      std::string err_msg =
        "RandomSolarize: threshold has to be between 0 and 255, got:" + std::to_string(threshold_value);
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (threshold_[dimension_zero] > threshold_[dimension_one]) {
    std::string err_msg = "RandomSolarize: threshold must be passed in a (min, max) format";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSolarizeOperation::Build() {
  std::shared_ptr<RandomSolarizeOp> tensor_op = std::make_shared<RandomSolarizeOp>(threshold_);
  return tensor_op;
}

Status RandomSolarizeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["threshold"] = threshold_;
  return Status::OK();
}

Status RandomSolarizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "threshold", kRandomSolarizeOperation));
  std::vector<uint8_t> threshold = op_params["threshold"];
  *operation = std::make_shared<vision::RandomSolarizeOperation>(threshold);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
