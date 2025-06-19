

#include "OURSdata/dataset/kernels/ir/vision/random_sharpness_ir.h"

#include <algorithm>

#include "OURSdata/dataset/kernels/image/random_sharpness_op.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr size_t dimension_zero = 0;
constexpr size_t dimension_one = 1;
constexpr size_t size_two = 2;

// Function to create RandomSharpness.
RandomSharpnessOperation::RandomSharpnessOperation(const std::vector<float> &degrees)
    : TensorOperation(true), degrees_(degrees) {}

RandomSharpnessOperation::~RandomSharpnessOperation() = default;

std::string RandomSharpnessOperation::Name() const { return kRandomSharpnessOperation; }

Status RandomSharpnessOperation::ValidateParams() {
  if (degrees_.size() != size_two || degrees_[dimension_zero] < 0.0 || degrees_[dimension_one] < 0.0) {
    std::string err_msg = "RandomSharpness: degrees must be a vector of two values and greater than or equal to 0.";
    MS_LOG(ERROR) << "RandomSharpness: degrees must be a vector of two values and greater than or equal to 0, got: "
                  << degrees_;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (degrees_[dimension_one] < degrees_[dimension_zero]) {
    std::string err_msg = "RandomSharpness: degrees must be in the format of (min, max).";
    MS_LOG(ERROR) << "RandomSharpness: degrees must be in the format of (min, max), got: " << degrees_;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSharpnessOperation::Build() {
  std::shared_ptr<RandomSharpnessOp> tensor_op =
    std::make_shared<RandomSharpnessOp>(degrees_[dimension_zero], degrees_[dimension_one]);
  return tensor_op;
}

Status RandomSharpnessOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["degrees"] = degrees_;
  return Status::OK();
}

Status RandomSharpnessOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "degrees", kRandomSharpnessOperation));
  std::vector<float> degrees = op_params["degrees"];
  *operation = std::make_shared<vision::RandomSharpnessOperation>(degrees);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
