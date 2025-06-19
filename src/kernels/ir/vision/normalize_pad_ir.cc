

#include "OURSdata/dataset/kernels/ir/vision/normalize_pad_ir.h"

#include <algorithm>

#include "OURSdata/dataset/kernels/image/normalize_pad_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// NormalizePadOperation
NormalizePadOperation::NormalizePadOperation(const std::vector<float> &mean, const std::vector<float> &std,
                                             const std::string &dtype, bool is_hwc)
    : mean_(mean), std_(std), dtype_(dtype), is_hwc_(is_hwc) {}

NormalizePadOperation::~NormalizePadOperation() = default;

std::string NormalizePadOperation::Name() const { return kNormalizePadOperation; }

Status NormalizePadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorMeanStd("NormalizePad", mean_, std_));
  if (dtype_ != "float32" && dtype_ != "float16") {
    std::string err_msg = "NormalizePad: dtype must be float32 or float16, but got: " + dtype_;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> NormalizePadOperation::Build() {
  return std::make_shared<NormalizePadOp>(mean_, std_, dtype_, is_hwc_);
}

Status NormalizePadOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["mean"] = mean_;
  args["std"] = std_;
  args["dtype"] = dtype_;
  args["is_hwc"] = is_hwc_;
  *out_json = args;
  return Status::OK();
}

Status NormalizePadOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "mean", kNormalizePadOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "std", kNormalizePadOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "dtype", kNormalizePadOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "is_hwc", kNormalizePadOperation));
  std::vector<float> mean = op_params["mean"];
  std::vector<float> std = op_params["std"];
  std::string dtype = op_params["dtype"];
  bool is_hwc = op_params["is_hwc"];
  *operation = std::make_shared<vision::NormalizePadOperation>(mean, std, dtype, is_hwc);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
