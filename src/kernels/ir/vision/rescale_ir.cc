
#include "OURSdata/dataset/kernels/ir/vision/rescale_ir.h"

#include "OURSdata/dataset/kernels/image/rescale_op.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
#if defined(ENABLE_OURSdata_PYTHON)
// RescaleOperation
RescaleOperation::RescaleOperation(float rescale, float shift) : rescale_(rescale), shift_(shift) {}

RescaleOperation::~RescaleOperation() = default;

std::string RescaleOperation::Name() const { return kRescaleOperation; }

Status RescaleOperation::ValidateParams() {
  if (rescale_ < 0.0) {
    std::string err_msg = "Rescale: rescale must be greater than or equal to 0, got: " + std::to_string(rescale_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RescaleOperation::Build() {
  std::shared_ptr<RescaleOp> tensor_op = std::make_shared<RescaleOp>(rescale_, shift_);
  return tensor_op;
}

Status RescaleOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["rescale"] = rescale_;
  args["shift"] = shift_;
  *out_json = args;
  return Status::OK();
}

Status RescaleOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "rescale", kRescaleOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "shift", kRescaleOperation));
  float rescale = op_params["rescale"];
  float shift = op_params["shift"];
  *operation = std::make_shared<vision::RescaleOperation>(rescale, shift);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace ours
