
#include "OURSdata/dataset/kernels/ir/vision/equalize_ir.h"

#include "OURSdata/dataset/kernels/image/equalize_op.h"
#if defined(ENABLE_D)
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_equalize_op.h"
#endif
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// EqualizeOperation
EqualizeOperation::EqualizeOperation(const std::string &device_target) : device_target_(device_target) {}

EqualizeOperation::~EqualizeOperation() = default;

std::string EqualizeOperation::Name() const { return kEqualizeOperation; }

Status EqualizeOperation::ValidateParams() {
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Equalize: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> EqualizeOperation::Build() {
  if (device_target_ == "CPU") {
    return std::make_shared<EqualizeOp>();
#if defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppEqualizeOp>();
#endif
  } else {
    MS_LOG(ERROR) << "Equalize: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status EqualizeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status EqualizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kEqualizeOperation));
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::EqualizeOperation>(device_target);
  return Status::OK();
}

MapTargetDevice EqualizeOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Equalize: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
