
#include "OURSdata/dataset/kernels/ir/vision/invert_ir.h"

#include "OURSdata/dataset/kernels/image/invert_op.h"
#if defined(ENABLE_D)
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_invert_op.h"
#endif
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// InvertOperation
InvertOperation::InvertOperation(const std::string &device_target) : device_target_(device_target) {}

InvertOperation::~InvertOperation() = default;

std::string InvertOperation::Name() const { return kInvertOperation; }

Status InvertOperation::ValidateParams() {
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Invert: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> InvertOperation::Build() {
  if (device_target_ == "CPU") {
    return std::make_shared<InvertOp>();
#if defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppInvertOp> dvpp_tensor_op = std::make_shared<DvppInvertOp>();
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "Invert: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status InvertOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["device_target"] = device_target_;
  return Status::OK();
}

Status InvertOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kInvertOperation));
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::InvertOperation>(device_target);
  return Status::OK();
}

MapTargetDevice InvertOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Invert: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
