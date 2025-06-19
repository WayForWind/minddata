

#include "OURSdata/dataset/kernels/ir/vision/adjust_hue_ir.h"

#include "OURSdata/dataset/kernels/image/adjust_hue_op.h"
#if defined(ENABLE_D)
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_adjust_hue_op.h"
#endif
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// AdjustHueOperation
AdjustHueOperation::AdjustHueOperation(float hue_factor, const std::string &device_target)
    : hue_factor_(hue_factor), device_target_(device_target) {}

Status AdjustHueOperation::ValidateParams() {
  // hue_factor
  RETURN_IF_NOT_OK(ValidateScalar("AdjustHue", "hue_factor", hue_factor_, {-0.5, 0.5}, false, false));
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "AdjustHue: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustHueOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<AdjustHueOp> tensor_op = std::make_shared<AdjustHueOp>(hue_factor_);
    return tensor_op;
#if defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppAdjustHueOp>(hue_factor_);
#endif
  } else {
    MS_LOG(ERROR) << "AdjustHue: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status AdjustHueOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["hue_factor"] = hue_factor_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status AdjustHueOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "hue_factor", kAdjustHueOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kAdjustHueOperation));
  float hue_factor = op_params["hue_factor"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::AdjustHueOperation>(hue_factor, device_target);
  return Status::OK();
}

MapTargetDevice AdjustHueOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "AdjustHue: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
