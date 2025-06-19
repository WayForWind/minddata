
#include "OURSdata/dataset/kernels/ir/vision/decode_ir.h"

#include "OURSdata/dataset/kernels/image/decode_op.h"
#if defined(ENABLE_D)
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_decode_op.h"
#endif
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// DecodeOperation
DecodeOperation::DecodeOperation(bool rgb, const std::string &device_target)
    : rgb_(rgb), device_target_(device_target) {}

DecodeOperation::~DecodeOperation() = default;

std::string DecodeOperation::Name() const { return kDecodeOperation; }

Status DecodeOperation::ValidateParams() {
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Decode: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> DecodeOperation::Build() {
  if (device_target_ == "CPU") {
    return std::make_shared<DecodeOp>(rgb_);
#if defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppDecodeOp>();
#endif
  } else {
    MS_LOG(ERROR) << "Decode: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status DecodeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["rgb"] = rgb_;
  (*out_json)["device_target"] = device_target_;
  return Status::OK();
}

Status DecodeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "rgb", kDecodeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kDecodeOperation));
  bool rgb = op_params["rgb"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::DecodeOperation>(rgb, device_target);
  return Status::OK();
}

MapTargetDevice DecodeOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Decode: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
