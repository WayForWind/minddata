
#include "OURSdata/dataset/kernels/ir/vision/vertical_flip_ir.h"

#include "OURSdata/dataset/kernels/image/vertical_flip_op.h"
#if defined(ENABLE_D)
#include "OURSdata/dataset/kernels/image/dvpp/ascend910b/dvpp_vertical_flip_op.h"
#endif
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// VerticalFlipOperation
VerticalFlipOperation::VerticalFlipOperation(const std::string &device_target) : device_target_(device_target) {}

VerticalFlipOperation::~VerticalFlipOperation() = default;

std::string VerticalFlipOperation::Name() const { return kVerticalFlipOperation; }

Status VerticalFlipOperation::ValidateParams() {
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "VerticalFlip: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> VerticalFlipOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<VerticalFlipOp> tensor_op = std::make_shared<VerticalFlipOp>();
    return tensor_op;
#if defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppVerticalFlipOp> dvpp_tensor_op = std::make_shared<DvppVerticalFlipOp>();
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "VerticalFlip: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status VerticalFlipOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status VerticalFlipOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kVerticalFlipOperation));
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::VerticalFlipOperation>(device_target);
  return Status::OK();
}

MapTargetDevice VerticalFlipOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "VerticalFlip: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
