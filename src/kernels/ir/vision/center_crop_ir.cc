
#include "OURSdata/dataset/kernels/ir/vision/center_crop_ir.h"

#include "OURSdata/dataset/kernels/image/center_crop_op.h"

#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
CenterCropOperation::CenterCropOperation(const std::vector<int32_t> &size) : size_(size) {}

CenterCropOperation::~CenterCropOperation() = default;

std::string CenterCropOperation::Name() const { return kCenterCropOperation; }

Status CenterCropOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("CenterCrop", size_));
  return Status::OK();
}

std::shared_ptr<TensorOp> CenterCropOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified crop_width.
  constexpr size_t kSizeSize = 2;
  if (size_.size() == kSizeSize) {
    crop_width = size_[1];
  }

  std::shared_ptr<CenterCropOp> tensor_op = std::make_shared<CenterCropOp>(crop_height, crop_width);
  return tensor_op;
}

Status CenterCropOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["size"] = size_;
  return Status::OK();
}

Status CenterCropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kCenterCropOperation));
  std::vector<int32_t> size = op_params["size"];
  *operation = std::make_shared<CenterCropOperation>(size);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
