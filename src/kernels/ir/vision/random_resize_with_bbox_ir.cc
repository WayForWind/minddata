

#include "OURSdata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"

#include <algorithm>

#include "OURSdata/dataset/kernels/image/random_resize_with_bbox_op.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
namespace vision {
// RandomResizeWithBBoxOperation
RandomResizeWithBBoxOperation::RandomResizeWithBBoxOperation(const std::vector<int32_t> &size)
    : TensorOperation(true), size_(size) {}

RandomResizeWithBBoxOperation::~RandomResizeWithBBoxOperation() = default;

std::string RandomResizeWithBBoxOperation::Name() const { return kRandomResizeWithBBoxOperation; }

Status RandomResizeWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("RandomResizeWithBBox", size_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizeWithBBoxOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;

  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[dimension_zero];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == size_two) {
    width = size_[dimension_one];
  }

  std::shared_ptr<RandomResizeWithBBoxOp> tensor_op = std::make_shared<RandomResizeWithBBoxOp>(height, width);
  return tensor_op;
}

Status RandomResizeWithBBoxOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["size"] = size_;
  return Status::OK();
}

Status RandomResizeWithBBoxOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kRandomResizeWithBBoxOperation));
  std::vector<int32_t> size = op_params["size"];
  *operation = std::make_shared<vision::RandomResizeWithBBoxOperation>(size);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace ours
