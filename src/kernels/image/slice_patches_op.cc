
#include "OURSdata/dataset/kernels/image/slice_patches_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
const int32_t SlicePatchesOp::kDefNumH = 1;
const int32_t SlicePatchesOp::kDefNumW = 1;
const uint8_t SlicePatchesOp::kDefFillV = 0;
const SliceMode SlicePatchesOp::kDefSliceMode = SliceMode::kPad;

SlicePatchesOp::SlicePatchesOp(int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value)
    : num_height_(num_height), num_width_(num_width), slice_mode_(slice_mode), fill_value_(fill_value) {}

Status SlicePatchesOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(
    input.size() == 1,
    "size of input should be 1, which means 'input_columns' should be 1 when call this operator, but got:" +
      std::to_string(input.size()));

  const auto &in_tensor = input[0];
  auto in_type = in_tensor->type();
  auto in_shape = in_tensor->shape();

  CHECK_FAIL_RETURN_UNEXPECTED(in_type.IsNumeric(), "Input Tensor type should be numeric, got type is non-numeric.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    in_shape.Rank() >= 2, "Rank of input data should be greater than 2, but got:" + std::to_string(in_shape.Rank()));

  std::vector<std::shared_ptr<Tensor>> out;
  RETURN_IF_NOT_OK(SlicePatches(in_tensor, &out, num_height_, num_width_, slice_mode_, fill_value_));
  (void)std::copy(out.begin(), out.end(), std::back_inserter(*output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
