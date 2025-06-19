
#include "OURSdata/dataset/kernels/image/rgb_to_bgr_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status RgbToBgrOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  auto input_type = input->type();
  CHECK_FAIL_RETURN_UNEXPECTED(
    input_type != DataType::DE_UINT32 && input_type != DataType::DE_UINT64 && input_type != DataType::DE_INT64 &&
      !input_type.IsString(),
    "RgbToBgr: Input includes unsupported data type in [uint32, int64, uint64, string, bytes].");
  return RgbToBgr(input, output);
}
}  // namespace dataset
}  // namespace ours
