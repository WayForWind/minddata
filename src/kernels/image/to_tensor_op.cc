/
#include "OURSdata/dataset/kernels/image/to_tensor_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status ToTensorOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  auto input_type = input->type();
  CHECK_FAIL_RETURN_UNEXPECTED(
    input_type != DataType::DE_UINT32 && input_type != DataType::DE_UINT64 && input_type != DataType::DE_INT64 &&
      !input_type.IsString(),
    "ToTensor: Input includes unsupported data type in [uint32, int64, uint64, string, bytes].");
  // Rescale and convert HWC to CHW format
  return ToTensor(input, output, output_type_);
}
}  // namespace dataset
}  // namespace ours
