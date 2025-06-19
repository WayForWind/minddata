
#include "OURSdata/dataset/kernels/image/random_equalize_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RandomEqualizeOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Check input
  RETURN_IF_NOT_OK(ValidateImageRank("RandomEqualize", input->Rank()));
  if (input->Rank() == kDefaultImageRank) {
    int num_channels = static_cast<int>(input->shape()[kChannelIndexHWC]);
    if (num_channels != kMinImageChannel && num_channels != kDefaultImageChannel) {
      RETURN_STATUS_UNEXPECTED("RandomEqualize: input image is not in channel of 1 or 3, but got: " +
                               std::to_string(input->shape()[kChannelIndexHWC]));
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type() == DataType(DataType::DE_UINT8),
    "RandomEqualize: input image is not in type of uint8, but got: " + input->type().ToString());
  if (distribution_(random_generator_)) {
    return Equalize(input, output);
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
