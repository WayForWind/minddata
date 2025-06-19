
#include "OURSdata/dataset/kernels/image/invert_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// only supports RGB images
Status InvertOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->Rank() == kDefaultImageRank,
    "Invert: input tensor is not in shape of <H,W,C>, but got rank: " + std::to_string(input->Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape()[kChannelIndexHWC] == kDefaultImageChannel,
                               "Invert: the number of channels of input tensor is not 3, but got: " +
                                 std::to_string(input->shape()[kChannelIndexHWC]));
  return Invert(input, output);
}
}  // namespace dataset
}  // namespace ours
