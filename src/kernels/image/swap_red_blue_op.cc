
#include "OURSdata/dataset/kernels/image/swap_red_blue_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status SwapRedBlueOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  return SwapRedAndBlue(input, output);
}
}  // namespace dataset
}  // namespace ours
