
#include "OURSdata/dataset/kernels/image/random_resize_op.h"

#include <random>

#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RandomResizeOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  const auto output_count = input.size();
  output->resize(output_count);
  auto interpolation_random_resize = static_cast<InterpolationMode>(distribution_(random_generator_));
  std::shared_ptr<TensorOp> resize_op = std::make_shared<ResizeOp>(size1_, size2_, interpolation_random_resize);
  for (size_t i = 0; i < input.size(); i++) {
    RETURN_IF_NOT_OK(resize_op->Compute(input[i], &(*output)[i]));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
