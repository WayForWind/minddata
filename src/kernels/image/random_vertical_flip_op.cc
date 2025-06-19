

#include "OURSdata/dataset/kernels/image/random_vertical_flip_op.h"

#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status RandomVerticalFlipOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  const auto output_count = input.size();
  output->resize(output_count);

  for (const auto &image : input) {
    RETURN_IF_NOT_OK(ValidateImageDtype("RandomVerticalFlip", image->type()));
    RETURN_IF_NOT_OK(ValidateImageRank("RandomVerticalFlip", image->Rank()));
  }

  if (distribution_(random_generator_)) {
    for (size_t i = 0; i < input.size(); i++) {
      RETURN_IF_NOT_OK(VerticalFlip(input[i], &(*output)[i]));
    }
    return Status::OK();
  }
  *output = input;
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
