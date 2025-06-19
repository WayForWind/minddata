

#include "OURSdata/dataset/kernels/image/vertical_flip_op.h"

#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"

namespace ours {
namespace dataset {
Status VerticalFlipOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  RETURN_IF_NOT_OK(ValidateImage(input, "VerticalFlip", {1, 2, 3, 4, 5, 6, 10, 11, 12}));
  dsize_t rank = input->shape().Rank();
  if (rank <= kDefaultImageRank) {
    // [H, W] or [H, W, C]
    RETURN_IF_NOT_OK(VerticalFlip(input, output));
  } else {
    // reshape [..., H, W, C] to [N, H, W, C]
    auto input_shape = input->shape();
    dsize_t num_batch = input->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]);
    TensorShape new_shape({num_batch, input_shape[-3], input_shape[-2], input_shape[-1]});
    RETURN_IF_NOT_OK(input->Reshape(new_shape));

    // split [N, H, W, C] to N [H, W, C], and vertical flip N [H, W, C]
    std::vector<std::shared_ptr<Tensor>> input_vector_hwc, output_vector_hwc;
    RETURN_IF_NOT_OK(BatchTensorToTensorVector(input, &input_vector_hwc));
    for (int i = 0; i < num_batch; i++) {
      std::shared_ptr<Tensor> flip;
      RETURN_IF_NOT_OK(VerticalFlip(input_vector_hwc[i], &flip));
      output_vector_hwc.push_back(flip);
    }

    // integrate N [H, W, C] to [N, H, W, C], and reshape [..., H, W, C]
    RETURN_IF_NOT_OK(TensorVectorToBatchTensor(output_vector_hwc, output));
    RETURN_IF_NOT_OK((*output)->Reshape(input_shape));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
