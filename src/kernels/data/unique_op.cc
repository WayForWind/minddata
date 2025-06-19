

#include "OURSdata/dataset/kernels/data/unique_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
Status UniqueOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1,
                               "Unique: only support 1D input, got rank: " + std::to_string(input.size()));

  auto in_tensor = input[0];
  auto in_tensor_shape = in_tensor->shape();
  auto in_tensor_type = in_tensor->type();

  CHECK_FAIL_RETURN_UNEXPECTED(in_tensor_type.IsNumeric(),
                               "Unique: only support numeric datatype of input, got string.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    in_tensor_shape.Rank() >= 2,
    "Unique: input must be at least 2-D in order to do unique op, got rank:" + std::to_string(in_tensor_shape.Rank()));
  CHECK_FAIL_RETURN_UNEXPECTED(in_tensor->Size() <= std::numeric_limits<int32_t>::max(),
                               "Unique: Unique does not support size of input tensor large than: " +
                                 std::to_string(std::numeric_limits<int32_t>::max()) +
                                 ", got:" + std::to_string(in_tensor->Size()));

  RETURN_IF_NOT_OK(in_tensor->Reshape(TensorShape({in_tensor->Size()})));

  std::shared_ptr<Tensor> out;
  std::shared_ptr<Tensor> out_idx;
  std::shared_ptr<Tensor> out_cnt;

  RETURN_IF_NOT_OK(Unique(in_tensor, &out, &out_idx, &out_cnt));
  output->push_back(out);
  output->push_back(out_idx);
  output->push_back(out_cnt);
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
